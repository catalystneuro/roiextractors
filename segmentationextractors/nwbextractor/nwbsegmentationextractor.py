import os
import uuid
from datetime import datetime
from dateutil.tz import tzlocal
import numpy as np
import re
import yaml
from ..segmentationextractor import SegmentationExtractor
from lazy_ops import DatasetView
from hdmf.data_utils import DataChunkIterator
from nwb_conversion_tools import gui
from nwb_conversion_tools import NWBConverter
# TODO: put this within the save method
# from suite2p.io.nwb import save_nwb
try:
    from pynwb import NWBHDF5IO, TimeSeries, NWBFile
    from pynwb.base import Images
    from pynwb.image import GrayscaleImage
    from pynwb.ophys import ImageSegmentation, Fluorescence, OpticalChannel, TwoPhotonSeries
    from pynwb.device import Device

    HAVE_NWB = True
except ModuleNotFoundError:
    HAVE_NWB = False
try:
    from nwb_conversion_tools.ophys.sima.simaconverter import Sima2NWB
    from nwb_conversion_tools.ophys.suite2p.suite2pconverter import Suite2p2NWB
    from nwb_conversion_tools.ophys.schnitzerlab.extractconverter import Extract2NWB
    from nwb_conversion_tools.ophys.schnitzerlab.cnmfeconverter import Cnmfe2NWB
    from nwb_conversion_tools.gui.nwb_conversion_gui import nwb_conversion_gui
except:
    print('calling from nwbconversiontools')


def check_nwb_install():
    assert HAVE_NWB, "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"


def set_dynamic_table_property(dynamic_table, ids, row_ids, property_name, values, index=False,
                               default_value=np.nan, description='no description'):
    check_nwb_install()
    if not isinstance(row_ids, list) or not all(isinstance(x, int) for x in row_ids):
        raise TypeError("'ids' must be a list of integers")
    if any([i not in ids for i in row_ids]):
        raise ValueError("'ids' contains values outside the range of existing ids")
    if not isinstance(property_name, str):
        raise TypeError("'property_name' must be a string")
    if len(row_ids) != len(values) and index is False:
        raise ValueError("'ids' and 'values' should be lists of same size")

    if index is False:
        if property_name in dynamic_table:
            for (row_id, value) in zip(row_ids, values):
                dynamic_table[property_name].data[ids.index(row_id)] = value
        else:
            col_data = [default_value] * len(ids)  # init with default val
            for (row_id, value) in zip(row_ids, values):
                col_data[ids.index(row_id)] = value
            dynamic_table.add_column(
                name=property_name,
                description=description,
                data=col_data,
                index=index
            )
    else:
        if property_name in dynamic_table:
            raise NotImplementedError
        else:
            dynamic_table.add_column(
                name=property_name,
                description=description,
                data=values,
                index=index
            )


def get_dynamic_table_property(dynamic_table, *, row_ids=None, property_name):
    all_row_ids = list(dynamic_table.id[:])
    if row_ids is None:
        row_ids = all_row_ids
    return [dynamic_table[property_name][all_row_ids.index(x)] for x in row_ids]


def iter_datasetvieww(datasetview_obj):
    '''
    Generator to return a row of the array each time it is called.
    This will be wrapped with a DataChunkIterator class.

    Parameters
    ----------
    datasetview_obj: DatasetView
        2-D array to iteratively write to nwb.
    '''

    for i in range(datasetview_obj.shape[0]):
        curr_data = datasetview_obj[i]
        yield curr_data
    return


class NwbSegmentationExtractor(SegmentationExtractor, NWBConverter):
    '''
    Class used to extract data from the NWB data format. Also implements a
    static method to write any format specific object to NWB.
    '''

    def __init__(self, filepath, optical_channel_name=None,
                 imaging_plane_name=None, image_series_name=None,
                 processing_module_name=None,
                 neuron_roi_response_series_name=None,
                 background_roi_response_series_name=None):
        '''
        Parameters
        ----------
        filepath: str
            The location of the folder containing dataset.nwb file.
        optical_channel_name: str(optional)
            optical channel to extract data from
        imaging_plane_name: str(optional)
            imaging plane to extract data from
        image_series_name: str(optional)
            imaging series to extract data from
        processing_module_name: str(optional)
            processing module to extract data from
        neuron_roi_response_series_name: str(optional)
            name of roi response series to extract data from
        background_roi_response_series_name: str(optional)
            name of background roi response series to extract data from
        '''
        check_nwb_install()
        if not os.path.exists(filepath):
            raise Exception('file does not exist')

        self.filepath = filepath

        # with NWBHDF5IO(filepath, mode='r+') as io:
        self.io = NWBHDF5IO(filepath, mode='r+')
        nwbfile = self.io.read()
        _nwbchildren_type = [type(i).__name__ for i in nwbfile.all_children()]
        _nwbchildren_name = [i.name for i in nwbfile.all_children()]
        _procssing_module = [_nwbchildren_name[f]
                             for f, u in enumerate(_nwbchildren_type) if u == 'ProcessingModule']
        mod = nwbfile.processing[_procssing_module[0]]
        if len(_procssing_module) > 1:
            print('multiple processing modules found, picking the first one')
        elif not mod:
            raise Exception('no processing module found')

        # Extract image_mask/background:
        _plane_segmentation_exist = [i for i, e in enumerate(
            _nwbchildren_type) if e == 'PlaneSegmentation']
        if not _plane_segmentation_exist:
            print('could not find a plane segmentation to contain image mask')
        else:
            ps = nwbfile.all_children()[_plane_segmentation_exist[0]]
        # self.image_masks = np.moveaxis(np.array(ps['image_mask'].data), [0, 1, 2], [2, 0, 1])
        self.image_masks = DatasetView(ps['image_mask'].data).lazy_transpose([1, 2, 0])
        self.raw_images = self.image_masks

        # Extract pixel_mask/background:
        px_list = [ps['pixel_mask'][e] for e in range(ps['pixel_mask'].data.shape[0])]
        temp = np.empty((1, 4))
        for v, b in enumerate(px_list):
            temp = np.append(temp, np.append(b, v * np.ones([b.shape[0], 1]), axis=1), axis=0)
        self.pixel_masks = temp[1::, :]
        # Extract Image dimensions:
        self.extimage_dims = self.image_masks.shape[0:2]

        # Extract roi_response:
        _roi_exist = [_nwbchildren_name[val]
                      for val, i in enumerate(_nwbchildren_type) if i == 'RoiResponseSeries']
        if not _roi_exist:
            raise Exception('no ROI response series found')
        else:
            rrs_neurons = mod['Fluorescence'].get_roi_response_series(_roi_exist[0])
            self.roi_response = DatasetView(rrs_neurons.data)
            self._no_background_comps = 1
            self.roi_response_bk = np.nan * np.ones(
                [self._no_background_comps, self.roi_response.shape[1]])
            if len(_roi_exist) > 1:
                rrs_bk = mod['Fluorescence'].get_roi_response_series(_roi_exist[1])
                self.roi_response_bk = np.array(rrs_bk.data)
                self._no_background_comps = self.roi_response_bk.shape[0]

        # Extract planesegmentation dictionary values:
        _new_columns = [i for i in ps.colnames if i not in ['image_mask', 'pixel_mask']]
        for i in _new_columns:
            setattr(self, i, np.array(ps[i].data))

        # Extract samp_freq:
        self._samp_freq = rrs_neurons.rate
        self.total_time = rrs_neurons.rate * rrs_neurons.num_samples
        # Extract no_rois/ids:
        self._roi_idx = np.array(ps.id.data)

        # Imaging plane:
        _optical_channel_exist = [i for i, e in enumerate(
            _nwbchildren_type) if e == 'OpticalChannel']
        if not _optical_channel_exist:
            self.channel_names = ['OpticalChannel']
        else:
            self.channel_names = []
            for i in _optical_channel_exist:
                self.channel_names.append(nwbfile.all_children()[i].name)
        # Movie location:
        _image_series_exist = [i for i, e in enumerate(
            _nwbchildren_type) if e == 'TwoPhotonSeries']
        if not _image_series_exist:
            self.raw_movie_file_location = None
        else:
            self.raw_movie_file_location = \
                str(nwbfile.all_children()[_image_series_exist[0]].external_file[:])

        # property name/data extraction:
        self._property_name_exist = [
            i for i in ps.colnames if i not in ['image_mask', 'pixel_mask']]
        self.property_vals = []
        for i in self._property_name_exist:
            self.property_vals.append(np.array(ps[i].data))

    @property
    def image_dims(self):
        return list(self.extimage_dims)

    @property
    def no_rois(self):
        return self.roi_idx.size

    @property
    def roi_idx(self):
        return self._roi_idx

    @property
    def accepted_list(self):
        if self._accepted_list is None:
            return list(range(self.no_rois))
        else:
            return self._accepted_list

    @property
    def rejected_list(self):
        return [a for a in range(self.no_rois) if a not in set(self.accepted_list)]

    @property
    def roi_locs(self):
        no_ROIs = self.no_rois
        raw_images = self.raw_images
        roi_location = np.ndarray([2, no_ROIs], dtype='int')
        for i in range(no_ROIs):
            temp = np.where(raw_images[:, :, i] == np.amax(raw_images[:, :, i]))
            roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
        return roi_location

    @property
    def num_of_frames(self):
        extracted_signals = self.roi_response
        return extracted_signals.shape[1]

    @property
    def samp_freq(self):
        return self._samp_freq

    def get_traces(self, ROI_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames() + 1
        if ROI_ids is None:
            ROI_idx_ = range(self.get_num_rois())
        else:
            ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
            ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
        return np.array([self.roi_response[int(i), start_frame:end_frame] for i in ROI_idx_])

    def get_num_frames(self):
        return self.roi_response.shape[1]

    def get_sampling_frequency(self):
        return self.samp_freq

    def get_roi_locations(self, ROI_ids=None):
        if ROI_ids is None:
            return self.roi_locs
        else:
            ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
            ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
            return self.roi_locs[:, ROI_idx_]

    def get_roi_ids(self):
        return self.roi_idx

    def get_num_rois(self):
        return self.no_rois

    def get_pixel_masks(self, ROI_ids=None):
        if ROI_ids is None:
            ROI_idx_ = self.roi_idx
        else:
            ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
            ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
        temp = np.empty((1, 4))
        for i, roiid in enumerate(ROI_idx_):
            temp = \
                np.append(temp, self.pixel_masks[self.pixel_masks[:, 3] == roiid, :], axis=0)
        return temp[1::, :]

    def get_image_masks(self, ROI_ids=None):
        if ROI_ids is None:
            ROI_idx_ = range(self.get_num_rois())
        else:
            ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
            ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
        return np.array([self.raw_images[:, :, int(i)].T for i in ROI_idx_]).T

    def get_images(self):
        return None

    def get_movie_framesize(self):
        return self.image_dims

    def get_movie_location(self):
        return self.raw_movie_file_location

    def get_channel_names(self):
        return self.channel_names

    def get_num_channels(self):
        return len(self.channel_names)

    def get_property_data(self, property_name):
        ret_val = []
        for j, i in enumerate(property_name):
            if i in self._property_name_exist:
                ret_val.append(self.property_vals[j])
            else:
                raise Exception('enter valid property name. Names found: {}'.format(
                    self._property_name_exist))
        return ret_val

    @staticmethod
    def write_recording(segmentation_extractor_obj, nwb_filename, metadict, gui=False, **kwargs_fields):
        class_name = str(type(segmentation_extractor_obj))
        datapath = os.path.dirname(segmentation_extractor_obj.filepath)
        if isinstance(metadict, dict):
            with open(datapath+fr'\{class_name}_metafile.yml','r') as f:
                metafile = yaml.safe_dump(metadict)
        else:
            metafile = metadict
        if 'Sima' in class_name:
            converter_class = Sima2NWB
        elif 'Suite2p' in class_name:
            converter_class = Suite2p2NWB
        elif 'Extract' in class_name:
            converter_class = Extract2NWB
        elif 'Cnmfe' in class_name:
            converter_class = Cnmfe2NWB
        else:
            raise Exception('invalid object type')
        if os.path.isdir(segmentation_extractor_obj.filepath):
            path_type='folder'
        else:
            path_type='file'
        source_paths = dict(extract_path=dict(type=path_type, path=segmentation_extractor_obj.filepath))
        if gui:
            nwb_conversion_gui(
                metafile=metafile,
                conversion_class=converter_class,
                source_paths=source_paths,
                kwargs_fields=kwargs_fields
            )
        else:
            conv_obj = converter_class(segmentation_extractor_obj, None, metafile)
            conv_obj.run_conversion()
            conv_obj.save(nwb_filename)
