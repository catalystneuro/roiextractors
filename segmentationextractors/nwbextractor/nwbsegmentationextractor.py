import os
import uuid
from datetime import datetime

import pynwb
from dateutil.tz import tzlocal
import numpy as np
import re
import yaml
from segmentationextractors.segmentationextractor import SegmentationExtractor
from lazy_ops import DatasetView
from hdmf.data_utils import DataChunkIterator
try:
    from pynwb import NWBHDF5IO, TimeSeries, NWBFile
    from pynwb.base import Images
    from pynwb.image import GrayscaleImage
    from pynwb.ophys import ImageSegmentation, Fluorescence, OpticalChannel, TwoPhotonSeries, DfOverF
    from pynwb.device import Device

    HAVE_NWB = True
except ModuleNotFoundError:
    HAVE_NWB = False


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


class NwbSegmentationExtractor(SegmentationExtractor):
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
        self.image_masks = None
        self.pixel_masks = None
        self._roi_locs = None
        self._accepted_list = None
        # with NWBHDF5IO(filepath, mode='r+') as io:
        io = NWBHDF5IO(filepath, mode='r+')
        nwbfile = io.read()
        self.nwbfile = nwbfile
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
        if 'image_mask' in ps.colnames:
            self.image_masks = DatasetView(ps['image_mask'].data).lazy_transpose([1, 2, 0])
        if 'pixel_mask' in ps.colnames:
            # Extract pixel_mask/background:
            px_list = [ps['pixel_mask'][e] for e in range(ps['pixel_mask'].data.shape[0])]
            temp = np.empty((1, 4))
            for v, b in enumerate(px_list):
                temp = np.append(temp, np.append(b, v * np.ones([b.shape[0], 1]), axis=1), axis=0)
            self.pixel_masks = temp[1::, :]
        if 'RoiCentroid' in ps.colnames:
            self._roi_locs = ps['RoiCentroid']
        if 'Accepted' in ps.colnames:
            self._accepted_list = ps['Accepted'].data[:]
        # Extract Image dimensions:

        # Extract roi_response:
        self.roi_resp_dict = dict()
        self._roi_names = [_nwbchildren_name[val]
                      for val, i in enumerate(_nwbchildren_type) if i == 'RoiResponseSeries']
        if not self._roi_names:
            raise Exception('no ROI response series found')
        else:
            for roi_name in self._roi_names:
                self.roi_resp_dict[roi_name] = mod['Fluorescence'].get_roi_response_series(roi_name)
        self.roi_response = self.roi_resp_dict[self._roi_names[0]]

        # Extract samp_freq:
        self._samp_freq = self.roi_response.rate
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
            self.extimage_dims = None
        else:
            self.raw_movie_file_location = \
                nwbfile.all_children()[_image_series_exist[0]].external_file[:][0]
            self.extimage_dims = \
                nwbfile.all_children()[_image_series_exist[0]].dimension

        # property name/data extraction:
        self._property_name_exist = [
            i for i in ps.colnames if i not in ['image_mask', 'pixel_mask']]
        self.property_vals = []
        for i in self._property_name_exist:
            self.property_vals.append(np.array(ps[i].data))

        #Extracting stores images as GrayscaleImages:
        self._greyscaleimages = [_nwbchildren_name[f] for f, u in enumerate(_nwbchildren_type) if u == 'GrayscaleImage']

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
            return np.where(self._accepted_list==1)[0].tolist()

    @property
    def rejected_list(self):
        return [a for a in self.roi_idx if a not in set(self.accepted_list)]

    @property
    def roi_locs(self):
        if self._roi_locs is None:
            return None
        else:
            return self._roi_locs.data[:].T.tolist()

    @property
    def num_of_frames(self):
        extracted_signals = self.roi_response.data
        return extracted_signals.shape[1]

    @property
    def samp_freq(self):
        return self._samp_freq

    def get_traces(self, ROI_ids=None, start_frame=None, end_frame=None, name=None):
        if name is None:
            name = self._roi_names[0]
            print(f'returning traces for {name}')
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
        return np.array([self.roi_resp_dict[name].data[int(i), start_frame:end_frame] for i in range(self.no_rois)])

    def get_traces_info(self):
        roi_resp_dict = dict()
        for i in self._roi_names:
            roi_resp_dict[i] = self.get_traces(name=i)
        return roi_resp_dict

    def get_num_frames(self):
        return self.roi_response.data.shape[1]

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
        if self.pixel_masks is None:
            return None
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
        if self.image_masks is None:
            return None
        if ROI_ids is None:
            ROI_idx_ = range(self.get_num_rois())
        else:
            ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
            ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
        return np.array([self.image_masks[:, :, int(i)].T for i in ROI_idx_]).T

    def get_images(self):
        imag_dict = {i.name: np.array(i.data) for i in self.nwbfile.all_children() if i.name in self._greyscaleimages}
        _ = {i.name: i for i in self.nwbfile.all_children() if i.name in self._greyscaleimages}
        if imag_dict:
            parent_name = _[self._greyscaleimages[0]].parent.name
            return {parent_name: imag_dict}
        else:
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
    def write_segmentation(segext_obj, savepath, metadata_dict=None, **kwargs):
        source_path = segext_obj.filepath
        if isinstance(metadata_dict, str):
            with open(metadata_dict, 'r') as f:
                metadata = yaml.safe_load(f)

        #NWB file:
        nwbfile_args = dict(identifier=str(uuid.uuid4()), )
        nwbfile_args.update(**metadata_dict['NWBFile'])
        nwbfile = NWBFile(**nwbfile_args)

        #Subject:
        nwbfile.subject = pynwb.file.Subject(**metadata_dict['Subject'])

        #Device:
        if isinstance(metadata_dict['Ophys']['Device'], list):
            for devices in metadata_dict['Ophys']['Device']:
                nwbfile.create_device(**devices)
        else:
            nwbfile.create_device(**metadata_dict['Ophys']['Device'])

        #Processing Module:
        ophys_mod = nwbfile.create_processing_module('Ophys',
                                                     'contains optical physiology processed data')

        #ImageSegmentation:
        image_segmentation = ImageSegmentation(metadata_dict['Ophys']['ImageSegmentation']['name'])
        ophys_mod.add_data_interface(image_segmentation)

        #OPtical Channel:
        channel_names = segext_obj.get_channel_names()
        input_args=[dict(name=i) for i in channel_names]
        for j,i in enumerate(metadata_dict['Ophys']['ImagingPlane']['optical_channel']):
            input_args[j].update(**i)
        optical_channels=[OpticalChannel(input_args[j]) for j,i in enumerate(channel_names)]

        #Imaging Plane:
        input_kwargs = [dict(
            name='ImagingPlane',
            description='no description',
            device=i,
            excitation_lambda=np.nan,
            optical_channel=optical_channels,
            imaging_rate=1.0,
            indicator='unknown',
            location='unknown'
        ) for i in nwbfile.devices.values()]
        [input_kwargs[j].update(**i) for j,i in enumerate(metadata_dict['Ophys']['ImagingPlane'])]#update with metadata
        imaging_planes = [nwbfile.create_imaging_plane(i) for i in input_kwargs]

        #Plane Segmentation:
        input_kwargs = [dict(
            name='PlaneSegmentation',
            description='output from segmenting my favorite imaging plane',
            imaging_plane=i
        ) for i in imaging_planes]
        [input_kwargs[j].update(**i)
         for j,i in enumerate(metadata_dict['Ophys']['ImageSegmentation']['plane_segmentations'])]  # update with metadata
        ps = [image_segmentation.create_plane_segmentation(i) for i in input_kwargs]

        # ROI add:
        pixel_mask_exist = segext_obj.get_pixel_masks() is not None
        for i, roiid in enumerate(segext_obj.roi_idx):
            if pixel_mask_exist:
                [ps_loop.add_roi(id=roiid,
                           pixel_mask=segext_obj.get_pixel_masks(ROI_ids=[roiid])[:, 0:-1])
                for ps_loop in ps]
            else:
                [ps_loop.add_roi(id=roiid,
                                 image_mask=segext_obj.get_image_masks(ROI_ids=[roiid]))
                for ps_loop in ps]

        # adding columns to ROI table:
        [ps_loop.add_column(name='RoiCentroid',
                            description='x,y location of centroid of the roi in image_mask',
                            data=np.array(segext_obj.get_roi_locations()).T)
         for ps_loop in ps]
        accepted = np.zeros(segext_obj.no_rois)
        for j, i in enumerate(segext_obj.roi_idx):
            if i in segext_obj.accepted_list:
                accepted[j] = 1
        [ps_loop.add_column(name='Accepted',
                            description='1 if ROi was accepted or 0 if rejected as a cell during segmentation operation',
                            data=accepted)
         for ps_loop in ps]

        #Fluorescence Traces:
        input_kwargs = dict(
            rois=ps[0].create_roi_table_region('NeuronROIs', region=list(range(segext_obj.no_rois))),
            starting_time=0.0,
            rate=segext_obj.get_sampling_frequency(),
            unit='lumens'
        )
        container_type = [i for i in metadata_dict['Ophys'].keys() if i in ['DfOverF','Fluorescence']][0]
        f_container = eval(container_type+'()')
        ophys_mod.add_data_interface(f_container)
        for i in metadata_dict['Ophys'][container_type]['roi_response_series']:
            i.update(**input_kwargs,data=segext_obj.get_traces_info()[i['name']].T)
            f_container.create_roi_response_series(**i)

        #create Two Photon Series:
        input_kwargs = [dict(
            name='TwoPhotonSeries',
            description='no description',
            imaging_plane=i,
            external_file=[segext_obj.get_movie_location()],
            format='external',
            rate=segext_obj.get_sampling_frequency(),
            starting_time=0.0,
            starting_frame=[0],
            dimension=segext_obj.image_dims
        ) for i in imaging_planes]
        [input_kwargs[j].update(**i) for j,i in enumerate(metadata_dict['Ophys']['TwoPhotonSeries'])]
        tps = [nwbfile.add_acquisition(TwoPhotonSeries(**i)) for i in input_kwargs]

        #adding images:
        images_dict = segext_obj.get_images()
        if images_dict is not None:
            for img_set_name, img_set in images_dict.items():
                images = Images(img_set_name)
                for img_name, img_no in img_set.items():
                    images.add_image(GrayscaleImage(name=img_name, data=img_no))
                ophys_mod.add(images)

        # saving NWB file:
        with NWBHDF5IO(savepath, 'w') as io:
            io.write(nwbfile)

        with NWBHDF5IO(savepath, 'r') as io:
            io.read()