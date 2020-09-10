import os
import uuid
import numpy as np
import yaml
from datetime import datetime
from collections import abc
from lazy_ops import DatasetView
from pathlib import Path
from warnings import warn
from ...imagingextractor import ImagingExtractor
from ...segmentationextractor import SegmentationExtractor
from ...multisegmentationextractor import MultiSegmentationExtractor
from ...extraction_tools import PathType, check_get_frames_args, check_get_videos_args, _pixel_mask_extractor
from copy import deepcopy

try:
    from pynwb import NWBHDF5IO, TimeSeries, NWBFile
    from pynwb.base import Images
    from pynwb.image import GrayscaleImage
    from pynwb.ophys import ImageSegmentation, Fluorescence, OpticalChannel, TwoPhotonSeries, DfOverF
    from pynwb.file import Subject
    from pynwb.device import Device
    from hdmf.data_utils import DataChunkIterator

    HAVE_NWB = True
except ModuleNotFoundError:
    HAVE_NWB = False


def dict_recursive_update(metadata_base, metadata_input):
    return_dict = deepcopy(metadata_base)
    for base_key, base_val in return_dict.items():
        if metadata_input.get(base_key):
            if isinstance(base_val,dict):
                return_dict[base_key] = dict_recursive_update(base_val, metadata_input[base_key])
            elif isinstance(base_val,list):
                if isinstance(metadata_input[base_key], list):
                    for base_val_num,(a,b) in enumerate(zip(base_val, metadata_input[base_key])):
                        if isinstance(a, dict) and isinstance(b, dict):
                            return_dict[base_key][base_val_num] = dict_recursive_update(a, b)
                        elif isinstance(a, (int, float, str)) and isinstance(b, (int, float, str)):
                            return_dict[base_key][base_val_num] = b
                else:
                    continue
            else:
                return_dict[base_key] = metadata_input[base_key]
    return return_dict


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


class NwbImagingExtractor(ImagingExtractor):
    """
    Class used to extract data from the NWB data format. Also implements a
    static method to write any format specific object to NWB.
    """

    extractor_name = 'NwbImaging'
    installed = HAVE_NWB # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the Nwb Extractor run:\n\n pip install pynwb\n\n"  # error message when not installed

    def __init__(self, file_path, optical_channel_name=None,
                 imaging_plane_name=None, image_series_name=None,
                 processing_module_name=None,
                 neuron_roi_response_series_name=None,
                 background_roi_response_series_name=None):
        """
        Parameters
        ----------
        file_path: str
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
        """
        assert HAVE_NWB, self.installation_mesg
        ImagingExtractor.__init__(self)

    @check_get_frames_args
    def get_frames(self, frame_idxs):
        assert np.all(frame_idxs < self.get_num_frames())
        planes = np.zeros((len(frame_idxs), self._size_x, self._size_y))
        for i, frame_idx in enumerate(frame_idxs):
            plane = self._video[frame_idx]
            planes[i] = plane
        return planes

    def get_image_size(self):
        return [self._size_x, self._size_y]

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_dtype(self):
        return self._video.dtype

    def get_channel_names(self):
        """List of  channels in the recoding.

        Returns
        -------
        channel_names: list
            List of strings of channel names
        """
        return self._channel_names

    def get_num_channels(self):
        """Total number of active channels in the recording

        Returns
        -------
        no_of_channels: int
            integer count of number of channels
        """
        return self._num_channels

    @staticmethod
    def write_imaging(imaging, savepath):
        pass


class NwbSegmentationExtractor(SegmentationExtractor):

    extractor_name = 'NwbSegmentationExtractor'
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path):
        """
        Creating NwbSegmentationExtractor object from nwb file
        Parameters
        ----------
        file_path: str
            .nwb file location
        """
        check_nwb_install()
        SegmentationExtractor.__init__(self)
        if not os.path.exists(file_path):
            raise Exception('file does not exist')

        self.file_path = file_path
        self.image_masks = None
        self._roi_locs = None
        self._accepted_list = None
        self._rejected_list = None
        self._io = NWBHDF5IO(file_path, mode='r')
        self.nwbfile = self._io.read()

        ophys = self.nwbfile.processing.get('ophys')
        if ophys is None:
            raise Exception('could not find ophys processing module in nwbfile')
        else:
            # Extract roi_response:
            fluorescence = None
            dfof = None
            any_roi_response_series_found = False
            if 'Fluorescence' in ophys.data_interfaces:
                fluorescence = ophys.data_interfaces['Fluorescence']
            if 'DfOverF' in ophys.data_interfaces:
                dfof = ophys.data_interfaces['DfOverF']
            if fluorescence is None and dfof is None:
                raise Exception('could not find Fluorescence/DfOverF module in nwbfile')
            for trace_name in ['RoiResponseSeries', 'Dff', 'Neuropil', 'Deconvolved']:
                trace_name_segext = 'raw' if trace_name == 'RoiResponseSeries' else trace_name.lower()
                container = dfof if trace_name=='Dff' else fluorescence
                if container is not None and trace_name in container.roi_response_series:
                    any_roi_response_series_found = True
                    setattr(self, f'_roi_response_{trace_name_segext}',
                            DatasetView(container.roi_response_series[trace_name].data).lazy_transpose())
                    if self._sampling_frequency is None:
                        self._sampling_frequency = container.roi_response_series[trace_name].rate
            if not any_roi_response_series_found:
                raise Exception('could not find any of \'RoiResponseSeries\'/\'Dff\'/\'Neuropil\'/\'Deconvolved\' named RoiResponseSeries in nwbfile')

            # Extract image_mask/background:
            if 'ImageSegmentation' in ophys.data_interfaces:
                image_seg = ophys.data_interfaces['ImageSegmentation']
                if 'PlaneSegmentation' in image_seg.plane_segmentations:#this requirement in nwbfile is enforced
                    ps = image_seg.plane_segmentations['PlaneSegmentation']
                    if 'image_mask' in ps.colnames:
                        self.image_masks = DatasetView(ps['image_mask'].data).lazy_transpose([1, 2, 0])
                    else:
                        raise Exception('could not find any image_masks in nwbfile')
                    if 'RoiCentroid' in ps.colnames:
                        self._roi_locs = ps['RoiCentroid']
                    if 'Accepted' in ps.colnames:
                        self._accepted_list = ps['Accepted'].data[:]
                    if 'Rejected' in ps.colnames:
                        self._rejected_list = ps['Rejected'].data[:]
                    self._roi_idx = np.array(ps.id.data)
                else:
                    raise Exception('could not find any PlaneSegmentation in nwbfile')

            # Extracting stores images as GrayscaleImages:
            if 'SegmentationImages' in ophys.data_interfaces:
                images_container = ophys.data_interfaces['SegmentationImages']
                if 'correlation' in images_container.images:
                    self._image_correlation = images_container.images['correlation'].data[()]
                if 'mean' in images_container.images:
                    self._image_mean = images_container.images['mean'].data[()]

        # Imaging plane:
        if 'ImagingPlane' in self.nwbfile.imaging_planes:
            imaging_plane = self.nwbfile.imaging_planes['ImagingPlane']
            self._channel_names = [i.name for i in imaging_plane.optical_channel]

    def __del__(self):
        self._io.close()

    def get_accepted_list(self):
        if self._accepted_list is None:
            return list(range(self.get_num_rois()))
        else:
            return np.where(self._accepted_list==1)[0].tolist()

    def get_rejected_list(self):
        return self._rejected_list

    @property
    def roi_locations(self):
        if self._roi_locs is not None:
            return self._roi_locs.data[:].T

    def get_roi_ids(self):
        return self._roi_idx

    def get_image_size(self):
        return self.image_masks.shape[:2]

    @staticmethod
    def get_nwb_metadata(sgmextractor):
        """
        Converts metadata from the segmentation into nwb specific metadata
        Parameters
        ----------
        sgmextractor: SegmentationExtractor
        """
        metadata = {'NWBFile': {'session_start_time': datetime.now(),
                                'identifier': str(uuid.uuid4()),
                                'session_description': 'no description'},
                    'ophys': {'Device': [{'name': 'Microscope'}],
                              'Fluorescence': {'roi_response_series':[{'name': 'RoiResponseSeries',
                                                                       'description': 'array of raw fluorescence traces'}]},
                              'ImageSegmentation': {'plane_segmentations': [{'description': 'Segmented ROIs',
                                                                            'name': 'PlaneSegmentation'}]},
                              'ImagingPlane':[{'name': 'ImagingPlane',
                                               'description': 'no description',
                                               'excitation_lambda': np.nan,
                                               'indicator': 'unknown',
                                               'location': 'unknown',
                                               'optical_channels': [{'name': 'OpticalChannel',
                                                                     'emission_lambda': np.nan,
                                                                     'description': 'no description'}]}],
                              'TwoPhotonSeries': [{'name': 'TwoPhotonSeries'}]}}
        # Optical Channel name:
        for i in range(sgmextractor.get_num_channels()):
            ch_name = sgmextractor.get_channel_names()[i]
            if i == 0:
                metadata['ophys']['ImagingPlane'][0]['optical_channels'][i]['name'] = ch_name
            else:
                metadata['ophys']['ImagingPlane'][0]['optical_channels'].append(dict(
                    name=ch_name,
                    emission_lambda=np.nan,
                    description=f'{ch_name} description'
                ))

        # set roi_response_series rate:
        rate = np.float('NaN') if sgmextractor.get_sampling_frequency() is None else sgmextractor.get_sampling_frequency()
        for trace_name, trace_data in sgmextractor.get_traces_dict().items():
            if trace_name == 'raw':
                if trace_data is not None:
                    metadata['ophys']['Fluorescence']['roi_response_series'][0].update(rate=rate)
                continue
            if len(trace_data.shape) != 0:
                metadata['ophys']['Fluorescence']['roi_response_series'].append(dict(
                    name=trace_name.capitalize(),
                    description=f'description of {trace_name} traces',
                    rate=rate
                ))
        # adding imaging_rate:
        metadata['ophys']['ImagingPlane'][0].update(imaging_rate=rate)
        # TwoPhotonSeries update:
        metadata['ophys']['TwoPhotonSeries'][0].update(
            dimension=sgmextractor.get_image_size())
        return metadata

    @staticmethod
    def write_segmentation(segext_obj, save_path, plane_num=0, metadata=None, file_overwrite=False):
        if os.path.exists(save_path) and not file_overwrite:
            nwbfile_exist = True
            file_mode = 'r+'
        else:
            if os.path.exists(save_path):
                os.remove(save_path)
            if not os.path.exists(os.path.dirname(save_path)):
                os.mkdir(os.path.dirname(save_path))
            nwbfile_exist = False
            file_mode = 'w'
        # parse metadata correctly:
        if isinstance(segext_obj, MultiSegmentationExtractor):
            segext_objs = segext_obj.segmentations
            if metadata is not None and not isinstance(metadata, list):
                raise ValueError('for MultiSegmentationExtractor enter metadata as a list of SegmentationExtractor metadata')
        else:
            segext_objs = [segext_obj]
        metadata_base_list = [NwbSegmentationExtractor.get_nwb_metadata(sgobj) for sgobj in segext_objs]
        print(f'writing nwb for {segext_obj.extractor_name}\n')
        # updating base metadata with new:
        for num, data in enumerate(metadata_base_list):
            metadata_input = metadata[num] if metadata else {}
            metadata_base_list[num] = dict_recursive_update(metadata_base_list[num], metadata_input)
        #loop for every plane:
        with NWBHDF5IO(save_path, file_mode) as io:
            metadata_base_common = metadata_base_list[0]
            if nwbfile_exist:
                nwbfile = io.read()
            else:
                nwbfile = NWBFile(**metadata_base_common['NWBFile'])
                # Subject:
                if metadata_base_common.get('Subject'):
                    nwbfile.subject = Subject(**metadata_base_common['Subject'])

            # Processing Module:
            if 'ophys' not in nwbfile.processing:
                ophys = nwbfile.create_processing_module('ophys',
                                                             'contains optical physiology processed data')
            else:
                ophys = nwbfile.get_processing_module('ophys')

            for plane_no_loop, (segext_obj, metadata) in enumerate(zip(segext_objs, metadata_base_list)):
                # Device:
                if metadata['ophys']['Device'][0]['name'] not in nwbfile.devices:
                    nwbfile.create_device(**metadata['ophys']['Device'][0])

                # ImageSegmentation:
                image_segmentation_name = 'ImageSegmentation' if plane_no_loop==0 else f'ImageSegmentation_Plane{plane_no_loop}'
                if image_segmentation_name not in ophys.data_interfaces:
                    image_segmentation = ImageSegmentation(name=image_segmentation_name)
                    ophys.add_data_interface(image_segmentation)

                # OpticalChannel:
                optical_channels = [OpticalChannel(**i) for i in metadata['ophys']['ImagingPlane'][0]['optical_channels']]

                # ImagingPlane:
                image_plane_name = 'ImagingPlane' if plane_no_loop == 0 else f'ImagePlane_{plane_no_loop}'
                if image_plane_name not in nwbfile.imaging_planes.keys():
                    input_kwargs = dict(
                        name=image_plane_name,
                        device=nwbfile.get_device(metadata_base_common['ophys']['Device'][0]['name']),
                    )
                    _ = metadata['ophys']['ImagingPlane'][0].pop('optical_channels')
                    metadata['ophys']['ImagingPlane'][0].update(optical_channel=optical_channels)
                    input_kwargs.update(**metadata['ophys']['ImagingPlane'][0])
                    imaging_plane = nwbfile.create_imaging_plane(**input_kwargs)
                else:
                    imaging_plane = nwbfile.imaging_planes[image_plane_name]

                # PlaneSegmentation:
                input_kwargs = dict(
                    description='output from segmenting imaging plane',
                    imaging_plane=imaging_plane
                )
                if metadata['ophys']['ImageSegmentation']['plane_segmentations'][0]['name'] not in image_segmentation.plane_segmentations:
                    input_kwargs.update(**metadata['ophys']['ImageSegmentation']['plane_segmentations'][0])
                    ps = image_segmentation.create_plane_segmentation(**input_kwargs)
                    ps_exist = False
                else:
                    ps = image_segmentation.get_plane_segmentation(i['name'])
                    ps_exist = True

                # ROI add:
                image_masks = segext_obj.get_roi_image_masks()
                roi_ids = segext_obj.get_roi_ids()
                accepted_ids = [1 if k in segext_obj.get_accepted_list() else 0 for k in roi_ids]
                rejected_ids = [1 if k in segext_obj.get_rejected_list() else 0 for k in roi_ids]
                roi_locations = np.array(segext_obj.get_roi_locations()).T
                if not ps_exist:
                    ps.add_column(name='RoiCentroid',
                                  description='x,y location of centroid of the roi in image_mask')
                    ps.add_column(name='Accepted',
                                  description='1 if ROi was accepted or 0 if rejected as a cell during segmentation operation')
                    ps.add_column(name='Rejected',
                                  description='1 if ROi was rejected or 0 if accepted as a cell during segmentation operation')
                for num, row in enumerate(roi_ids): #Expects the existing ps to be a prior nwbsegext saved nwb file with existing columns
                    ps.add_row(id=row, image_mask=image_masks[:, :, num],
                               RoiCentroid=roi_locations[num,:],
                               Accepted=accepted_ids[num], Rejected=rejected_ids[num])

                # Fluorescence Traces:
                if 'Flourescence' not in ophys.data_interfaces:
                    fluorescence = Fluorescence()
                    ophys.add_data_interface(fluorescence)
                else:
                    fluorescence = ophys.data_interfaces['Fluorescence']
                roi_response_dict = segext_obj.get_traces_dict()
                roi_table_region = ps.create_roi_table_region(description=f'region for Imaging plane{plane_no_loop}',
                                                              region=list(range(segext_obj.get_num_rois())))
                rate = np.float('NaN') if segext_obj.get_sampling_frequency() is None else segext_obj.get_sampling_frequency()
                for i, j in roi_response_dict.items():
                    data = getattr(segext_obj, f'_roi_response_{i}')
                    if data is not None:
                        trace_name = 'RoiResponseSeries' if i == 'raw' else i.capitalize()
                        trace_name = trace_name if plane_no_loop==0 else trace_name+f'_Plane{plane_no_loop}'
                        input_kwargs = dict(name=trace_name, data=data.T, rois=roi_table_region, rate=rate)
                        if trace_name not in fluorescence.roi_response_series:
                            fluorescence.create_roi_response_series(**input_kwargs)

                # create Two Photon Series:
                if 'TwoPhotonSeries' not in nwbfile.acquisition:
                    warn('could not find TwoPhotonSeries, using ImagingExtractor to create an nwbfile')

                # adding images:
                images_dict = segext_obj.get_images_dict()
                images_name = 'SegmentationImages' if plane_no_loop==0 else f'SegmentationImages_Plane{plane_no_loop}'
                if images_name not in ophys.data_interfaces:
                    images = Images(images_name)
                    for img_name, img_no in images_dict.items():
                        if img_no is not None:
                            images.add_image(GrayscaleImage(name=img_name, data=img_no))
                    ophys.add(images)

            # saving NWB file:
            io.write(nwbfile)

        # test read
        with NWBHDF5IO(save_path, 'r') as io:
            io.read()
