import os
import uuid
import numpy as np
import yaml
from lazy_ops import DatasetView
from ...imagingextractor import ImagingExtractor
from ...segmentationextractor import SegmentationExtractor
from ...extraction_tools import check_get_frames_args, _pixel_mask_extractor


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

    def __init__(self, filepath, optical_channel_name=None,
                 imaging_plane_name=None, image_series_name=None,
                 processing_module_name=None,
                 neuron_roi_response_series_name=None,
                 background_roi_response_series_name=None):
        """
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

    def __init__(self, filepath):
        """
        Creating NwbSegmentationExtractor object from nwb file
        Parameters
        ----------
        filepath: str
            .nwb file location
        """
        check_nwb_install()
        SegmentationExtractor.__init__(self)
        if not os.path.exists(filepath):
            raise Exception('file does not exist')

        self.filepath = filepath
        self.image_masks = None
        self.pixel_masks = None
        self._roi_locs = None
        self._accepted_list = None
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
        if 'RoiCentroid' in ps.colnames:
            self._roi_locs = ps['RoiCentroid']
        if 'Accepted' in ps.colnames:
            self._accepted_list = ps['Accepted'].data[:]
        # Extract Image dimensions:

        # Extract roi_response:
        self._roi_response_dict = dict()
        self._roi_names = [_nwbchildren_name[val]
                      for val, i in enumerate(_nwbchildren_type) if i == 'RoiResponseSeries']
        if not self._roi_names:
            raise Exception('no ROI response series found')
        else:
            for roi_name in self._roi_names:
                self._roi_response_dict[roi_name] = mod['Fluorescence'].get_roi_response_series(roi_name).data[:].T
        self._roi_response = self._roi_response_dict[self._roi_names[0]]

        # Extract samp_freq:
        self._sampling_frequency = mod['Fluorescence'].get_roi_response_series(self._roi_names[0]).rate
        # Extract no_rois/ids:
        self._roi_idx = np.array(ps.id.data)

        # Imaging plane:
        _optical_channel_exist = [i for i, e in enumerate(
            _nwbchildren_type) if e == 'OpticalChannel']
        if _optical_channel_exist:
            self._channel_names = []
            for i in _optical_channel_exist:
                self._channel_names.append(nwbfile.all_children()[i].name)
        # Movie location:
        _image_series_exist = [i for i, e in enumerate(
            _nwbchildren_type) if e == 'TwoPhotonSeries']
        if not _image_series_exist:
            self._extimage_dims = None
        else:
            self._raw_movie_file_location = \
                nwbfile.all_children()[_image_series_exist[0]].external_file[:][0]
            self._extimage_dims = \
                nwbfile.all_children()[_image_series_exist[0]].dimension

        # property name/data extraction:
        self._property_name_exist = [
            i for i in ps.colnames if i not in ['image_mask', 'pixel_mask']]
        self.property_vals = []
        for i in self._property_name_exist:
            self.property_vals.append(np.array(ps[i].data))

        #Extracting stores images as GrayscaleImages:
        self._greyscaleimages = [_nwbchildren_name[f] for f, u in enumerate(_nwbchildren_type) if u == 'GrayscaleImage']

    def get_accepted_list(self):
        if self._accepted_list is None:
            return list(range(self.no_rois))
        else:
            return np.where(self._accepted_list==1)[0].tolist()

    def get_rejected_list(self):
        return [a for a in self.roi_ids if a not in set(self.get_accepted_list())]

    @property
    def roi_locations(self):
        if self._roi_locs is None:
            return None
        else:
            return self._roi_locs.data[:].T

    def get_roi_ids(self):
        return self._roi_idx

    def get_images(self):
        imag_dict = {i.name: np.array(i.data) for i in self.nwbfile.all_children() if i.name in self._greyscaleimages}
        _ = {i.name: i for i in self.nwbfile.all_children() if i.name in self._greyscaleimages}
        if imag_dict:
            parent_name = _[self._greyscaleimages[0]].parent.name
            return {parent_name: imag_dict}
        else:
            return None

    def get_image_size(self):
        return self._extimage_dims

    @staticmethod
    def write_segmentation(segext_obj, savepath, metadata_dict, **kwargs):
        if isinstance(metadata_dict, str):
            with open(metadata_dict, 'r') as f:
                metadata_dict = yaml.safe_load(f)

        # NWBfile:
        nwbfile_args = dict(identifier=str(uuid.uuid4()), )
        if 'NWBFile' in metadata_dict:
            nwbfile_args.update(**metadata_dict['NWBFile'])
        nwbfile = NWBFile(**nwbfile_args)

        # Subject:
        if 'Subject' in metadata_dict:
            nwbfile.subject = Subject(**metadata_dict['Subject'])

        # Device:
        if isinstance(metadata_dict['ophys']['Device'], list):
            for devices in metadata_dict['ophys']['Device']:
                nwbfile.create_device(**devices)
        else:
            nwbfile.create_device(**metadata_dict['ophys']['Device'])

        # Processing Module:
        ophys_mod = nwbfile.create_processing_module('ophys',
                                                     'contains optical physiology processed data')

        # ImageSegmentation:
        image_segmentation = ImageSegmentation(metadata_dict['ophys']['ImageSegmentation']['name'])
        ophys_mod.add_data_interface(image_segmentation)

        # OpticalChannel:
        channel_names = segext_obj.get_channel_names()
        input_args = [dict(name=i) for i in channel_names]
        for j, i in enumerate(metadata_dict['ophys']['ImagingPlane']['optical_channel']):
            input_args[j].update(**i)
        optical_channels = [OpticalChannel(input_args[j]) for j, i in enumerate(channel_names)]

        # ImagingPlane:
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
        [input_kwargs[j].update(**i) for j, i in enumerate(metadata_dict['ophys']['ImagingPlane'])]  # update with
        # metadata
        imaging_planes = [nwbfile.create_imaging_plane(i) for i in input_kwargs]

        # PlaneSegmentation:
        input_kwargs = [dict(
            name='PlaneSegmentation',
            description='output from segmenting my favorite imaging plane',
            imaging_plane=i
        ) for i in imaging_planes]
        [input_kwargs[j].update(**i) for j, i in
         enumerate(metadata_dict['ophys']['ImageSegmentation']['plane_segmentations'])]  # update with metadata
        ps = [image_segmentation.create_plane_segmentation(i) for i in input_kwargs]

        # ROI add:
        pixel_mask_exist = segext_obj.get_pixel_masks() is not None
        for i, roiid in enumerate(segext_obj.roi_idx):
            if pixel_mask_exist:
                [ps_loop.add_roi(
                    id=roiid,
                    pixel_mask=segext_obj.get_pixel_masks(roi_ids=[roiid])[:, 0:-1])
                 for ps_loop in ps]
            else:
                [ps_loop.add_roi(
                    id=roiid,
                    image_mask=segext_obj.get_image_masks(roi_ids=[roiid]))
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

        # Fluorescence Traces:
        input_kwargs = dict(
            rois=ps[0].create_roi_table_region('NeuronROIs', region=list(range(segext_obj.no_rois))),
            starting_time=0.0,
            rate=segext_obj.get_sampling_frequency(),
            unit='lumens'
        )
        container_type = [i for i in metadata_dict['ophys'].keys() if i in ['DfOverF','Fluorescence']][0]
        f_container = eval(container_type+'()')
        ophys_mod.add_data_interface(f_container)
        for i in metadata_dict['ophys'][container_type]['roi_response_series']:
            i.update(**input_kwargs,data=segext_obj.get_traces_info()[i['name']].T)
            f_container.create_roi_response_series(**i)

        # create TwoPhotonSeries:
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
        [input_kwargs[j].update(**i) for j, i in enumerate(metadata_dict['ophys']['TwoPhotonSeries'])]
        [nwbfile.add_acquisition(TwoPhotonSeries(**i)) for i in input_kwargs]

        # adding images:
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

        # test read
        with NWBHDF5IO(savepath, 'r') as io:
            io.read()
