import os
import uuid
import numpy as np
import yaml
from roiextractors import ImagingExtractor, SegmentationExtractor
from lazy_ops import DatasetView

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

    #TODO placeholders
    def get_frame(self, frame_idx, channel=0):
        assert frame_idx < self.get_num_frames()
        return self._video[frame_idx]

    def get_frames(self, frame_idxs):
        assert np.all(frame_idxs < self.get_num_frames())
        planes = np.zeros((len(frame_idxs), self._size_x, self._size_y))
        for i, frame_idx in enumerate(frame_idxs):
            plane = self._video[frame_idx]
            planes[i] = plane
        return planes

    # TODO make decorator to check and correct inputs
    def get_video(self, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        end_frame = min(end_frame, self.get_num_frames())

        video = self._video[start_frame: end_frame]

        return video

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

    def get_traces(self, roi_ids=None, start_frame=None, end_frame=None, name=None):
        if name is None:
            name = self._roi_names[0]
            print(f'returning traces for {name}')
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames() + 1
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
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

    def get_roi_locations(self, roi_ids=None):
        if roi_ids is None:
            return self.roi_locs
        else:
            roi_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
            return self.roi_locs[:, roi_idx_]

    def get_roi_ids(self):
        return self.roi_idx

    def get_num_rois(self):
        return self.no_rois

    def get_pixel_masks(self, roi_ids=None):
        if self.pixel_masks is None:
            return None
        if roi_ids is None:
            roi_idx_ = self.roi_idx
        else:
            roi_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        temp = np.empty((1, 4))
        for i, roiid in enumerate(roi_idx_):
            temp = \
                np.append(temp, self.pixel_masks[self.pixel_masks[:, 3] == roiid, :], axis=0)
        return temp[1::, :]

    def get_image_masks(self, roi_ids=None):
        if self.image_masks is None:
            return None
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return np.array([self.image_masks[:, :, int(i)].T for i in roi_idx_]).T

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
    def write_segmentation(segext_obj, save_path, plane_num=0, file_overwrite=False):
        if os.path.exists(save_path) and not file_overwrite:
            nwbfile_exist = True
            file_mode = 'r+'
        else:
            if os.path.exists(save_path):
                os.remove(save_path)
            nwbfile_exist = False
            file_mode = 'w'
        # parse metadata correctly:
        if segext_obj.extractor_name == 'MultiSegmentationExtractor':
            segext_objs = segext_obj.segmentations
            metadata_list = segext_obj.get_experiment_metadata()
        else:
            metadata_list = [segext_obj.get_experiment_metadata()]
            segext_objs = [segext_obj]
        print(f'writing nwb for {segext_objs[0].extractor_name}\n')
        with NWBHDF5IO(save_path, file_mode) as io:
            metadata = metadata_list[0]
            if nwbfile_exist:
                nwbfile = io.read()
                containers_imageextractors = nwbfile_container_exist(
                    nwbfile, ['NWBFile', 'TwoPhotonSeries'])
                container_segextractors = nwbfile_container_exist(
                    nwbfile, ['ProcessingModule']
                )
                if len(containers_imageextractors.keys()) != 2 or len(container_segextractors.keys()) != 0:
                    raise Exception('nwbfile path provided is not an an output of ImageExtractor nwbwrite method')
            else:
                nwbfile_args = dict(identifier=str(uuid.uuid4()), )
                nwbfile_args.update(**metadata['NWBFile'])
                nwbfile = NWBFile(**nwbfile_args)

                # Subject:
                if metadata.get('Subject'):
                    nwbfile.subject = Subject(**metadata['Subject'])

                # Device:
                if isinstance(metadata['ophys']['Device'], list):
                    for devices in metadata['ophys']['Device']:
                        nwbfile.create_device(**devices)
                else:
                    nwbfile.create_device(**metadata['ophys']['Device'])

            # Processing Module:
            ophys_mod = nwbfile.create_processing_module('ophys',
                                                         'contains optical physiology processed data')

            for plane_no_loop, (segext_obj, metadata) in enumerate(zip(segext_objs, metadata_list)):
                # ImageSegmentation:
                image_segmentation = ImageSegmentation()
                ophys_mod.add_data_interface(image_segmentation)

                # OPtical Channel:
                channel_names = segext_obj.get_channel_names()
                input_args = [dict(name=k) for k in channel_names]
                for j, i in enumerate(metadata['ophys']['ImagingPlane']):
                    for j2, i2 in enumerate(i['optical_channels']):
                        input_args[j2].update(**i2)
                    optical_channels = [OpticalChannel(**input_args[j]) for j, i in enumerate(channel_names)]

                # ImagingPlane:
                input_kwargs = dict(
                    name=f'ImagingPlane{i}',
                    description='no description',
                    device=list(nwbfile.devices.values())[0],
                    excitation_lambda=np.nan,
                    imaging_rate=1.0,
                    indicator='unknown',
                    location='unknown'
                )
                for j, i in enumerate(metadata['ophys']['ImagingPlane']):
                    _ = i.pop('optical_channels')
                    i.update(optical_channel=optical_channels)
                    input_kwargs.update(**i)
                    imaging_plane = nwbfile.create_imaging_plane(**input_kwargs)

                # PlaneSegmentation:
                input_kwargs = dict(
                    description='output from segmenting my favorite imaging plane',
                    imaging_plane=imaging_plane
                )
                ps = []  # multiple plane segmentations per imagesegmentation/plane
                for j, i in enumerate(metadata['ophys']['ImageSegmentation']['plane_segmentations']):
                    input_kwargs.update(**i)
                    ps.append(image_segmentation.create_plane_segmentation(**input_kwargs))

                # ROI add:
                image_mask_list = [segext_obj.get_roi_image_masks()]
                roi_id_list = [segext_obj.get_roi_ids()]
                accepted_id_locs = [[1 if k in [segext_obj.get_accepted_list()][j] else 0 for k in i]
                                    for j, i in enumerate(roi_id_list)]
                for j, ps_loop in enumerate(ps):
                    [ps_loop.add_roi(id=id, image_mask=image_mask_list[j][:, :, arg_id])
                     for arg_id, id in enumerate(roi_id_list[j])]
                    # adding columns to ROI table:
                    ps_loop.add_column(name='RoiCentroid',
                                       description='x,y location of centroid of the roi in image_mask',
                                       data=np.array([segext_obj.get_roi_locations().T][j]))
                    ps_loop.add_column(name='Accepted',
                                       description='1 if ROi was accepted or 0 if rejected as a cell during segmentation operation',
                                       data=accepted_id_locs[j])

                # Fluorescence Traces:
                rate = np.float(
                    'NaN') if segext_obj.get_sampling_frequency() is None else segext_obj.get_sampling_frequency()
                input_kwargs = dict(
                    starting_time=0.0,
                    rate=rate,
                    unit='lumens'
                )
                container_type = [i for i in metadata['ophys'].keys() if i in ['DfOverF', 'Fluorescence']][0]
                f_container = eval(container_type + '()')
                ophys_mod.add_data_interface(f_container)
                roi_response_dict = segext_obj.get_traces_dict()
                c = 0
                for plane_no in range(segext_obj.get_num_planes()):
                    input_kwargs.update(rois=ps[plane_no].create_roi_table_region(
                        description=f'region for Imaging plane{plane_no}',
                        region=list(range(segext_obj.get_num_rois()))))
                    for i, j in roi_response_dict.items():
                        data = getattr(segext_obj, f'_roi_response_{i}')
                        if data is not None:
                            trace_name = 'RoiResponseSeries' if i == 'raw' else i.capitalize()
                            input_kwargs.update(name=trace_name)
                            input_kwargs.update(data=data.T)
                        c += 1
                        f_container.create_roi_response_series(**input_kwargs)

                # create Two Photon Series:
                input_kwargs = dict(
                    description='no description',
                    imaging_plane=imaging_plane,
                    external_file=[segext_obj.get_movie_location()],
                    format='external',
                    rate=rate,
                    starting_time=0.0,
                    starting_frame=[0],
                    dimension=segext_obj.get_image_size()
                )  # Multiple TwoPhotonSeries possible?
                [input_kwargs.update(**i) for j, i in enumerate(metadata['ophys']['TwoPhotonSeries'])]
                tps = nwbfile.add_acquisition(TwoPhotonSeries(**input_kwargs))

                # adding images:
                images_dict = segext_obj.get_images_dict()
                images = Images(f'SegmentationImages_Plane_{plane_no_loop}')
                for img_name, img_no in images_dict.items():
                    if img_no is not None:
                        images.add_image(GrayscaleImage(name=img_name, data=img_no))
                ophys_mod.add(images)

            # saving NWB file:
            io.write(nwbfile)

        # test read
        with NWBHDF5IO(save_path, 'r') as io:
            io.read()
