import os
import uuid
import numpy as np
import yaml
from datetime import datetime
from collections import abc
from lazy_ops import DatasetView
from pathlib import Path

from lazy_ops import DatasetView
from ...imagingextractor import ImagingExtractor
from ...segmentationextractor import SegmentationExtractor
from ...extraction_tools import PathType, check_get_frames_args, check_get_videos_args, _pixel_mask_extractor


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


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


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

    def __init__(self, file_path, optical_series_name='TwoPhotonSeries'):
        """
        Parameters
        ----------
        file_path: str
            The location of the folder containing dataset.nwb file
        optical_series_name: str (optional)
            optical series to extract data from
        """
        assert HAVE_NWB, self.installation_mesg
        ImagingExtractor.__init__(self)
        self._path = file_path

        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            if optical_series_name is not None:
                self._optical_series_name = optical_series_name
            else:
                a_names = list(nwbfile.acquisition)
                if len(a_names) > 1:
                    raise ValueError('More than one acquisition found. You must specify electrical_series.')
                if len(a_names) == 0:
                    raise ValueError('No acquisitions found in the .nwb file.')
                self._optical_series_name = a_names[0]

            opts = nwbfile.acquisition[self._optical_series_name]
            assert isinstance(opts, TwoPhotonSeries), "The optical series must be of type pynwb.TwoPhotonSeries"

            #TODO if external file --> return another proper extractor (e.g. TiffImagingExtractor)
            assert opts.external_file is None, "Only 'raw' format is currently supported"

            if hasattr(opts, 'timestamps') and opts.timestamps:
                self._sampling_frequency = 1. / np.median(np.diff(opts.timestamps))
                self._imaging_start_time = opts.timestamps[0]
            else:
                self._sampling_frequency = opts.rate
                if hasattr(os, 'starting_time'):
                    self._imaging_start_time = opts.starting_time
                else:
                    self._imaging_start_time = 0.

            if len(opts.data.shape) == 3:
                self._num_frames, self._size_x, self._size_y = opts.data.shape
                self._num_channels = 1
                self._channel_names = opts.imaging_plane.optical_channel[0].name
            else:
                raise NotImplementedError("4D volumetric data are currently not supported")

            # Fill epochs dictionary
            self._epochs = {}
            if nwbfile.epochs is not None:
                df_epochs = nwbfile.epochs.to_dataframe()
                self._epochs = {row['tags'][0]: {
                    'start_frame': self.time_to_frame(row['start_time']),
                    'end_frame': self.time_to_frame(row['stop_time'])}
                    for _, row in df_epochs.iterrows()}

            self._kwargs = {'file_path': str(Path(file_path).absolute()),
                            'optical_series_name': optical_series_name}
            self.make_nwb_metadata(nwbfile=nwbfile, opts=opts)

    def make_nwb_metadata(self, nwbfile, opts):
        # Metadata dictionary - useful for constructing a nwb file
        self.nwb_metadata = dict()
        self.nwb_metadata['NWBFile'] = {
            'session_description': nwbfile.session_description,
            'identifier': nwbfile.identifier,
            'session_start_time': nwbfile.session_start_time,
            'institution': nwbfile.institution,
            'lab': nwbfile.lab
        }
        self.nwb_metadata['Ophys'] = dict()
        # Update metadata with Device info
        self.nwb_metadata['Ophys']['Device'] = []
        for dev in nwbfile.devices:
            self.nwb_metadata['Ophys']['Device'].append({'name': dev})

        # Update metadata with ElectricalSeries info
        self.nwb_metadata['Ophys']['TwoPhotonSeries'] = []
        self.nwb_metadata['Ophys']['TwoPhotonSeries'].append({
            'name': opts.name
        })

    #TODO use lazy_ops
    @check_get_frames_args
    def get_frames(self, frame_idxs, channel=0):
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            opts = nwbfile.acquisition[self._optical_series_name]
            if frame_idxs.size > 1 and np.all(np.diff(frame_idxs) > 0):
                return opts.data[frame_idxs]
            else:
                sorted_idxs = np.sort(frame_idxs)
                argsorted_idxs = np.argsort(frame_idxs)
                return opts.data[sorted_idxs][argsorted_idxs]

    @check_get_videos_args
    def get_video(self, start_frame=None, end_frame=None, channel=0):
        with NWBHDF5IO(self._path, 'r') as io:
            nwbfile = io.read()
            opts = nwbfile.acquisition[self._optical_series_name]
            video = opts.data[start_frame:end_frame]
            return video

    def get_image_size(self):
        return [self._size_x, self._size_y]

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

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
    def add_devices(imaging, nwbfile, metadata):
        # Devices
        if 'Ophys' not in metadata:
            metadata['Ophys'] = dict()
        if 'Device' not in metadata['Ophys']:
            metadata['Ophys']['Device'] = [{'name': 'Device'}]
        # Tests if devices exist in nwbfile, if not create them from metadata
        for dev in metadata['Ophys']['Device']:
            if dev['name'] not in nwbfile.devices:
                nwbfile.create_device(name=dev['name'])

        return nwbfile

    @staticmethod
    def add_two_photon_series(imaging, nwbfile, metadata):
        """
        Auxiliary static method for nwbextractor.
        Adds two photon series from imaging object as TwoPhotonSeries to nwbfile object.
        """
        if 'Ophys' not in metadata:
            metadata['Ophys'] = {}

        if 'Ophys' not in metadata or 'TwoPthotonSeries' not in metadata['Ophys']:
            metadata['Ophys']['TwoPhotonSeries'] = [{'name': 'TwoPhotonSeries',
                                                     'description': 'optical_series_description'}]
        # Tests if ElectricalSeries already exists in acquisition
        nwb_es_names = [ac for ac in nwbfile.acquisition]
        opts = metadata['Ophys']['TwoPhotonSeries'][0]
        if opts['name'] not in nwb_es_names:
            # retrieve device
            device = nwbfile.devices[list(nwbfile.devices.keys())[0]]

            # create optical channel
            if 'OpticalChannel' not in metadata['Ophys']:
                metadata['Ophys']['OpticalChannel'] = [{'name': 'OpticalChannel',
                                                        'description': 'no description',
                                                        'emission_lambda': 500.}]

            optical_channel = OpticalChannel(**metadata['Ophys']['OpticalChannel'][0])
            # sampling rate
            rate = float(imaging.get_sampling_frequency())

            if 'ImagingPlane' not in metadata['Ophys']:
                metadata['Ophys']['ImagingPlane'] = [{'name': 'ImagingPlane',
                                                      'description': 'no description',
                                                      'excitation_lambda': 600.,
                                                      'indicator': 'Indicator',
                                                      'location': 'Location',
                                                      'grid_spacing': [.01, .01],
                                                      'grid_spacing_unit': 'meters'}]
            imaging_meta = {'optical_channel': optical_channel,
                            'imaging_rate': rate,
                            'device': device}
            metadata['Ophys']['ImagingPlane'][0] = update_dict(metadata['Ophys']['ImagingPlane'][0], imaging_meta)

            imaging_plane = nwbfile.create_imaging_plane(**metadata['Ophys']['ImagingPlane'][0])

            # def data_generator(imaging, channels_ids):
            #     #  generates data chunks for iterator
            #     for id in channels_ids:
            #         data = recording.get_traces(channel_ids=[id]).flatten()
            #         yield data
            #
            # data = data_generator(imaging=imaging, channels_ids=curr_ids)
            # ophys_data = DataChunkIterator(data=data, iter_axis=1)
            acquisition_name = opts['name']

            # using internal data. this data will be stored inside the NWB file
            ophys_ts = TwoPhotonSeries(
                name=acquisition_name,
                data=imaging.get_video(),
                imaging_plane=imaging_plane,
                rate=rate,
                unit='normalized amplitude',
                comments='Generated from RoiInterface::NwbImagingExtractor',
                description='acquisition_description'
            )

            nwbfile.add_acquisition(ophys_ts)

        return nwbfile

    @staticmethod
    def add_epochs(imaging, nwbfile):
        """
        Auxiliary static method for nwbextractor.
        Adds epochs from recording object to nwbfile object.
        """
        # add/update epochs
        for (name, ep) in imaging._epochs.items():
            if nwbfile.epochs is None:
                nwbfile.add_epoch(
                    start_time=imaging.frame_to_time(ep['start_frame']),
                    stop_time=imaging.frame_to_time(ep['end_frame']),
                    tags=name
                )
            else:
                if [name] in nwbfile.epochs['tags'][:]:
                    ind = nwbfile.epochs['tags'][:].index([name])
                    nwbfile.epochs['start_time'].data[ind] = imaging.frame_to_time(ep['start_frame'])
                    nwbfile.epochs['stop_time'].data[ind] = imaging.frame_to_time(ep['end_frame'])
                else:
                    nwbfile.add_epoch(
                        start_time=imaging.frame_to_time(ep['start_frame']),
                        stop_time=imaging.frame_to_time(ep['end_frame']),
                        tags=name
                    )

        return nwbfile

    @staticmethod
    def write_imaging(imaging: ImagingExtractor, save_path: PathType = None, nwbfile=None,
                      metadata: dict = None):
        '''
        Parameters
        ----------
        imaging: ImagingExtractor
        save_path: PathType
            Required if an nwbfile is not passed. Must be the path to the nwbfile
            being appended, otherwise one is created and written.
        nwbfile: NWBFile
            Required if a save_path is not specified. If passed, this function
            will fill the relevant fields within the nwbfile. E.g., calling

            roiextractors.NwbImagingExtractor.write_imaging(
                my_imaging_extractor, my_nwbfile
            )

            will result in the appropriate changes to the my_nwbfile object.
        metadata: dict
            metadata info for constructing the nwb file (optional).
        '''
        assert HAVE_NWB, NwbImagingExtractor.installation_mesg

        if nwbfile is not None:
            assert isinstance(nwbfile, NWBFile), "'nwbfile' should be of type pynwb.NWBFile"

        assert save_path is None or nwbfile is None, \
            'Either pass a save_path location, or nwbfile object, but not both!'

        # Update any previous metadata with user passed dictionary
        if metadata is None:
            metadata = dict()
        if hasattr(imaging, 'nwb_metadata'):
            metadata = update_dict(imaging.nwb_metadata, metadata)

        if nwbfile is None:
            if os.path.exists(save_path):
                read_mode = 'r+'
            else:
                read_mode = 'w'

            with NWBHDF5IO(save_path, mode=read_mode) as io:
                if read_mode == 'r+':
                    nwbfile = io.read()
                else:
                    # Default arguments will be over-written if contained in metadata
                    nwbfile_kwargs = dict(session_description='no description',
                                          identifier=str(uuid.uuid4()),
                                          session_start_time=datetime.now())
                    if 'NWBFile' in metadata:
                        nwbfile_kwargs.update(metadata['NWBFile'])
                    nwbfile = NWBFile(**nwbfile_kwargs)

                    NwbImagingExtractor.add_devices(imaging=imaging,
                                                    nwbfile=nwbfile,
                                                    metadata=metadata)

                    NwbImagingExtractor.add_two_photon_series(imaging=imaging,
                                                              nwbfile=nwbfile,
                                                              metadata=metadata)

                    NwbImagingExtractor.add_epochs(imaging=imaging,
                                                   nwbfile=nwbfile)

                # Write to file
                io.write(nwbfile)
        else:
            NwbImagingExtractor.add_devices(imaging=imaging,
                                            nwbfile=nwbfile,
                                            metadata=metadata)

            NwbImagingExtractor.add_two_photon_series(imaging=imaging,
                                                      nwbfile=nwbfile,
                                                      metadata=metadata)

            NwbImagingExtractor.add_epochs(imaging=imaging,
                                           nwbfile=nwbfile)


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
        SegmentationExtractor.__init__(self)
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
            roi_idx = [np.where(np.array(i) == self.roi_ids)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return np.array([self._roi_response_dict[name][int(i), start_frame:end_frame] for i in roi_idx_])

    def get_num_frames(self):
        return self._roi_response.shape[1]

    def get_roi_locations(self, roi_ids=None):
        if roi_ids is None:
            return self.roi_locations
        else:
            roi_idx = [np.where(np.array(i) == self.roi_ids)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
            return self.roi_locations[:, roi_idx_]

    def get_roi_ids(self):
        return self._roi_idx

    def get_num_rois(self):
        return self.roi_ids.size

    def get_roi_pixel_masks(self, roi_ids=None):
        if self.pixel_masks is None:
            return None
        if roi_ids is None:
            roi_idx_ = self.roi_ids
        else:
            roi_idx = [np.where(np.array(i) == self.roi_ids)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        temp = np.empty((1, 4))
        for i, roiid in enumerate(roi_idx_):
            temp = \
                np.append(temp, self.pixel_masks[self.pixel_masks[:, 3] == roiid, :], axis=0)
        return temp[1::, :]

    def get_roi_image_masks(self, roi_ids=None):
        if self.image_masks is None:
            return None
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.roi_ids)[0] for i in roi_ids]
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

    def get_image_size(self):
        return self._extimage_dims

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
