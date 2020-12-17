import os
import uuid
from collections import abc
from datetime import datetime
from pathlib import Path
from warnings import warn

import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.data_utils import DataChunkIterator
from lazy_ops import DatasetView
from pynwb import NWBFile, NWBHDF5IO
from pynwb.base import Images
from pynwb.file import Subject
from pynwb.image import GrayscaleImage
from pynwb.ophys import ImageSegmentation, Fluorescence, OpticalChannel, TwoPhotonSeries

from ...extraction_tools import PathType, FloatType, IntType, \
    check_get_frames_args, check_get_videos_args, \
    dict_recursive_update
from ...imagingextractor import ImagingExtractor
from ...multisegmentationextractor import MultiSegmentationExtractor
from ...segmentationextractor import SegmentationExtractor

HAVE_NWB = True


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


def get_default_nwb_metadata():
    metadata = {'NWBFile': {'session_start_time': datetime.now(),
                            'identifier': str(uuid.uuid4()),
                            'session_description': 'no description'},
                'Ophys': {'Device': [{'name': 'Microscope'}],
                          'Fluorescence': {'roi_response_series': [{'name': 'RoiResponseSeries',
                                                                    'description': 'array of raw fluorescence traces'}]},
                          'ImageSegmentation': {'plane_segmentations': [{'description': 'Segmented ROIs',
                                                                         'name': 'PlaneSegmentation'}]},
                          'ImagingPlane': [{'name': 'ImagingPlane',
                                            'description': 'no description',
                                            'excitation_lambda': np.nan,
                                            'indicator': 'unknown',
                                            'location': 'unknown',
                                            'optical_channel': [{'name': 'OpticalChannel',
                                                                 'emission_lambda': np.nan,
                                                                 'description': 'no description'}]}],
                          'TwoPhotonSeries': [{'name': 'TwoPhotonSeries',
                                               'description': 'no description',
                                               'comments': 'Generalized from RoiInterface'}]}}
    return metadata


class NwbImagingExtractor(ImagingExtractor):
    """
    Class used to extract data from the NWB data format. Also implements a
    static method to write any format specific object to NWB.
    """

    extractor_name = 'NwbImaging'
    installed = HAVE_NWB  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the Nwb Extractor run:\n\n pip install pynwb\n\n"  # error message when not installed

    def __init__(self, file_path: PathType, optical_series_name: str = 'TwoPhotonSeries'):
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

        self.io = NWBHDF5IO(self._path, 'r')
        self.nwbfile = self.io.read()
        if optical_series_name is not None:
            self._optical_series_name = optical_series_name
        else:
            a_names = list(self.nwbfile.acquisition)
            if len(a_names) > 1:
                raise ValueError('More than one acquisition found. You must specify two_photon_series.')
            if len(a_names) == 0:
                raise ValueError('No acquisitions found in the .nwb file.')
            self._optical_series_name = a_names[0]

        opts = self.nwbfile.acquisition[self._optical_series_name]
        assert isinstance(opts, TwoPhotonSeries), "The optical series must be of type pynwb.TwoPhotonSeries"

        # TODO if external file --> return another proper extractor (e.g. TiffImagingExtractor)
        assert opts.external_file is None, "Only 'raw' format is currently supported"

        if hasattr(opts, 'timestamps') and opts.timestamps:
            self._sampling_frequency = 1. / np.median(np.diff(opts.timestamps))
            self._imaging_start_time = opts.timestamps[0]
        else:
            self._sampling_frequency = opts.rate
            self._imaging_start_time = opts.get(os, 'starting_time', 0.)

        if len(opts.data.shape) == 3:
            self._num_frames, self._size_x, self._size_y = opts.data.shape
            self._num_channels = 1
            self._channel_names = opts.imaging_plane.optical_channel[0].name
        else:
            raise NotImplementedError("4D volumetric data are currently not supported")

        # Fill epochs dictionary
        self._epochs = {}
        if self.nwbfile.epochs is not None:
            df_epochs = self.nwbfile.epochs.to_dataframe()
            # TODO implement add_epoch() method in base class
            self._epochs = {row['tags'][0]: {
                'start_frame': self.time_to_frame(row['start_time']),
                'end_frame': self.time_to_frame(row['stop_time'])}
                for _, row in df_epochs.iterrows()}

        self._kwargs = {'file_path': str(Path(file_path).absolute()),
                        'optical_series_name': optical_series_name}

    def __del__(self):
        self.io.close()

    def time_to_frame(self, time: FloatType):
        return int((time - self._imaging_start_time) * self.get_sampling_frequency())

    def frame_to_time(self, frame: IntType):
        return float(frame / self.get_sampling_frequency() + self._imaging_start_time)

    def make_nwb_metadata(self, nwbfile, opts):
        # Metadata dictionary - useful for constructing a nwb file
        self.nwb_metadata = dict(
            NWBFile=dict(
                session_description=nwbfile.session_description,
                identifier=nwbfile.identifier,
                session_start_time=nwbfile.session_start_time,
                institution=nwbfile.institution,
                lab=nwbfile.lab
            ),
            Ophys=dict(
                Device=[
                    dict(name=dev) for dev in nwbfile.devices
                ],
                TwoPhotonSeries=[
                    dict(
                        name=opts.name
                    )
                ]
            )
        )

    # TODO use lazy_ops
    @check_get_frames_args
    def get_frames(self, frame_idxs, channel=0):
        opts = self.nwbfile.acquisition[self._optical_series_name]
        if frame_idxs.size > 1 and np.all(np.diff(frame_idxs) > 0):
            return opts.data[frame_idxs]
        else:
            sorted_idxs = np.sort(frame_idxs)
            argsorted_idxs = np.argsort(frame_idxs)
            return opts.data[sorted_idxs][argsorted_idxs]

    @check_get_videos_args
    def get_video(self, start_frame=None, end_frame=None, channel=0):
        opts = self.nwbfile.acquisition[self._optical_series_name]
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
        metadata = dict_recursive_update(get_default_nwb_metadata(), metadata)
        # Tests if devices exist in nwbfile, if not create them from metadata
        for dev in metadata['Ophys']['Device']:
            if dev['name'] not in nwbfile.devices:
                nwbfile.create_device(name=dev['name'])

        return nwbfile

    @staticmethod
    def add_two_photon_series(imaging, nwbfile, metadata, num_chunks=10):
        """
        Auxiliary static method for nwbextractor.
        Adds two photon series from imaging object as TwoPhotonSeries to nwbfile object.
        """
        metadata = dict_recursive_update(get_default_nwb_metadata(), metadata)
        metadata = update_dict(metadata, NwbImagingExtractor.get_nwb_metadata(imaging))
        # Tests if ElectricalSeries already exists in acquisition
        nwb_es_names = [ac for ac in nwbfile.acquisition]
        opts = metadata['Ophys']['TwoPhotonSeries'][0]
        if opts['name'] not in nwb_es_names:
            # retrieve device
            device = nwbfile.devices[list(nwbfile.devices.keys())[0]]
            metadata['Ophys']['ImagingPlane'][0]['optical_channel'] = \
                [OpticalChannel(**i) for i in metadata['Ophys']['ImagingPlane'][0]['optical_channel']]
            metadata['Ophys']['ImagingPlane'][0] = update_dict(metadata['Ophys']['ImagingPlane'][0], {'device': device})

            imaging_plane = nwbfile.create_imaging_plane(**metadata['Ophys']['ImagingPlane'][0])

            def data_generator(imaging, num_chunks):
                num_frames = imaging.get_num_frames()
                # chunk size is not None
                chunk_size = num_frames//num_chunks
                if num_frames%chunk_size > 0:
                    num_chunks += 1
                for i in range(num_chunks):
                    video = imaging.get_video(start_frame=i * chunk_size,
                                              end_frame=min((i + 1) * chunk_size, num_frames))
                    data = np.squeeze(video)
                    yield data

            data = H5DataIO(DataChunkIterator(data_generator(imaging, num_chunks)), compression=True)
            acquisition_name = opts['name']

            # using internal data. this data will be stored inside the NWB file
            two_p_series_kwargs = update_dict(
                metadata['Ophys']['TwoPhotonSeries'][0],
                dict(
                    data=data,
                    imaging_plane=imaging_plane)
                )
            ophys_ts = TwoPhotonSeries(**two_p_series_kwargs)

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
    def get_nwb_metadata(imgextractor: ImagingExtractor):
        """
        Converts metadata from the segmentation into nwb specific metadata
        Parameters
        ----------
        imgextractor: ImagingExtractor
        """
        metadata = get_default_nwb_metadata()
        # Optical Channel name:
        for i in range(imgextractor.get_num_channels()):
            ch_name = imgextractor.get_channel_names()[i]
            if i == 0:
                metadata['Ophys']['ImagingPlane'][0]['optical_channel'][i]['name'] = ch_name
            else:
                metadata['Ophys']['ImagingPlane'][0]['optical_channel'].append(dict(
                    name=ch_name,
                    emission_lambda=np.nan,
                    description=f'{ch_name} description'
                ))

        # set imaging plane rate:
        rate = np.float('NaN') if imgextractor.get_sampling_frequency() is None else imgextractor.get_sampling_frequency()
        # adding imaging_rate:
        metadata['Ophys']['ImagingPlane'][0].update(imaging_rate=rate)
        # TwoPhotonSeries update:
        metadata['Ophys']['TwoPhotonSeries'][0].update(
            dimension=imgextractor.get_image_size())
        # remove what Segmentation extractor will input:
        _ = metadata['Ophys'].pop('ImageSegmentation')
        _ = metadata['Ophys'].pop('Fluorescence')
        return metadata

    @staticmethod
    def write_imaging(imaging: ImagingExtractor, save_path: PathType = None, nwbfile=None,
                      metadata: dict = None, overwrite: bool = False, num_chunks: int = 10):
        """
        Parameters
        ----------
        imaging: ImagingExtractor
            The imaging extractor object to be written to nwb
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
        overwrite: bool
            If True and save_path is existing, it is overwritten
        num_chunks: int
            Number of chunks for writing data to file
        """
        assert HAVE_NWB, NwbImagingExtractor.installation_mesg

        assert save_path is None or nwbfile is None, \
            'Either pass a save_path location, or nwbfile object, but not both!'

        if nwbfile is not None:
            assert isinstance(nwbfile, NWBFile), "'nwbfile' should be of type pynwb.NWBFile"

        # Update any previous metadata with user passed dictionary
        if metadata is None:
            metadata = dict()
        if hasattr(imaging, 'nwb_metadata'):
            metadata = update_dict(imaging.nwb_metadata, metadata)
        # update with default arguments:
        metadata = dict_recursive_update(NwbImagingExtractor.get_nwb_metadata(imaging), metadata)
        if nwbfile is None:
            save_path = Path(save_path)
            assert save_path.suffix == '.nwb', "'save_path' file is not an .nwb file"

            if save_path.is_file():
                if not overwrite:
                    read_mode = 'r+'
                else:
                    save_path.unlink()
                    read_mode = 'w'
            else:
                read_mode = 'w'

            with NWBHDF5IO(str(save_path), mode=read_mode) as io:
                if read_mode == 'r+':
                    nwbfile = io.read()
                else:
                    nwbfile = NWBFile(**metadata['NWBFile'])

                    NwbImagingExtractor.add_devices(imaging=imaging,
                                                    nwbfile=nwbfile,
                                                    metadata=metadata)

                    NwbImagingExtractor.add_two_photon_series(imaging=imaging,
                                                              nwbfile=nwbfile,
                                                              metadata=metadata,
                                                              num_chunks=num_chunks)

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

    def __init__(self, file_path: PathType):
        """
        Creating NwbSegmentationExtractor object from nwb file
        Parameters
        ----------
        file_path: PathType
            .nwb file location
        """
        check_nwb_install()
        SegmentationExtractor.__init__(self)
        file_path = Path(file_path)
        if not file_path.is_file():
            raise Exception('file does not exist')

        self.file_path = file_path
        self.image_masks = None
        self._roi_locs = None
        self._accepted_list = None
        self._rejected_list = None
        self._io = NWBHDF5IO(str(file_path), mode='r')
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
                container = dfof if trace_name == 'Dff' else fluorescence
                if container is not None and trace_name in container.roi_response_series:
                    any_roi_response_series_found = True
                    setattr(self, f'_roi_response_{trace_name_segext}',
                            DatasetView(container.roi_response_series[trace_name].data).lazy_transpose())
                    if self._sampling_frequency is None:
                        self._sampling_frequency = container.roi_response_series[trace_name].rate
            if not any_roi_response_series_found:
                raise Exception(
                    'could not find any of \'RoiResponseSeries\'/\'Dff\'/\'Neuropil\'/\'Deconvolved\' named RoiResponseSeries in nwbfile')

            # Extract image_mask/background:
            if 'ImageSegmentation' in ophys.data_interfaces:
                image_seg = ophys.data_interfaces['ImageSegmentation']
                if 'PlaneSegmentation' in image_seg.plane_segmentations:  # this requirement in nwbfile is enforced
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
            return np.where(self._accepted_list == 1)[0].tolist()

    def get_rejected_list(self):
        if self._rejected_list is not None:
            rej_list = np.where(self._rejected_list == 1)[0].tolist()
            if len(rej_list) > 0:
                return rej_list

    @property
    def roi_locations(self):
        if self._roi_locs is not None:
            return self._roi_locs.data[:].T

    def get_roi_ids(self):
        return list(self._roi_idx)

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
        metadata = get_default_nwb_metadata()
        # Optical Channel name:
        for i in range(sgmextractor.get_num_channels()):
            ch_name = sgmextractor.get_channel_names()[i]
            if i == 0:
                metadata['Ophys']['ImagingPlane'][0]['optical_channel'][i]['name'] = ch_name
            else:
                metadata['Ophys']['ImagingPlane'][0]['optical_channel'].append(dict(
                    name=ch_name,
                    emission_lambda=np.nan,
                    description=f'{ch_name} description'
                ))

        # set roi_response_series rate:
        rate = np.float(
            'NaN') if sgmextractor.get_sampling_frequency() is None else sgmextractor.get_sampling_frequency()
        for trace_name, trace_data in sgmextractor.get_traces_dict().items():
            if trace_name == 'raw':
                if trace_data is not None:
                    metadata['Ophys']['Fluorescence']['roi_response_series'][0].update(rate=rate)
                continue
            if len(trace_data.shape) != 0:
                metadata['Ophys']['Fluorescence']['roi_response_series'].append(dict(
                    name=trace_name.capitalize(),
                    description=f'description of {trace_name} traces',
                    rate=rate
                ))
        # adding imaging_rate:
        metadata['Ophys']['ImagingPlane'][0].update(imaging_rate=rate)
        # remove what imaging extractor will input:
        _ = metadata['Ophys'].pop('TwoPhotonSeries')
        return metadata

    @staticmethod
    def write_segmentation(segext_obj: SegmentationExtractor, save_path, plane_num=0, metadata=None, overwrite=True):
        save_path = Path(save_path)
        assert save_path.suffix == '.nwb'
        if save_path.is_file() and not overwrite:
            nwbfile_exist = True
            file_mode = 'r+'
        else:
            if save_path.is_file():
                os.remove(save_path)
            if not save_path.parent.is_dir():
                save_path.parent.mkdir(parents=True)
            nwbfile_exist = False
            file_mode = 'w'

        # parse metadata correctly:
        if isinstance(segext_obj, MultiSegmentationExtractor):
            segext_objs = segext_obj.segmentations
            if metadata is not None:
                assert isinstance(metadata, list), "For MultiSegmentationExtractor enter 'metadata' as a list of " \
                                                   "SegmentationExtractor metadata"
                assert len(metadata) == len(segext_objs), "The 'metadata' argument should be a list with the same " \
                                                          "number of elements as the segmentations in the " \
                                                          "MultiSegmentationExtractor"
        else:
            segext_objs = [segext_obj]
            if metadata is not None and not isinstance(metadata, list):
                metadata = [metadata]
        metadata_base_list = [NwbSegmentationExtractor.get_nwb_metadata(sgobj) for sgobj in segext_objs]

        print(f'writing nwb for {segext_obj.extractor_name}\n')
        # updating base metadata with new:
        for num, data in enumerate(metadata_base_list):
            metadata_input = metadata[num] if metadata else {}
            metadata_base_list[num] = dict_recursive_update(metadata_base_list[num], metadata_input)
        # loop for every plane:
        with NWBHDF5IO(str(save_path), file_mode) as io:
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
                if metadata['Ophys']['Device'][0]['name'] not in nwbfile.devices:
                    nwbfile.create_device(**metadata['Ophys']['Device'][0])

                # ImageSegmentation:
                image_segmentation_name = 'ImageSegmentation' if plane_no_loop == 0 else f'ImageSegmentation_Plane{plane_no_loop}'
                if image_segmentation_name not in ophys.data_interfaces:
                    image_segmentation = ImageSegmentation(name=image_segmentation_name)
                    ophys.add(image_segmentation)
                else:
                    image_segmentation = ophys.data_interfaces.get(image_segmentation_name)

                # OpticalChannel:
                optical_channels = [OpticalChannel(**i) for i in
                                    metadata['Ophys']['ImagingPlane'][0]['optical_channel']]

                # ImagingPlane:
                image_plane_name = 'ImagingPlane' if plane_no_loop == 0 else f'ImagePlane_{plane_no_loop}'
                if image_plane_name not in nwbfile.imaging_planes.keys():
                    input_kwargs = dict(
                        name=image_plane_name,
                        device=nwbfile.get_device(metadata_base_common['Ophys']['Device'][0]['name']),
                    )
                    metadata['Ophys']['ImagingPlane'][0]['optical_channel'] = optical_channels
                    input_kwargs.update(**metadata['Ophys']['ImagingPlane'][0])
                    if 'imaging_rate' in input_kwargs:
                        input_kwargs['imaging_rate'] = float(input_kwargs['imaging_rate'])
                    imaging_plane = nwbfile.create_imaging_plane(**input_kwargs)
                else:
                    imaging_plane = nwbfile.imaging_planes[image_plane_name]

                # PlaneSegmentation:
                input_kwargs = dict(
                    description='output from segmenting imaging plane',
                    imaging_plane=imaging_plane
                )
                ps_metadata = metadata['Ophys']['ImageSegmentation']['plane_segmentations'][0]
                if ps_metadata['name'] not in image_segmentation.plane_segmentations:
                    input_kwargs.update(**ps_metadata)
                    ps = image_segmentation.create_plane_segmentation(**input_kwargs)
                    ps_exist = False
                else:
                    ps = image_segmentation.get_plane_segmentation(ps_metadata['name'])
                    ps_exist = True

                # ROI add:
                image_masks = segext_obj.get_roi_image_masks()
                roi_ids = segext_obj.get_roi_ids()
                accepted_list = segext_obj.get_accepted_list()
                accepted_list = [] if accepted_list is None else accepted_list
                rejected_list = segext_obj.get_rejected_list()
                rejected_list = [] if rejected_list is None else rejected_list
                accepted_ids = [1 if k in accepted_list else 0 for k in roi_ids]
                rejected_ids = [1 if k in rejected_list else 0 for k in roi_ids]
                roi_locations = np.array(segext_obj.get_roi_locations()).T
                if not ps_exist:
                    ps.add_column(name='RoiCentroid',
                                  description='x,y location of centroid of the roi in image_mask')
                    ps.add_column(name='Accepted',
                                  description='1 if ROi was accepted or 0 if rejected as a cell during segmentation operation')
                    ps.add_column(name='Rejected',
                                  description='1 if ROi was rejected or 0 if accepted as a cell during segmentation operation')
                    for num, row in enumerate(roi_ids):
                        ps.add_roi(id=row, image_mask=image_masks[:, :, num],
                                   RoiCentroid=roi_locations[num, :],
                                   Accepted=accepted_ids[num], Rejected=rejected_ids[num])

                # Fluorescence Traces:
                if 'Flourescence' not in ophys.data_interfaces:
                    fluorescence = Fluorescence()
                    ophys.add(fluorescence)
                else:
                    fluorescence = ophys.data_interfaces['Fluorescence']
                roi_response_dict = segext_obj.get_traces_dict()
                roi_table_region = ps.create_roi_table_region(description=f'region for Imaging plane{plane_no_loop}',
                                                              region=list(range(segext_obj.get_num_rois())))
                rate = np.float(
                    'NaN') if segext_obj.get_sampling_frequency() is None else segext_obj.get_sampling_frequency()
                for i, j in roi_response_dict.items():
                    data = getattr(segext_obj, f'_roi_response_{i}')
                    if data is not None:
                        data = np.asarray(data)
                        trace_name = 'RoiResponseSeries' if i == 'raw' else i.capitalize()
                        trace_name = trace_name if plane_no_loop == 0 else trace_name + f'_Plane{plane_no_loop}'
                        input_kwargs = dict(name=trace_name, data=data.T, rois=roi_table_region, rate=rate, unit='n.a.')
                        if trace_name not in fluorescence.roi_response_series:
                            fluorescence.create_roi_response_series(**input_kwargs)

                # create Two Photon Series:
                if 'TwoPhotonSeries' not in nwbfile.acquisition:
                    warn('could not find TwoPhotonSeries, using ImagingExtractor to create an nwbfile')

                # adding images:
                images_dict = segext_obj.get_images_dict()
                if any([image is not None for image in images_dict.values()]):
                    images_name = 'SegmentationImages' if plane_no_loop == 0 else f'SegmentationImages_Plane{plane_no_loop}'
                    if images_name not in ophys.data_interfaces:
                        images = Images(images_name)
                        for img_name, img_no in images_dict.items():
                            if img_no is not None:
                                images.add_image(GrayscaleImage(name=img_name, data=img_no))
                        ophys.add(images)

            # saving NWB file:
            io.write(nwbfile)

        # test read
        with NWBHDF5IO(str(save_path), 'r') as io:
            io.read()
