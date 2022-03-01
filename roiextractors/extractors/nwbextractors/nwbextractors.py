from collections import abc
from pathlib import Path
from typing import Union

import numpy as np
from lazy_ops import DatasetView

try:
    from pynwb import NWBHDF5IO
    from pynwb.ophys import TwoPhotonSeries

    HAVE_NWB = True
except ImportError:
    HAVE_NWB = False
from ...extraction_tools import (
    NumpyArray,
    PathType,
    FloatType,
    IntType,
    check_get_frames_args,
    check_get_videos_args,
)
from ...imagingextractor import ImagingExtractor
from ...segmentationextractor import SegmentationExtractor


def temporary_deprecation_message():
    raise NotImplementedError(
        "ROIExtractors no longer supports direct write to NWB. This method will be removed in a future release.\n\n"
        "Please install nwb-conversion-tools and import the corresponding write method from there.\n\nFor example,\n\n"
        "from roiextractors import NwbSegmentationExtractor\nNwbSegmentationExtractor.write_segmentation(...)\n\n"
        "would become\n\nfrom nwb_conversion_tools import roiextractors\nroiextractors.write_segmentation(...)"
    )


def check_nwb_install():
    assert (
        HAVE_NWB
    ), "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"


class NwbImagingExtractor(ImagingExtractor):
    """
    Class used to extract data from the NWB data format. Also implements a
    static method to write any format specific object to NWB.
    """

    extractor_name = "NwbImaging"
    installed = HAVE_NWB  # check at class level if installed or not
    is_writable = True
    mode = "file"
    installation_mesg = "To use the Nwb Extractor run:\n\n pip install pynwb\n\n"  # error message when not installed

    def __init__(
        self, file_path: PathType, optical_series_name: str = "TwoPhotonSeries"
    ):
        """
        Parameters
        ----------
        file_path: str
            The location of the folder containing dataset.nwb file
        optical_series_name: str (optional)
            optical series to extract data from
        """
        ImagingExtractor.__init__(self)
        self._path = file_path

        self.io = NWBHDF5IO(str(self._path), "r")
        self.nwbfile = self.io.read()
        if optical_series_name is not None:
            self._optical_series_name = optical_series_name
        else:
            a_names = list(self.nwbfile.acquisition)
            if len(a_names) > 1:
                raise ValueError(
                    "More than one acquisition found. You must specify two_photon_series."
                )
            if len(a_names) == 0:
                raise ValueError("No acquisitions found in the .nwb file.")
            self._optical_series_name = a_names[0]
        opts = self.nwbfile.acquisition[self._optical_series_name]
        assert isinstance(
            opts, TwoPhotonSeries
        ), "The optical series must be of type pynwb.TwoPhotonSeries"

        # TODO if external file --> return another proper extractor (e.g. TiffImagingExtractor)
        assert opts.external_file is None, "Only 'raw' format is currently supported"

        if len(opts.data.shape) == 3:
            self._num_frames, self._size_x, self._size_y = opts.data.shape
            self._channel_names = [i.name for i in opts.imaging_plane.optical_channel]
            self._num_channels = len(self._channel_names)
        else:
            raise NotImplementedError("4D volumetric data are currently not supported")
        if hasattr(opts, "timestamps") and opts.timestamps:
            self._sampling_frequency = 1.0 / np.median(np.diff(opts.timestamps))
            self._imaging_start_time = opts.timestamps[0]
            self.set_times(np.array(opts.timestamps))
        else:
            self._sampling_frequency = opts.rate
            self._imaging_start_time = opts.fields.get("starting_time", 0.0)
        # Fill epochs dictionary
        self._epochs = {}
        if self.nwbfile.epochs is not None:
            df_epochs = self.nwbfile.epochs.to_dataframe()
            # TODO implement add_epoch() method in base class
            self._epochs = {
                row["tags"][0]: {
                    "start_frame": self.time_to_frame(row["start_time"]),
                    "end_frame": self.time_to_frame(row["stop_time"]),
                }
                for _, row in df_epochs.iterrows()
            }
        self._kwargs = {
            "file_path": str(Path(file_path).absolute()),
            "optical_series_name": optical_series_name,
        }

    def __del__(self):
        self.io.close()

    def time_to_frame(self, times: Union[FloatType, NumpyArray]):
        if self._times is None:
            return (
                (times - self._imaging_start_time) * self.get_sampling_frequency()
            ).astype("int64")
        else:
            return super().time_to_frame(times)

    def frame_to_time(self, frames: Union[IntType, NumpyArray]):
        if self._times is None:
            return (
                frames / self.get_sampling_frequency() + self._imaging_start_time
            ).astype("float")
        else:
            return super().frame_to_time(frames)

    def make_nwb_metadata(self, nwbfile, opts):
        # Metadata dictionary - useful for constructing a nwb file
        self.nwb_metadata = dict(
            NWBFile=dict(
                session_description=nwbfile.session_description,
                identifier=nwbfile.identifier,
                session_start_time=nwbfile.session_start_time,
                institution=nwbfile.institution,
                lab=nwbfile.lab,
            ),
            Ophys=dict(
                Device=[dict(name=dev) for dev in nwbfile.devices],
                TwoPhotonSeries=[dict(name=opts.name)],
            ),
        )

    @check_get_frames_args
    def get_frames(self, frame_idxs, channel=0):
        opts = self.nwbfile.acquisition[self._optical_series_name]
        if frame_idxs.size > 1 and np.all(np.diff(frame_idxs) > 0):
            return opts.data[frame_idxs].transpose([0, 2, 1])
        else:
            sorted_idxs = np.sort(frame_idxs)
            argsorted_idxs = np.argsort(frame_idxs)
            return opts.data[sorted_idxs][argsorted_idxs].transpose([0, 2, 1])

    @check_get_videos_args
    def get_video(self, start_frame=None, end_frame=None, channel=0):
        opts = self.nwbfile.acquisition[self._optical_series_name]
        video = opts.data[start_frame:end_frame].transpose([0, 2, 1])
        return video

    def get_image_size(self):
        return [self._size_y, self._size_x]

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
        temporary_deprecation_message()

    @staticmethod
    def add_two_photon_series(
        imaging, nwbfile, metadata, buffer_size=10, use_times=False
    ):
        temporary_deprecation_message()

    @staticmethod
    def add_epochs(imaging, nwbfile):
        temporary_deprecation_message()

    @staticmethod
    def get_nwb_metadata(imgextractor: ImagingExtractor):
        temporary_deprecation_message()

    @staticmethod
    def write_imaging(
        imaging: ImagingExtractor,
        save_path: PathType = None,
        nwbfile=None,
        metadata: dict = None,
        overwrite: bool = False,
        buffer_size: int = 10,
        use_times: bool = False,
    ):
        temporary_deprecation_message()


class NwbSegmentationExtractor(SegmentationExtractor):
    extractor_name = "NwbSegmentationExtractor"
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = "file"
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
            raise Exception("file does not exist")
        self.file_path = file_path
        self._image_masks = None
        self._roi_locs = None
        self._accepted_list = None
        self._rejected_list = None
        self._io = NWBHDF5IO(str(file_path), mode="r")
        self.nwbfile = self._io.read()

        ophys = self.nwbfile.processing.get("ophys")
        if ophys is None:
            raise Exception("could not find ophys processing module in nwbfile")
        else:
            # Extract roi_response:
            fluorescence = None
            dfof = None
            any_roi_response_series_found = False
            if "Fluorescence" in ophys.data_interfaces:
                fluorescence = ophys.data_interfaces["Fluorescence"]
            if "DfOverF" in ophys.data_interfaces:
                dfof = ophys.data_interfaces["DfOverF"]
            if fluorescence is None and dfof is None:
                raise Exception("could not find Fluorescence/DfOverF module in nwbfile")
            for trace_name in ["RoiResponseSeries", "Dff", "Neuropil", "Deconvolved"]:
                trace_name_segext = (
                    "raw" if trace_name == "RoiResponseSeries" else trace_name.lower()
                )
                container = dfof if trace_name == "Dff" else fluorescence
                if (
                    container is not None
                    and trace_name in container.roi_response_series
                ):
                    any_roi_response_series_found = True
                    setattr(
                        self,
                        f"_roi_response_{trace_name_segext}",
                        DatasetView(
                            container.roi_response_series[trace_name].data
                        ).lazy_transpose(),
                    )
                    if self._sampling_frequency is None:
                        self._sampling_frequency = container.roi_response_series[
                            trace_name
                        ].rate
            if not any_roi_response_series_found:
                raise Exception(
                    "could not find any of 'RoiResponseSeries'/'Dff'/'Neuropil'/'Deconvolved'"
                    "named RoiResponseSeries in nwbfile"
                )
            # Extract image_mask/background:
            if "ImageSegmentation" in ophys.data_interfaces:
                image_seg = ophys.data_interfaces["ImageSegmentation"]
                if (
                    "PlaneSegmentation" in image_seg.plane_segmentations
                ):  # this requirement in nwbfile is enforced
                    ps = image_seg.plane_segmentations["PlaneSegmentation"]
                    if "image_mask" in ps.colnames:
                        self._image_masks = DatasetView(
                            ps["image_mask"].data
                        ).lazy_transpose([2, 1, 0])
                    else:
                        raise Exception("could not find any image_masks in nwbfile")
                    if "RoiCentroid" in ps.colnames:
                        self._roi_locs = ps["RoiCentroid"]
                    if "Accepted" in ps.colnames:
                        self._accepted_list = ps["Accepted"].data[:]
                    if "Rejected" in ps.colnames:
                        self._rejected_list = ps["Rejected"].data[:]
                    self._roi_idx = np.array(ps.id.data)
                else:
                    raise Exception("could not find any PlaneSegmentation in nwbfile")
            # Extracting stores images as GrayscaleImages:
            if "SegmentationImages" in ophys.data_interfaces:
                images_container = ophys.data_interfaces["SegmentationImages"]
                if "correlation" in images_container.images:
                    self._image_correlation = (
                        images_container.images["correlation"].data[()].T
                    )
                if "mean" in images_container.images:
                    self._image_mean = images_container.images["mean"].data[()].T
        # Imaging plane:
        if "ImagingPlane" in self.nwbfile.imaging_planes:
            imaging_plane = self.nwbfile.imaging_planes["ImagingPlane"]
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
        return self._image_masks.shape[:2]

    @staticmethod
    def get_nwb_metadata(sgmextractor):
        temporary_deprecation_message()

    @staticmethod
    def write_segmentation(
        segext_obj: SegmentationExtractor,
        save_path: PathType = None,
        plane_num=0,
        metadata: dict = None,
        overwrite: bool = True,
        buffer_size: int = 10,
        nwbfile=None,
    ):
        temporary_deprecation_message()
