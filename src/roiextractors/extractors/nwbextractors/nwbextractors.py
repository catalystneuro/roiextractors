"""Imaging and segmentation extractors for NWB files.

Classes
-------
NwbImagingExtractor
    Extracts imaging data from NWB files.
NwbSegmentationExtractor
    Extracts segmentation data from NWB files.
"""

import warnings
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
from lazy_ops import DatasetView
from pynwb import NWBHDF5IO
from pynwb.ophys import OnePhotonSeries, TwoPhotonSeries

from ...extraction_tools import (
    ArrayType,
    FloatType,
    IntType,
    PathType,
    raise_multi_channel_or_depth_not_implemented,
)
from ...imagingextractor import ImagingExtractor
from ...segmentationextractor import SegmentationExtractor


def temporary_deprecation_message():
    """Issue a deprecation warning and raise a NotImplementedError with a migration message."""
    from warnings import warn

    warn(
        "The write_imaging function is deprecated and will be removed on or after September 2025. ROIExtractors is no longer supporting write operations.",
        DeprecationWarning,
        stacklevel=2,
    )

    raise NotImplementedError(
        "ROIExtractors no longer supports direct write to NWB. This method will be removed in a future release.\n\n"
        "Please install neuroconv and import the corresponding write method from there.\n\nFor example,\n\n"
        "from roiextractors import NwbSegmentationExtractor\nNwbSegmentationExtractor.write_segmentation(...)\n\n"
        "would become\n\nfrom neuroconv import roiextractors\nroiextractors.write_segmentation(...)"
    )


class NwbImagingExtractor(ImagingExtractor):
    """An imaging extractor for NWB files.

    Class used to extract data from the NWB data format. Also implements a
    static method to write any format specific object to NWB.
    """

    extractor_name = "NwbImaging"
    mode = "file"

    def __init__(self, file_path: PathType, optical_series_name: Optional[str] = "TwoPhotonSeries"):
        """Create ImagingExtractor object from NWB file.

        Parameters
        ----------
        file_path: str
            The location of the folder containing dataset.nwb file
        optical_series_name: string, optional
            The name of the optical series to extract data from.
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
                raise ValueError("More than one acquisition found. You must specify two_photon_series.")
            if len(a_names) == 0:
                raise ValueError("No acquisitions found in the .nwb file.")
            self._optical_series_name = a_names[0]

        self.photon_series = self.nwbfile.acquisition[self._optical_series_name]
        valid_photon_series_types = [OnePhotonSeries, TwoPhotonSeries]
        assert any(
            [isinstance(self.photon_series, photon_series_type) for photon_series_type in valid_photon_series_types]
        ), "The optical series must be of type pynwb.ophys.OnePhotonSeries or pynwb.ophys.TwoPhotonSeries."

        # TODO if external file --> return another proper extractor (e.g. TiffImagingExtractor)
        assert self.photon_series.external_file is None, "Only 'raw' format is currently supported"

        # Load the two video structures that TwoPhotonSeries supports.
        self._data_has_channels_axis = True
        if len(self.photon_series.data.shape) == 3:
            self._num_channels = 1
            self._num_samples, self._columns, self._num_rows = self.photon_series.data.shape
        else:
            raise_multi_channel_or_depth_not_implemented(extractor_name=self.extractor_name)

        # Set channel names (This should disambiguate which optical channel)
        self._channel_names = [i.name for i in self.photon_series.imaging_plane.optical_channel]

        # Set sampling frequency
        if hasattr(self.photon_series, "timestamps") and self.photon_series.timestamps:
            self._sampling_frequency = 1.0 / np.median(np.diff(self.photon_series.timestamps))
            self._imaging_start_time = self.photon_series.timestamps[0]
            self.set_times(np.array(self.photon_series.timestamps))
        else:
            self._sampling_frequency = self.photon_series.rate
            self._imaging_start_time = self.photon_series.fields.get("starting_time", 0.0)

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
        """Close the NWB file."""
        self.io.close()

    def time_to_frame(self, times: Union[FloatType, ArrayType]) -> np.ndarray:
        if self._times is None:
            return ((times - self._imaging_start_time) * self.get_sampling_frequency()).astype("int64")
        else:
            return super().time_to_frame(times)

    def frame_to_time(self, frames: Union[IntType, ArrayType]) -> np.ndarray:
        warnings.warn(
            "frame_to_time() is deprecated and will be removed on or after January 2026. "
            "Use sample_indices_to_time() instead.",
            FutureWarning,
            stacklevel=2,
        )
        if self._times is None:
            return (frames / self.get_sampling_frequency() + self._imaging_start_time).astype("float")
        else:
            return super().frame_to_time(frames)

    def make_nwb_metadata(
        self, nwbfile, opts
    ):  # TODO: refactor to use two photon series name directly rather than via opts
        """Create metadata dictionary for NWB file.

        Parameters
        ----------
        nwbfile: pynwb.NWBFile
            The NWBFile object associated with the metadata.
        opts: object
            The options object with name of TwoPhotonSeries as an attribute.

        Notes
        -----
        Metadata dictionary is stored in the nwb_metadata attribute.
        """
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

    def get_frames(self, frame_idxs: ArrayType, channel: Optional[int] = 0):
        """Get specific video frames from indices.

        Parameters
        ----------
        frame_idxs: array-like
            Indices of frames to return.
        channel: int, optional
            Channel index. Deprecated: This parameter will be removed in August 2025.

        Returns
        -------
        frames: numpy.ndarray
            The video frames.
        """
        if channel != 0:
            from warnings import warn

            warn(
                "The 'channel' parameter in get_frames() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
        squeeze_data = False
        if isinstance(frame_idxs, int):
            squeeze_data = True
            frame_idxs = [frame_idxs]
        elif isinstance(frame_idxs, np.ndarray):
            frame_idxs = frame_idxs.tolist()
        frames = self.photon_series.data[frame_idxs].transpose([0, 2, 1])
        if squeeze_data:
            frames = frames.squeeze()
        return frames

    def get_series(self, start_sample=None, end_sample=None) -> np.ndarray:
        start_sample = start_sample if start_sample is not None else 0
        end_sample = end_sample if end_sample is not None else self.get_num_samples()

        series = self.photon_series.data
        series = series[start_sample:end_sample].transpose([0, 2, 1])
        return series

    def get_video(self, start_frame=None, end_frame=None, channel: Optional[int] = 0) -> np.ndarray:
        """Get the video frames.

        Parameters
        ----------
        start_frame: int, optional
            Start frame index (inclusive).
        end_frame: int, optional
            End frame index (exclusive).
        channel: int, optional
            Channel index. Deprecated: This parameter will be removed in August 2025.

        Returns
        -------
        video: numpy.ndarray
            The video frames.

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_series() instead.
        """
        warnings.warn(
            "get_video() is deprecated and will be removed in or after September 2025. " "Use get_series() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if channel != 0:
            from warnings import warn

            warn(
                "The 'channel' parameter in get_video() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.get_series(start_sample=start_frame, end_sample=end_frame)

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return (self._num_rows, self._columns)  # TODO: change name of _columns to _num_cols for consistency

    def get_image_size(self) -> Tuple[int, int]:
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        return (self._num_rows, self._columns)  # TODO: change name of _columns to _num_cols for consistency

    def get_num_samples(self):
        return self._num_samples

    def get_num_frames(self):
        """Get the number of frames in the video.

        Returns
        -------
        num_frames: int
            Number of frames in the video.

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_num_samples() instead.
        """
        warnings.warn(
            "get_num_frames() is deprecated and will be removed in or after September 2025. "
            "Use get_num_samples() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_num_samples()

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_channel_names(self):
        return self._channel_names

    def get_num_channels(self):
        return self._num_channels

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # NWB files may have timestamps but need to check the specific implementation
        # For now, return None to use calculated timestamps based on sampling frequency
        # TODO: extract the timestamps if the MiroscopySeries has timestamps
        return None

    @staticmethod
    def add_devices(imaging, nwbfile, metadata):
        """Add devices to the NWBFile (deprecated)."""
        temporary_deprecation_message()

    @staticmethod
    def add_two_photon_series(imaging, nwbfile, metadata, buffer_size=10, use_times=False):
        """Add TwoPhotonSeries to NWBFile (deprecated)."""
        temporary_deprecation_message()

    @staticmethod
    def add_epochs(imaging, nwbfile):
        """Add epochs to NWBFile (deprecated)."""
        temporary_deprecation_message()

    @staticmethod
    def get_nwb_metadata(imgextractor: ImagingExtractor):
        """Return the metadata dictionary for the NWB file (deprecated)."""
        temporary_deprecation_message()


class NwbSegmentationExtractor(SegmentationExtractor):
    """An segmentation extractor for NWB files."""

    extractor_name = "NwbSegmentationExtractor"
    mode = "file"
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path: PathType):
        """Create NwbSegmentationExtractor object from nwb file.

        Parameters
        ----------
        file_path: PathType
            .nwb file location
        """
        super().__init__()
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

        assert "ophys" in self.nwbfile.processing, "Ophys processing module is not in nwbfile."
        ophys = self.nwbfile.processing.get("ophys")

        # Extract roi_responses:
        fluorescence = None
        df_over_f = None
        any_roi_response_series_found = False
        if "Fluorescence" in ophys.data_interfaces:
            fluorescence = ophys.data_interfaces["Fluorescence"]
        if "DfOverF" in ophys.data_interfaces:
            df_over_f = ophys.data_interfaces["DfOverF"]
        if fluorescence is None and df_over_f is None:
            raise Exception("Could not find Fluorescence/DfOverF module in nwbfile.")
        for trace_name in self.get_traces_dict().keys():
            trace_name_segext = "RoiResponseSeries" if trace_name in ["raw", "dff"] else trace_name.capitalize()
            container = df_over_f if trace_name == "dff" else fluorescence
            if container is not None and trace_name_segext in container.roi_response_series:
                any_roi_response_series_found = True
                setattr(
                    self,
                    f"_roi_response_{trace_name}",
                    DatasetView(container.roi_response_series[trace_name_segext].data),
                )
                if self._sampling_frequency is None:
                    self._sampling_frequency = container.roi_response_series[trace_name_segext].rate
        if not any_roi_response_series_found:
            raise Exception(
                "could not find any of 'RoiResponseSeries'/'Dff'/'Neuropil'/'Deconvolved'"
                "named RoiResponseSeries in nwbfile"
            )
        # Extract image_mask/background:
        if "ImageSegmentation" in ophys.data_interfaces:
            image_seg = ophys.data_interfaces["ImageSegmentation"]
        assert len(image_seg.plane_segmentations), "Could not find any PlaneSegmentation in nwbfile."
        if "PlaneSegmentation" in image_seg.plane_segmentations:  # this requirement in nwbfile is enforced
            ps = image_seg.plane_segmentations["PlaneSegmentation"]
            assert "image_mask" in ps.colnames, "Could not find any image_masks in nwbfile."
            self._image_masks = DatasetView(ps["image_mask"].data).lazy_transpose([2, 1, 0])
            self._roi_locs = ps["ROICentroids"] if "ROICentroids" in ps.colnames else None
            self._accepted_list = ps["Accepted"].data[:] if "Accepted" in ps.colnames else None
            self._rejected_list = ps["Rejected"].data[:] if "Rejected" in ps.colnames else None

        # Extracting stored images as GrayscaleImages:
        self._segmentation_images = None
        if "SegmentationImages" in ophys.data_interfaces:
            images_container = ophys.data_interfaces["SegmentationImages"]
            self._segmentation_images = images_container.images
        # Imaging plane:
        if "ImagingPlane" in self.nwbfile.imaging_planes:
            imaging_plane = self.nwbfile.imaging_planes["ImagingPlane"]
            self._channel_names = [i.name for i in imaging_plane.optical_channel]

    def __del__(self):
        """Close the NWB file."""
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

    def get_images_dict(self):
        """Return traces as a dictionary with key as the name of the ROIResponseSeries.

        Returns
        -------
        images_dict: dict
            dictionary with key, values representing different types of Images used in segmentation:
            Mean, Correlation image
        """
        images_dict = super().get_images_dict()
        if self._segmentation_images is not None:
            images_dict.update(
                (image_name, image_data[:].T) for image_name, image_data in self._segmentation_images.items()
            )

        return images_dict

    def get_roi_locations(self, roi_ids: Optional[Iterable[int]] = None) -> np.ndarray:
        """Return the locations of the Regions of Interest (ROIs).

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

        Returns
        -------
        roi_locs: numpy.ndarray
            2-D array: 2 X no_ROIs. The pixel ids (x,y) where the centroid of the ROI is.
        """
        if self._roi_locs is None:
            return
        all_ids = self.get_roi_ids()
        roi_idxs = slice(None) if roi_ids is None else [all_ids.index(i) for i in roi_ids]
        # ROIExtractors uses height x width x (depth), but NWB uses width x height x depth
        tranpose_image_convention = (1, 0) if len(self.get_image_size()) == 2 else (1, 0, 2)
        return np.array(self._roi_locs.data)[roi_idxs, tranpose_image_convention].T  # h5py fancy indexing is slow

    def get_frame_shape(self):
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        frame_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._image_masks.shape[:2]

    def get_image_shape(self):
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._image_masks.shape[:2]

    def get_image_size(self):
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._image_masks.shape[:2]

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # NWB files may have timestamps but need to check the specific implementation
        # For now, return None to use calculated timestamps based on sampling frequency
        # TODO: check if the RoiResponseSeries has timestamps
        return None

    @staticmethod
    def get_nwb_metadata(sgmextractor):
        """Return the metadata dictionary for the NWB file (deprecated)."""
        temporary_deprecation_message()
