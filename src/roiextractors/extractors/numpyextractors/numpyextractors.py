"""Imaging and Segmentation Extractors for .npy files.

Classes
-------
NumpyImagingExtractor
    An ImagingExtractor specified by timeseries .npy file, sampling frequency, and channel names.
NumpySegmentationExtractor
    A Segmentation extractor specified by image masks and traces .npy files.
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple
from warnings import warn

import numpy as np

from ...extraction_tools import ArrayType, FloatType, PathType
from ...imagingextractor import ImagingExtractor
from ...segmentationextractor import SegmentationExtractor


class NumpyImagingExtractor(ImagingExtractor):
    """An ImagingExtractor specified by timeseries .npy file, sampling frequency, and channel names."""

    extractor_name = "NumpyImagingExtractor"
    installation_mesg = ""  # error message when not installed

    def __init__(
        self,
        timeseries: PathType,
        sampling_frequency: FloatType,
        channel_names: ArrayType = None,
    ):
        """Create a NumpyImagingExtractor from a .npy file.

        Parameters
        ----------
        timeseries: PathType
            Path to .npy file.
        sampling_frequency: FloatType
            Sampling frequency of the video in Hz.
        channel_names: ArrayType
            List of channel names.
        """
        ImagingExtractor.__init__(self)

        if isinstance(timeseries, (str, Path)):
            timeseries = Path(timeseries)
            if timeseries.is_file():
                assert timeseries.suffix == ".npy", "'timeseries' file is not a numpy file (.npy)"
                self.is_dumpable = True
                self._video = np.load(timeseries, mmap_mode="r")
                self._kwargs = {
                    "timeseries": str(Path(timeseries).absolute()),
                    "sampling_frequency": sampling_frequency,
                }
            else:
                raise ValueError("'timeseries' is does not exist")
        elif isinstance(timeseries, np.ndarray):
            self.is_dumpable = False
            self._video = timeseries
            self._kwargs = {
                "timeseries": timeseries,
                "sampling_frequency": sampling_frequency,
            }
        else:
            raise TypeError("'timeseries' can be a str or a numpy array")

        self._sampling_frequency = float(sampling_frequency)

        self._sampling_frequency = sampling_frequency
        self._channel_names = channel_names

        (
            self._num_samples,
            self._num_rows,
            self._num_columns,
            self._num_channels,
        ) = self.get_volume_shape(self._video)

        if len(self._video.shape) == 3:
            # check if this converts to np.ndarray
            self._video = self._video[np.newaxis, :]

        if self._channel_names is not None:
            assert len(self._channel_names) == self._num_channels, (
                "'channel_names' length is different than number " "of channels"
            )
        else:
            self._channel_names = [f"channel_{ch}" for ch in range(self._num_channels)]

    @staticmethod
    def get_volume_shape(video) -> Tuple[int, int, int, int]:
        """Get the shape of a video (num_frames, num_rows, num_columns, num_channels).

        Parameters
        ----------
        video: numpy.ndarray
            The video to get the shape of.

        Returns
        -------
        video_shape: tuple
            The shape of the video (num_frames, num_rows, num_columns, num_channels).
        """
        if len(video.shape) == 3:
            # 1 channel
            num_channels = 1
            num_frames, num_rows, num_columns = video.shape
        else:
            num_frames, num_rows, num_columns, num_channels = video.shape
        return num_frames, num_rows, num_columns, num_channels

    def get_frames(self, frame_idxs=None, channel: Optional[int] = 0) -> np.ndarray:
        """Get specific video frames from indices.

        Parameters
        ----------
        frame_idxs: array-like, optional
            Indices of frames to return. If None, returns all frames.
        channel: int, optional
            Channel index. Deprecated: This parameter will be removed in August 2025.

        Returns
        -------
        frames: numpy.ndarray
            The video frames.
        """
        if channel != 0:
            warn(
                "The 'channel' parameter in get_frames() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
        if frame_idxs is None:
            frame_idxs = [frame for frame in range(self.get_num_frames())]

        frames = self._video.take(indices=frame_idxs, axis=0)
        if channel is not None:
            frames = frames[..., channel].squeeze()

        return frames

    def get_series(self, start_sample=None, end_sample=None) -> np.ndarray:
        return self._video[start_sample:end_sample, ..., 0]

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
            warn(
                "The 'channel' parameter in get_video() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self._video[start_frame:end_frame, ..., channel]

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return (self._num_rows, self._num_columns)

    def get_image_size(self) -> Tuple[int, int]:
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        return (self._num_rows, self._num_columns)

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
        # Numpy arrays do not have native timestamps
        return None

    @staticmethod
    def write_imaging(imaging, save_path, overwrite: bool = False):
        """Write a NumpyImagingExtractor to a .npy file.

        Parameters
        ----------
        imaging: NumpyImagingExtractor
            The imaging extractor object to be written to file.
        save_path: str or PathType
            Path to .npy file.
        overwrite: bool
            If True, overwrite file if it already exists.
        """
        warn(
            "The write_imaging function is deprecated and will be removed on or after September 2025. ROIExtractors is no longer supporting write operations.",
            DeprecationWarning,
            stacklevel=2,
        )
        save_path = Path(save_path)
        assert save_path.suffix == ".npy", "'save_path' should have a .npy extension"

        if save_path.is_file():
            if not overwrite:
                raise FileExistsError("The specified path exists! Use overwrite=True to overwrite it.")
            else:
                save_path.unlink()

        np.save(save_path, imaging.get_video())


class NumpySegmentationExtractor(SegmentationExtractor):
    """A Segmentation extractor specified by image masks and traces .npy files.

    NumpySegmentationExtractor objects are built to contain all data coming from
    a file format for which there is currently no support. To construct this,
    all data must be entered manually as arguments.
    """

    extractor_name = "NumpySegmentationExtractor"
    mode = "file"
    installation_mesg = ""  # error message when not installed

    def __init__(
        self,
        image_masks,
        raw=None,
        dff=None,
        deconvolved=None,
        neuropil=None,
        accepted_lst=None,
        mean_image=None,
        correlation_image=None,
        roi_ids=None,
        roi_locations=None,
        sampling_frequency=None,
        rejected_list=None,
        channel_names=None,
        movie_dims=None,
        accepted_list=None,
    ):
        """Create a NumpySegmentationExtractor from a .npy file.

        Parameters
        ----------
        image_masks: np.ndarray
            Binary image for each of the regions of interest
        raw: np.ndarray
            Fluorescence response of each of the ROI in time
        dff: np.ndarray
            DfOverF response of each of the ROI in time
        deconvolved: np.ndarray
            deconvolved response of each of the ROI in time
        neuropil: np.ndarray
            neuropil response of each of the ROI in time
        mean_image: np.ndarray
            Mean image
        correlation_image: np.ndarray
            correlation image
        roi_ids: int list
            Unique ids of the ROIs if any
        roi_locations: np.ndarray
            x and y location representative of ROI mask
        sampling_frequency: float
            Frame rate of the movie
        rejected_list: list
            list of ROI ids that are rejected manually or via automated rejection
        channel_names: list
            list of strings representing channel names
        movie_dims: tuple
            height x width of the movie
        """
        accepted_list = accepted_lst if accepted_lst is not None else accepted_list
        if accepted_lst is not None:
            warnings.warn(
                "The 'accepted_lst' parameter is deprecated and will be removed on or after January 2026. "
                "Use 'accepted_list' instead.",
                FutureWarning,
                stacklevel=2,
            )

        SegmentationExtractor.__init__(self)
        if isinstance(image_masks, (str, Path)):
            image_masks = Path(image_masks)
            if image_masks.is_file():
                assert image_masks.suffix == ".npy", "'image_masks' file is not a numpy file (.npy)"

                self.is_dumpable = True
                self._image_masks = np.load(image_masks, mmap_mode="r")

                if raw is not None:
                    raw = Path(raw)
                    assert raw.suffix == ".npy", "'raw' file is not a numpy file (.npy)"
                    self._roi_response_raw = np.load(raw, mmap_mode="r")
                if dff is not None:
                    dff = Path(dff)
                    assert dff.suffix == ".npy", "'dff' file is not a numpy file (.npy)"
                    self._roi_response_dff = np.load(dff, mmap_mode="r")
                    self._roi_response_neuropil = np.load(neuropil, mmap_mode="r")
                if deconvolved is not None:
                    deconvolved = Path(deconvolved)
                    assert deconvolved.suffix == ".npy", "'deconvolved' file is not a numpy file (.npy)"
                    self._roi_response_deconvolved = np.load(deconvolved, mmap_mode="r")
                if neuropil is not None:
                    neuropil = Path(neuropil)
                    assert neuropil.suffix == ".npy", "'neuropil' file is not a numpy file (.npy)"
                    self._roi_response_neuropil = np.load(neuropil, mmap_mode="r")

                self._kwargs = {"image_masks": str(Path(image_masks).absolute())}
                if raw is not None:
                    self._kwargs.update({"raw": str(Path(raw).absolute())})
                if raw is not None:
                    self._kwargs.update({"dff": str(Path(dff).absolute())})
                if raw is not None:
                    self._kwargs.update({"neuropil": str(Path(neuropil).absolute())})
                if raw is not None:
                    self._kwargs.update({"deconvolved": str(Path(deconvolved).absolute())})

            else:
                raise ValueError("'timeeseries' is does not exist")
        elif isinstance(image_masks, np.ndarray):
            NoneType = type(None)
            assert isinstance(raw, (np.ndarray, NoneType))
            assert isinstance(dff, (np.ndarray, NoneType))
            assert isinstance(neuropil, (np.ndarray, NoneType))
            assert isinstance(deconvolved, (np.ndarray, NoneType))
            self.is_dumpable = False
            self._image_masks = image_masks
            self._roi_response_raw = raw
            assert self._image_masks.shape[-1] == self._roi_response_raw.shape[-1], (
                "Inconsistency between image masks and raw traces. "
                "Image masks must be (px, py, num_rois), "
                "traces must be (num_frames, num_rois)"
            )
            self._roi_response_dff = dff
            if self._roi_response_dff is not None:
                assert self._image_masks.shape[-1] == self._roi_response_dff.shape[-1], (
                    "Inconsistency between image masks and raw traces. "
                    "Image masks must be (px, py, num_rois), "
                    "traces must be (num_frames, num_rois)"
                )
            self._roi_response_neuropil = neuropil
            if self._roi_response_neuropil is not None:
                assert self._image_masks.shape[-1] == self._roi_response_neuropil.shape[-1], (
                    "Inconsistency between image masks and raw traces. "
                    "Image masks must be (px, py, num_rois), "
                    "traces must be (num_frames, num_rois)"
                )
            self._roi_response_deconvolved = deconvolved
            if self._roi_response_deconvolved is not None:
                assert self._image_masks.shape[-1] == self._roi_response_deconvolved.shape[-1], (
                    "Inconsistency between image masks and raw traces. "
                    "Image masks must be (px, py, num_rois), "
                    "traces must be (num_frames, num_rois)"
                )
            self._kwargs = {
                "image_masks": image_masks,
                "signal": raw,
                "dff": dff,
                "neuropil": neuropil,
                "deconvolved": deconvolved,
            }
        else:
            raise TypeError("'image_masks' can be a str or a numpy array")
        self._movie_dims = movie_dims if movie_dims is not None else image_masks.shape
        self._image_mean = mean_image
        self._image_correlation = correlation_image
        if roi_ids is None:
            self._roi_ids = list(np.arange(image_masks.shape[2]))
        else:
            self._roi_ids = roi_ids
        self._roi_locs = roi_locations
        self._sampling_frequency = sampling_frequency
        self._channel_names = channel_names
        self._rejected_list = rejected_list
        self._accepted_list = accepted_list

    @property
    def image_dims(self):
        """Return the dimensions of the image.

        Returns
        -------
        image_dims: list
            The dimensions of the image (num_rois, num_rows, num_columns).
        """
        return list(self._image_masks.shape[0:2])

    def get_accepted_list(self):
        if self._accepted_list is None:
            return self.get_roi_ids()
        else:
            return self._accepted_list

    def get_rejected_list(self):
        if self._rejected_list is None:
            return [a for a in self.get_roi_ids() if a not in set(self.get_accepted_list())]
        else:
            return self._rejected_list

    @property
    def roi_locations(self):
        """Returns the center locations (x, y) of each ROI."""
        if self._roi_locs is None:
            num_ROIs = self.get_num_rois()
            raw_images = self._image_masks
            roi_location = np.ndarray([2, num_ROIs], dtype="int")
            for i in range(num_ROIs):
                temp = np.where(raw_images[:, :, i] == np.amax(raw_images[:, :, i]))
                roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
            return roi_location
        else:
            return self._roi_locs

    @staticmethod
    def write_segmentation(segmentation_object, save_path):
        """Write a NumpySegmentationExtractor to a .npy file.

        Parameters
        ----------
        segmentation_object: NumpySegmentationExtractor
            The segmentation extractor object to be written to file.
        save_path: str or PathType
            Path to .npy file.

        Notes
        -----
        This method is not implemented yet.
        """
        warn(
            "The write_segmentation function is deprecated and will be removed on or after September 2025. ROIExtractors is no longer supporting write operations.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError

    # defining the abstract class informed methods:
    def get_roi_ids(self):
        if self._roi_ids is None:
            return list(range(self.get_num_rois()))
        else:
            return self._roi_ids

    def get_frame_shape(self):
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        frame_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._movie_dims

    def get_image_shape(self):
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        warnings.warn(
            "get_image_size is deprecated and will be removed on or after January 2026. "
            "Use get_frame_shape instead.",
            FutureWarning,
            stacklevel=2,
        )

        return self.get_frame_shape()

    def get_num_samples(self):
        """Get the number of samples in the recording (duration of recording).

        Returns
        -------
        num_samples: int
            Number of samples in the recording.
        """
        for trace in self.get_traces_dict().values():
            if trace is not None and len(trace.shape) > 0:
                return trace.shape[0]

    def get_image_size(self):
        warnings.warn(
            "get_image_size is deprecated and will be removed on or after January 2026. "
            "Use get_frame_shape instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_frame_shape()

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # Numpy arrays do not have native timestamps
        return None
