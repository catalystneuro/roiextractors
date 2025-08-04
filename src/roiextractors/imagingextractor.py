"""Base class definitions for all ImagingExtractors.

Classes
-------
ImagingExtractor
    Abstract class that contains all the meta-data and input data from the imaging data.
FrameSliceImagingExtractor
    Class to get a lazy frame slice.
"""

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from math import prod
from typing import Optional, Tuple, Union

import numpy as np

from .core_utils import _convert_bytes_to_str, _convert_seconds_to_str
from .extraction_tools import ArrayType, DtypeType, FloatType


class ImagingExtractor(ABC):
    """Abstract class that contains all the meta-data and input data from the imaging data."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the ImagingExtractor object."""
        self._args = args
        self._kwargs = kwargs
        self._times = None
        self.name = self.__class__.__name__
        self.is_volumetric = False

    def _repr_text(self) -> str:
        """Generate text representation of the ImagingExtractor object."""
        num_samples = self.get_num_samples()
        sample_shape = self.get_sample_shape()
        dtype = self.get_dtype()
        sf_hz = self.get_sampling_frequency()

        # Format sampling frequency
        if not sf_hz.is_integer():
            sampling_frequency_repr = f"{sf_hz:f} Hz"
        else:
            sampling_frequency_repr = f"{sf_hz:0.1f}Hz"

        # Calculate duration
        duration = num_samples / sf_hz
        duration_repr = _convert_seconds_to_str(duration)

        # Calculate memory size using product of all dimensions in image_size
        memory_size = num_samples * prod(sample_shape) * dtype.itemsize
        memory_repr = _convert_bytes_to_str(memory_size)

        # Format shape string based on whether data is volumetric or not
        sample_shape_repr = f"{sample_shape[0]} rows x {sample_shape[1]} columns "
        if self.is_volumetric:
            sample_shape_repr += f"x {sample_shape[2]} planes"

        return (
            f"{self.name}\n"
            f"  Number of samples: {num_samples:,} \n"
            f"  Sample shape: {sample_shape_repr} \n"
            f"  Sampling rate: {sampling_frequency_repr}\n"
            f"  Duration: {duration_repr}\n"
            f"  Imaging data memory: {memory_repr} ({dtype} dtype)"
        )

    def __repr__(self):
        return self._repr_text()

    @abstractmethod
    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        pass

    def get_image_size(self) -> Tuple:
        """Get the size of the video.

        Returns
        -------
        image_size: tuple
            Size of the video. For regular imaging extractors, this is (num_rows, num_columns).
            For volumetric imaging extractors, this is (num_rows, num_columns, num_planes).

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_image_shape() instead for consistent behavior across all extractors.
        """
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_image_shape()

    def get_frame_shape(self) -> tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        frame_shape: tuple
            Shape of the video frame (num_rows, num_columns).

        Notes
        -----
        This method is equivalent to get_image_shape()
        """
        return self.get_image_shape()

    def get_sample_shape(self) -> tuple[int, int] | tuple[int, int, int]:
        """
        Get the shape of a single sample elements from the series.

        If the series is volumetric, the shape is the shape of the volume and otherwise
        returns the shape of a single frame/image.

        Returns
        -------
        sample_shape: tuple
            Shape of a single sample from the time series (num_rows, num_columns, num_planes)
            if volumetric (num_rows, num_columns) otherwise
        """
        if self.is_volumetric:
            return self.get_volume_shape()
        else:
            return self.get_frame_shape()

    def get_num_planes(self) -> int:
        """Get the number of depth planes.

        Returns
        -------
        num_planes: int
            The number of depth planes.

        Raises
        ------
        NotImplementedError
            If the extractor is not volumetric.
        """
        if not self.is_volumetric:
            raise NotImplementedError(
                "This extractor is not volumetric. "
                "The get_num_planes method is only available for volumetric extractors."
            )
        raise NotImplementedError("This method must be implemented by volumetric extractor subclasses.")

    def get_volume_shape(self) -> Tuple[int, int, int]:
        """Get the shape of the volumetric video (num_rows, num_columns, num_planes).

        Returns
        -------
        video_shape: tuple
            Shape of the volumetric video (num_rows, num_columns, num_planes).

        Raises
        ------
        NotImplementedError
            If the extractor is not volumetric or does not implement get_num_planes method.
        """
        if not self.is_volumetric:
            raise NotImplementedError(
                "This extractor is not volumetric. "
                "The get_volume_shape method is only available for volumetric extractors."
            )

        if not hasattr(self, "get_num_planes") or not callable(getattr(self, "get_num_planes")):
            raise NotImplementedError(
                "This extractor does not implement get_num_planes method. "
                "The get_volume_shape method requires the get_num_planes method to be implemented."
            )

        frame_shape = self.get_frame_shape()
        return (frame_shape[0], frame_shape[1], self.get_num_planes())

    @abstractmethod
    def get_num_samples(self) -> int:
        """Get the number of samples in the video.

        Returns
        -------
        num_samples: int
            Number of samples in the video.
        """
        pass

    def get_num_frames(self) -> int:
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

    @abstractmethod
    def get_sampling_frequency(self) -> float:
        """Get the sampling frequency in Hz.

        Returns
        -------
        sampling_frequency: float
            Sampling frequency in Hz.
        """
        pass

    @abstractmethod
    def get_channel_names(self) -> list:
        """Get the channel names in the recoding.

        Returns
        -------
        channel_names: list
            List of strings of channel names
        """
        pass

    def get_num_channels(self) -> int:
        """Get the total number of active channels in the recording.

        Returns
        -------
        num_channels: int
            Integer count of number of channels.

        Deprecated
        ----------
        This method will be removed in or after August 2025.
        """
        warnings.warn(
            "get_num_channels() is deprecated and will be removed in or after August 2025.",
            DeprecationWarning,
            stacklevel=2,
        )

    def get_dtype(self) -> DtypeType:
        """Get the data type of the video.

        Returns
        -------
        dtype: dtype
            Data type of the video.
        """
        return self.get_series(start_sample=0, end_sample=2).dtype

    @abstractmethod
    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
        """Get the series of samples.

        Parameters
        ----------
        start_sample: int, optional
            Start sample index (inclusive).
        end_sample: int, optional
            End sample index (exclusive).

        Returns
        -------
        series: numpy.ndarray
            The series of samples.

        Notes
        -----
        Importantly, we follow the convention that the dimensions of the array are returned in their matrix order,
        More specifically:
        (time, height, width)

        Which is equivalent to:
        (samples, rows, columns)

        For volumetric data, the dimensions are:
        (time, height, width, planes)

        Which is equivalent to:
        (samples, rows, columns, planes)

        Note that this does not match the cartesian convention:
        (t, x, y)

        Where x is the columns width or and y is the rows or height.
        """
        pass

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
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
            warnings.warn(
                "The 'channel' parameter in get_video() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.get_series(start_sample=start_frame, end_sample=end_frame)

    def get_samples(self, sample_indices: ArrayType) -> np.ndarray:
        """Get specific samples from indices (not necessarily continuous).

        Parameters
        ----------
        sample_indices: array-like
            Indices of samples to return.

        Returns
        -------
        samples: numpy.ndarray
            The samples.
        """
        assert (
            max(sample_indices) < self.get_num_samples()
        ), "'sample_indices' range beyond number of available samples!"
        if np.all(np.diff(sample_indices) == 0):
            return self.get_series(start_sample=sample_indices[0], end_sample=sample_indices[-1])
        relative_indices = np.array(sample_indices) - sample_indices[0]
        series = self.get_series(start_sample=sample_indices[0], end_sample=sample_indices[-1] + 1)
        return series[relative_indices]

    def get_frames(self, frame_idxs: ArrayType, channel: Optional[int] = 0) -> np.ndarray:
        """Get specific video frames from indices (not necessarily continuous).

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

        Deprecated
        ----------
        This method will be removed on or after January 2026.
        Use get_samples() instead.
        """
        warnings.warn(
            "get_frames() is deprecated and will be removed on or after January 2026. " "Use get_samples() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if channel != 0:
            warnings.warn(
                "The 'channel' parameter in get_frames() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.get_samples(sample_indices=frame_idxs)

    @abstractmethod
    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Retrieve the original unaltered timestamps for the data in this interface.

        This function should retrieve the data on-demand by re-initializing the IO.
        Can be overridden to return None if the extractor does not have native timestamps.

        Parameters
        ----------
        start_sample : int, optional
            The starting sample index. If None, starts from the beginning.
        end_sample : int, optional
            The ending sample index. If None, goes to the end.

        Returns
        -------
        timestamps: numpy.ndarray or None
            The timestamps for the data stream, or None if native timestamps are not available.
        """
        return None

    def get_timestamps(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
        """
        Retrieve the timestamps for the data in this extractor.

        Parameters
        ----------
        start_sample : int, optional
            The starting sample index. If None, starts from the beginning.
        end_sample : int, optional
            The ending sample index. If None, goes to the end.

        Returns
        -------
        timestamps: numpy.ndarray
            The timestamps for the data stream.
        """
        # Set defaults
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self.get_num_samples()

        # Return cached timestamps if available
        if self._times is not None:
            return self._times[start_sample:end_sample]

        # See if native timetstamps are available from the format
        native_timestamps = self.get_native_timestamps()
        if native_timestamps is not None:
            self._times = native_timestamps  # Cache the native timestamps
            return native_timestamps[start_sample:end_sample]

        # Fallback to calculated timestamps from sampling frequency
        sample_indices = np.arange(start_sample, end_sample)
        return sample_indices / self.get_sampling_frequency()

    def sample_indices_to_time(self, sample_indices: Union[FloatType, np.ndarray]) -> Union[FloatType, np.ndarray]:
        """Convert user-inputted sample indices to times with units of seconds.

        Parameters
        ----------
        sample_indices: int or array-like
            The sample indices to be converted to times.

        Returns
        -------
        times: float or array-like
            The corresponding times in seconds.

        Deprecated
        ----------
        This method will be removed in or after January 2026.
        Use get_timestamps() instead.
        """
        warnings.warn(
            "sample_indices_to_time() is deprecated and will be removed in or after January 2026. "
            "Use get_timestamps() instead.",
            FutureWarning,
            stacklevel=2,
        )
        # Convert to numpy array if needed to handle indexing
        sample_indices = np.array(sample_indices)

        # Get all timestamps and index into them
        if sample_indices.ndim == 0:
            # Single index
            start_sample = int(sample_indices)
            end_sample = start_sample + 1
            timestamps = self.get_timestamps(start_sample=start_sample, end_sample=end_sample)
            return timestamps[0]
        else:
            # Multiple indices - get the range covering all indices
            start_sample = int(sample_indices.min())
            end_sample = int(sample_indices.max()) + 1
            timestamps = self.get_timestamps(start_sample=start_sample, end_sample=end_sample)
            # Adjust indices to be relative to the start_sample
            relative_indices = sample_indices - start_sample
            return timestamps[relative_indices]

    def frame_to_time(self, frames: Union[FloatType, np.ndarray]) -> Union[FloatType, np.ndarray]:
        """Convert user-inputted frame indices to times with units of seconds.

        Parameters
        ----------
        frames: int or array-like
            The frame or frames to be converted to times.

        Returns
        -------
        times: float or array-like
            The corresponding times in seconds.

        Deprecated
        ----------
        This method will be removed in or after October 2025.
        Use sample_indices_to_time() instead.
        """
        warnings.warn(
            "frame_to_time() is deprecated and will be removed in or after October 2025. "
            "Use sample_indices_to_time() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.sample_indices_to_time(frames)

    def time_to_sample_indices(self, times: Union[FloatType, ArrayType]) -> Union[FloatType, np.ndarray]:
        """Convert user-inputted times (in seconds) to sample indices.

        Parameters
        ----------
        times: float or array-like
            The times (in seconds) to be converted to sample indices.

        Returns
        -------
        sample_indices: float or array-like
            The corresponding sample indices.
        """
        # Default implementation
        if self._times is None:
            return np.round(times * self.get_sampling_frequency()).astype("int64")
        else:
            return np.searchsorted(self._times, times).astype("int64")

    def time_to_frame(self, times: Union[FloatType, ArrayType]) -> Union[FloatType, np.ndarray]:
        """Convert a user-inputted times (in seconds) to a frame indices.

        Parameters
        ----------
        times: float or array-like
            The times (in seconds) to be converted to frame indices.

        Returns
        -------
        frames: float or array-like
            The corresponding frame indices.

        Deprecated
        ----------
        This method will be removed in or after October 2025.
        Use time_to_sample_indices() instead.
        """
        warnings.warn(
            "time_to_frame() is deprecated and will be removed in or after October 2025. "
            "Use time_to_sample_indices() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.time_to_sample_indices(times)

    def set_times(self, times: ArrayType) -> None:
        """Set the recording times (in seconds) for each frame.

        Parameters
        ----------
        times: array-like
            The times in seconds for each frame.

        Raises
        ------
        ValueError
            If the length of 'times' does not match the number of samples.
        """
        num_samples = self.get_num_samples()
        num_timestamps = len(times)

        if num_timestamps != num_samples:
            raise ValueError(
                f"Mismatch between the number of samples and timestamps: "
                f"{num_samples} samples, but {num_timestamps} timestamps provided. "
                "Ensure the length of 'times' matches the number of samples."
            )

        self._times = np.array(times).astype("float64", copy=False)

    def has_time_vector(self) -> bool:
        """Detect if the ImagingExtractor has a time vector set or not.

        Returns
        -------
        has_times: bool
            True if the ImagingExtractor has a time vector set, otherwise False.
        """
        return self._times is not None

    def copy_times(self, extractor) -> None:
        """Copy times from another extractor.

        Parameters
        ----------
        extractor
            The extractor from which the times will be copied.
        """
        if extractor._times is not None:
            self.set_times(deepcopy(extractor._times))

    def slice_samples(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None):
        """Return a new ImagingExtractor ranging from the start_sample to the end_sample.

        Parameters
        ----------
        start_sample: int, optional
            Start sample index (inclusive).
        end_sample: int, optional
            End sample index (exclusive).

        Returns
        -------
        imaging: SampleSlicedImagingExtractor
            The sliced ImagingExtractor object.
        """
        return SampleSlicedImagingExtractor(parent_imaging=self, start_sample=start_sample, end_sample=end_sample)

    def frame_slice(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None):
        """Return a new ImagingExtractor ranging from the start_frame to the end_frame.

        Parameters
        ----------
        start_frame: int, optional
            Start frame index (inclusive).
        end_frame: int, optional
            End frame index (exclusive).

        Returns
        -------
        imaging: FrameSliceImagingExtractor
            The sliced ImagingExtractor object.

        Deprecated
        ----------
        This method will be removed in or after October 2025.
        Use slice_samples() instead.
        """
        warnings.warn(
            "frame_slice() is deprecated and will be removed in or after October 2025. " "Use slice_samples() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.slice_samples(start_sample=start_frame, end_sample=end_frame)


class SampleSlicedImagingExtractor(ImagingExtractor):
    """Class to get a lazy sample slice.

    Do not use this class directly but use `.slice_samples(...)` on an ImagingExtractor object.
    """

    extractor_name = "SampleSlicedImagingExtractor"

    def __init__(
        self, parent_imaging: ImagingExtractor, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ):
        """Initialize an ImagingExtractor whose samples subset the parent.

        Subset is exclusive on the right bound, that is, the indexes of this ImagingExtractor range over
        [0, ..., end_sample-start_sample-1], which is used to resolve the index mapping in `get_frames(frame_idxs=[...])`.

        Parameters
        ----------
        parent_imaging : ImagingExtractor
            The ImagingExtractor object to subset the samples of.
        start_sample : int, optional
            The left bound of the samples to subset.
            The default is the start sample of the parent.
        end_sample : int, optional
            The right bound of the samples, exclusively, to subset.
            The default is end sample of the parent.
        """
        self._parent_imaging = parent_imaging
        self._start_frame = start_sample
        self._end_frame = end_sample
        self._num_samples = self._end_frame - self._start_frame

        parent_size = self._parent_imaging.get_num_samples()
        if start_sample is None:
            start_sample = 0
        else:
            assert 0 <= start_sample < parent_size
        if end_sample is None:
            end_sample = parent_size
        else:
            assert 0 < end_sample <= parent_size
        assert end_sample > start_sample, "'start_sample' must be smaller than 'end_sample'!"

        super().__init__()
        if getattr(self._parent_imaging, "_times") is not None:
            self._times = self._parent_imaging._times[start_sample:end_sample]

        # Inherit volumetric properties from parent
        self.is_volumetric = self._parent_imaging.is_volumetric

    def get_samples(self, sample_indices: ArrayType) -> np.ndarray:
        assert max(sample_indices) < self._num_samples, "'sample_indices' range beyond number of available samples!"
        mapped_sample_indices = np.array(sample_indices) + self._start_frame
        return self._parent_imaging.get_samples(sample_indices=mapped_sample_indices)

    def get_frames(self, frame_idxs: ArrayType, channel: Optional[int] = 0) -> np.ndarray:
        warnings.warn(
            "get_frames() is deprecated and will be removed on or after January 2026. " "Use get_samples() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if channel != 0:
            warnings.warn(
                "The 'channel' parameter in get_frames() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.get_samples(sample_indices=frame_idxs)

    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
        assert start_sample is None or start_sample >= 0, (
            f"'start_sample' must be greater than or equal to zero! Received '{start_sample}'.\n"
            "Negative slicing semantics are not supported."
        )
        start_sample_shifted = (start_sample or 0) + self._start_frame
        end_sample_shifted = end_sample
        if end_sample is not None:
            end_sample_shifted = end_sample + self._start_frame
        return self._parent_imaging.get_series(start_sample=start_sample_shifted, end_sample=end_sample_shifted)

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: Optional[int] = 0
    ) -> np.ndarray:
        warnings.warn(
            "get_video() is deprecated and will be removed in or after September 2025. " "Use get_series() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if channel != 0:
            warnings.warn(
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
        return self._parent_imaging.get_image_shape()

    def get_image_size(self) -> Tuple[int, int]:
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        return tuple(self._parent_imaging.get_image_size())

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_num_frames(self) -> int:
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

    def get_sampling_frequency(self) -> float:
        return self._parent_imaging.get_sampling_frequency()

    def get_channel_names(self) -> list:
        return self._parent_imaging.get_channel_names()

    def get_num_channels(self) -> int:
        """Get the total number of active channels in the recording.

        Returns
        -------
        num_channels: int
            Integer count of number of channels.

        Deprecated
        ----------
        This method will be removed in or after August 2025.
        """
        warnings.warn(
            "get_num_channels() is deprecated and will be removed in or after August 2025.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._parent_imaging.get_num_channels()

    def get_num_planes(self) -> int:
        """Get the number of depth planes.

        Returns
        -------
        num_planes: int
            The number of depth planes.

        Raises
        ------
        NotImplementedError
            If the parent extractor is not volumetric.
        """
        if not self.is_volumetric:
            raise NotImplementedError(
                "This extractor is not volumetric. "
                "The get_num_planes method is only available for volumetric extractors."
            )
        return self._parent_imaging.get_num_planes()

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # Get the full original timestamps from parent, but return only our slice range
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self.get_num_samples()

        # Map relative indices to absolute indices in the parent
        actual_start = self._start_sample + start_sample
        actual_end = self._start_sample + end_sample

        # Get timestamps from parent for our specific range
        return self._parent_imaging.get_native_timestamps(start_sample=actual_start, end_sample=actual_end)


class FrameSliceImagingExtractor(SampleSlicedImagingExtractor):
    """Class to get a lazy frame slice.

    Do not use this class directly but use `.frame_slice(...)` on an ImagingExtractor object.

    Deprecated
    ----------
    This class will be removed in or after October 2025.
    Use SampleSlicedImagingExtractor instead.
    """

    extractor_name = "FrameSliceImagingExtractor"

    def __init__(
        self, parent_imaging: ImagingExtractor, start_frame: Optional[int] = None, end_frame: Optional[int] = None
    ):
        """Initialize an ImagingExtractor whose frames subset the parent.

        Subset is exclusive on the right bound, that is, the indexes of this ImagingExtractor range over
        [0, ..., end_frame-start_frame-1], which is used to resolve the index mapping in `get_frames(frame_idxs=[...])`.

        Parameters
        ----------
        parent_imaging : ImagingExtractor
            The ImagingExtractor object to sebset the frames of.
        start_frame : int, optional
            The left bound of the frames to subset.
            The default is the start frame of the parent.
        end_frame : int, optional
            The right bound of the frames, exlcusively, to subset.
            The default is end frame of the parent.

        Deprecated
        ----------
        This class will be removed in or after October 2025.
        Use SampleSlicedImagingExtractor instead.
        """
        warnings.warn(
            "FrameSliceImagingExtractor is deprecated and will be removed in or after October 2025. "
            "Use SampleSlicedImagingExtractor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(parent_imaging=parent_imaging, start_sample=start_frame, end_sample=end_frame)

    def get_frames(self, frame_idxs: ArrayType, channel: Optional[int] = 0) -> np.ndarray:
        warnings.warn(
            "get_frames() is deprecated and will be removed on or after January 2026. " "Use get_samples() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if channel != 0:
            warnings.warn(
                "The 'channel' parameter in get_frames() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.get_samples(sample_idxs=frame_idxs)

    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
        assert start_sample is None or start_sample >= 0, (
            f"'start_sample' must be greater than or equal to zero! Received '{start_sample}'.\n"
            "Negative slicing semantics are not supported."
        )
        start_sample_shifted = (start_sample or 0) + self._start_frame
        end_sample_shifted = end_sample
        if end_sample is not None:
            end_sample_shifted = end_sample + self._start_frame
        return self._parent_imaging.get_series(start_sample=start_sample_shifted, end_sample=end_sample_shifted)

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: Optional[int] = 0
    ) -> np.ndarray:
        warnings.warn(
            "get_video() is deprecated and will be removed in or after September 2025. " "Use get_series() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if channel != 0:
            warnings.warn(
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
        return self._parent_imaging.get_image_shape()

    def get_image_size(self) -> Tuple[int, int]:
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        return tuple(self._parent_imaging.get_image_size())

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_num_frames(self) -> int:
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

    def get_sampling_frequency(self) -> float:
        return self._parent_imaging.get_sampling_frequency()

    def get_channel_names(self) -> list:
        return self._parent_imaging.get_channel_names()

    def get_num_channels(self) -> int:
        """Get the total number of active channels in the recording.

        Returns
        -------
        num_channels: int
            Integer count of number of channels.

        Deprecated
        ----------
        This method will be removed in or after August 2025.
        """
        warnings.warn(
            "get_num_channels() is deprecated and will be removed in or after August 2025.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._parent_imaging.get_num_channels()

    def get_num_planes(self) -> int:
        """Get the number of depth planes.

        Returns
        -------
        num_planes: int
            The number of depth planes.

        Raises
        ------
        NotImplementedError
            If the parent extractor is not volumetric.
        """
        if not self.is_volumetric:
            raise NotImplementedError(
                "This extractor is not volumetric. "
                "The get_num_planes method is only available for volumetric extractors."
            )
        return self._parent_imaging.get_num_planes()
