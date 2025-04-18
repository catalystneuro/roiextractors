"""Base class definitions for all ImagingExtractors.

Classes
-------
ImagingExtractor
    Abstract class that contains all the meta-data and input data from the imaging data.
FrameSliceImagingExtractor
    Class to get a lazy frame slice.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple
from copy import deepcopy
import warnings

import numpy as np

from .extraction_tools import ArrayType, PathType, DtypeType, FloatType
from math import prod


class ImagingExtractor(ABC):
    """Abstract class that contains all the meta-data and input data from the imaging data."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the ImagingExtractor object."""
        self._args = args
        self._kwargs = kwargs
        self._times = None
        self.name = self.__class__.__name__

    def _repr_text(self):
        """Generate text representation of the ImagingExtractor object."""
        num_samples = self.get_num_samples()
        image_size = self.get_image_size()
        dtype = self.get_dtype()
        sf_hz = self.get_sampling_frequency()

        # Format sampling frequency
        if not sf_hz.is_integer():
            sampling_frequency_repr = f"{sf_hz:f} Hz"
        else:
            sampling_frequency_repr = f"{sf_hz:0.1f}Hz"

        # Calculate duration
        duration = num_samples / sf_hz
        duration_str = self._convert_seconds_to_str(duration)

        # Check if this is a volumetric extractor
        is_volumetric_extractor = hasattr(self, "get_num_planes") and callable(getattr(self, "get_num_planes"))

        # Calculate memory size using product of all dimensions in image_size
        memory_size = num_samples * prod(image_size) * dtype.itemsize
        memory_str = self._convert_bytes_to_str(memory_size)

        # Format shape string based on whether it's volumetric or not
        if is_volumetric_extractor:
            num_planes = self.get_num_planes()
            shape_str = (
                f"[{num_samples:,} frames × {image_size[0]} pixels × {image_size[1]} pixels × {num_planes} planes]"
            )
        else:
            shape_str = f"[{num_samples:,} frames × {image_size[0]} pixels × {image_size[1]} pixels]"

        return (
            f"{self.name} {shape_str}\n"
            f"  Sampling rate: {sampling_frequency_repr}\n"
            f"  Duration: {duration_str}\n"
            f"  Memory: {memory_str} ({dtype} dtype)"
        )

    def __repr__(self):
        return self._repr_text()

    def _convert_seconds_to_str(self, seconds):
        """Convert seconds to a human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}min"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def _convert_bytes_to_str(self, size_in_bytes):
        """
        Convert bytes to a human-readable string.

        Convert bytes to a human-readable string using IEC binary prefixes (KiB, MiB, GiB).
        Note that RAM memory is typically measured in IEC binary prefixes  while disk storage is typically
        measured in SI binary prefixes.
        """
        if size_in_bytes < 1024:
            return f"{size_in_bytes}B"
        elif size_in_bytes < 1024 * 1024:
            size_kb = size_in_bytes / 1024
            return f"{size_kb:.1f}KiB"
        elif size_in_bytes < 1024 * 1024 * 1024:
            size_mb = size_in_bytes / (1024 * 1024)
            return f"{size_mb:.1f}MiB"
        else:
            size_gb = size_in_bytes / (1024 * 1024 * 1024)
            return f"{size_gb:.1f}GiB"

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
        return self.get_frames(frame_idxs=[0], channel=0).dtype

    @abstractmethod
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

        Notes
        -----
        Importantly, we follow the convention that the dimensions of the array are returned in their matrix order,
        More specifically:
        (time, height, width)

        Which is equivalent to:
        (samples, rows, columns)

        Note that this does not match the cartesian convention:
        (t, x, y)

        Where x is the columns width or and y is the rows or height.
        """
        if channel != 0:
            warnings.warn(
                "The 'channel' parameter in get_video() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
        pass

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
        """
        if channel != 0:
            warnings.warn(
                "The 'channel' parameter in get_frames() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
        assert max(frame_idxs) <= self.get_num_samples(), "'frame_idxs' exceed number of samples"
        if np.all(np.diff(frame_idxs) == 0):
            return self.get_video(start_frame=frame_idxs[0], end_frame=frame_idxs[-1])
        relative_indices = np.array(frame_idxs) - frame_idxs[0]
        return self.get_video(start_frame=frame_idxs[0], end_frame=frame_idxs[-1] + 1)[relative_indices, ..., channel]

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
        """
        # Default implementation
        if self._times is None:
            return frames / self.get_sampling_frequency()
        else:
            return self._times[frames]

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
        """
        # Default implementation
        if self._times is None:
            return np.round(times * self.get_sampling_frequency()).astype("int64")
        else:
            return np.searchsorted(self._times, times).astype("int64")

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
        """
        return FrameSliceImagingExtractor(parent_imaging=self, start_frame=start_frame, end_frame=end_frame)


class FrameSliceImagingExtractor(ImagingExtractor):
    """Class to get a lazy frame slice.

    Do not use this class directly but use `.frame_slice(...)` on an ImagingExtractor object.
    """

    extractor_name = "FrameSliceImagingExtractor"
    is_writable = True
    installation_mesg = ""

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

        """
        self._parent_imaging = parent_imaging
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._num_samples = self._end_frame - self._start_frame

        parent_size = self._parent_imaging.get_num_samples()
        if start_frame is None:
            start_frame = 0
        else:
            assert 0 <= start_frame < parent_size
        if end_frame is None:
            end_frame = parent_size
        else:
            assert 0 < end_frame <= parent_size
        assert end_frame > start_frame, "'start_frame' must be smaller than 'end_frame'!"

        super().__init__()
        if getattr(self._parent_imaging, "_times") is not None:
            self._times = self._parent_imaging._times[start_frame:end_frame]

    def get_frames(self, frame_idxs: ArrayType, channel: Optional[int] = 0) -> np.ndarray:
        assert max(frame_idxs) < self._num_samples, "'frame_idxs' range beyond number of available frames!"
        mapped_frame_idxs = np.array(frame_idxs) + self._start_frame
        return self._parent_imaging.get_frames(frame_idxs=mapped_frame_idxs, channel=channel)

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: Optional[int] = 0
    ) -> np.ndarray:
        assert start_frame >= 0, (
            f"'start_frame' must be greater than or equal to zero! Received '{start_frame}'.\n"
            "Negative slicing semantics are not supported."
        )
        start_frame_shifted = start_frame + self._start_frame
        return self._parent_imaging.get_video(start_frame=start_frame_shifted, end_frame=end_frame, channel=channel)

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
