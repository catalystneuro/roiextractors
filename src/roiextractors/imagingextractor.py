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

import numpy as np

from .extraction_tools import ArrayType, PathType, DtypeType, FloatType


class ImagingExtractor(ABC):
    """Abstract class that contains all the meta-data and input data from the imaging data."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the ImagingExtractor object."""
        self._args = args
        self._kwargs = kwargs
        self._times = None

    @abstractmethod
    def get_image_size(self) -> Tuple[int, int]:
        """Get the size of the video (num_rows, num_columns).

        Returns
        -------
        image_size: tuple
            Size of the video (num_rows, num_columns).
        """
        pass

    @abstractmethod
    def get_num_frames(self) -> int:
        """Get the number of frames in the video.

        Returns
        -------
        num_frames: int
            Number of frames in the video.
        """
        pass

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
    def get_dtype(self) -> DtypeType:
        """Get the data type of the video.

        Returns
        -------
        dtype: dtype
            Data type of the video.
        """
        pass

    @abstractmethod
    def get_video(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> np.ndarray:
        """Get the video frames.

        Parameters
        ----------
        start_frame: int, optional
            Start frame index (inclusive).
        end_frame: int, optional
            End frame index (exclusive).

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
        pass

    def get_frames(self, frame_idxs: ArrayType) -> np.ndarray:
        """Get specific video frames from indices (not necessarily continuous).

        Parameters
        ----------
        frame_idxs: array-like
            Indices of frames to return.

        Returns
        -------
        frames: numpy.ndarray
            The video frames.
        """
        start_frame = min(frame_idxs)
        end_frame = max(frame_idxs) + 1
        assert start_frame >= 0, f"All 'frame_idxs' must be greater than or equal to zero but received {start_frame}."
        assert (
            end_frame <= self.get_num_frames()
        ), f"All 'frame_idxs' must be less than the number of frames ({self.get_num_frames()}) but received {end_frame}."
        relative_indices = np.array(frame_idxs) - start_frame
        return self.get_video(start_frame=start_frame, end_frame=end_frame)[relative_indices, ...]

    def frame_to_time(self, frames: ArrayType) -> Union[FloatType, np.ndarray]:
        """Convert user-inputted frame indices to times with units of seconds.

        Parameters
        ----------
        frames: array-like
            The frame or frames to be converted to times.

        Returns
        -------
        times: float or array-like
            The corresponding times in seconds.
        """
        # Default implementation
        frames = np.asarray(frames)
        if self._times is None:
            return frames / self.get_sampling_frequency()
        else:
            return self._times[frames]

    def time_to_frame(self, times: ArrayType) -> Union[FloatType, np.ndarray]:
        """Convert a user-inputted times (in seconds) to a frame indices.

        Parameters
        ----------
        times: array-like
            The times (in seconds) to be converted to frame indices.

        Returns
        -------
        frames: float or array-like
            The corresponding frame indices.
        """
        # Default implementation
        times = np.asarray(times)
        if self._times is None:
            return np.round(times * self.get_sampling_frequency()).astype("int64")
        else:
            return np.searchsorted(self._times, times).astype("int64")

    def set_times(self, times: ArrayType) -> None:
        """Set the recording times (in seconds) for each frame.

        Parameters
        ----------
        times: array-like
            The times in seconds for each frame
        """
        assert len(times) == self.get_num_frames(), "'times' should have the same length of the number of frames!"
        self._times = np.array(times).astype("float64")

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

    def __eq__(self, imaging_extractor2):
        image_size_equal = self.get_image_size() == imaging_extractor2.get_image_size()
        num_frames_equal = self.get_num_frames() == imaging_extractor2.get_num_frames()
        sampling_frequency_equal = np.isclose(
            self.get_sampling_frequency(), imaging_extractor2.get_sampling_frequency()
        )
        dtype_equal = self.get_dtype() == imaging_extractor2.get_dtype()
        video_equal = np.array_equal(self.get_video(), imaging_extractor2.get_video())
        times_equal = np.allclose(
            self.frame_to_time(np.arange(self.get_num_frames())),
            imaging_extractor2.frame_to_time(np.arange(imaging_extractor2.get_num_frames())),
        )
        imaging_extractors_equal = all(
            [
                image_size_equal,
                num_frames_equal,
                sampling_frequency_equal,
                dtype_equal,
                video_equal,
                times_equal,
            ]
        )

        return imaging_extractors_equal

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
        num_frames = self.get_num_frames()
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else num_frames
        assert 0 <= start_frame < num_frames, f"'start_frame' must be in [0, {num_frames}) but got {start_frame}"
        assert 0 < end_frame <= num_frames, f"'end_frame' must be in (0, {num_frames}] but got {end_frame}"
        assert (
            start_frame <= end_frame
        ), f"'start_frame' ({start_frame}) must be less than or equal to 'end_frame' ({end_frame})"

        return FrameSliceImagingExtractor(parent_imaging=self, start_frame=start_frame, end_frame=end_frame)


class FrameSliceImagingExtractor(ImagingExtractor):
    """Class to get a lazy frame slice.

    Do not use this class directly but use `.frame_slice(...)` on an ImagingExtractor object.
    """

    extractor_name = "FrameSliceImagingExtractor"
    installed = True
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
        self._num_frames = self._end_frame - self._start_frame

        super().__init__()
        if getattr(self._parent_imaging, "_times") is not None:
            self._times = self._parent_imaging._times[start_frame:end_frame]

    def get_frames(self, frame_idxs: ArrayType) -> np.ndarray:
        assert max(frame_idxs) < self._num_frames, "'frame_idxs' range beyond number of available frames!"
        assert min(frame_idxs) >= 0, "'frame_idxs' must be greater than or equal to zero!"
        mapped_frame_idxs = np.array(frame_idxs) + self._start_frame
        return self._parent_imaging.get_frames(frame_idxs=mapped_frame_idxs)

    def get_video(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> np.ndarray:
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else self._num_frames
        assert 0 <= start_frame < end_frame, f"'start_frame' must be in [0, end_frame) but got {start_frame}"
        assert (
            start_frame < end_frame <= self._num_frames
        ), f"'end_frame' must be in (start_frame, {self._num_frames}] but got {end_frame}"
        assert isinstance(start_frame, int), "'start_frame' must be an integer"
        assert isinstance(end_frame, int), "'end_frame' must be an integer"

        start_frame_shifted = start_frame + self._start_frame
        end_frame_shifted = end_frame + self._start_frame
        return self._parent_imaging.get_video(start_frame=start_frame_shifted, end_frame=end_frame_shifted)

    def get_image_size(self) -> Tuple[int, int]:
        return tuple(self._parent_imaging.get_image_size())

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return self._parent_imaging.get_sampling_frequency()

    def get_dtype(self) -> DtypeType:
        return self._parent_imaging.get_dtype()
