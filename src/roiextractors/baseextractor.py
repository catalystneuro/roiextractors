from abc import ABC, abstractmethod
from typing import Union, Tuple
from copy import deepcopy
import numpy as np
from .extraction_tools import ArrayType, FloatType


class BaseExtractor(ABC):

    def __init__(self):
        self._times = None

    @abstractmethod
    def get_image_size(self) -> Tuple[int, int]:
        """Get the size of each image in the recording (num_rows, num_columns).

        Returns
        -------
        image_size: tuple
            Size of each image (num_rows, num_columns).
        """
        pass

    @abstractmethod
    def get_num_frames(self) -> int:
        """Get the number of frames in the recording.

        Returns
        -------
        num_frames: int
            Number of frames in the recording.
        """
        pass

    @abstractmethod
    def get_sampling_frequency(self) -> float:
        """Get the sampling frequency of the recording in Hz.

        Returns
        -------
        sampling_frequency: float
            Sampling frequency of the recording in Hz.
        """
        pass

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
