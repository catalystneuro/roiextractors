from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from copy import deepcopy

from spikeextractors.baseextractor import BaseExtractor

from .extraction_tools import (
    ArrayType,
    PathType,
    NumpyArray,
    DtypeType,
    IntType,
    FloatType,
    check_get_videos_args,
)


class ImagingExtractor(ABC, BaseExtractor):
    """An abstract class that contains all the meta-data and input data from
    the imaging data.
    """

    def __init__(self):
        BaseExtractor.__init__(self)
        assert self.installed, self.installation_mesg
        self._memmapped = False

    @abstractmethod
    def get_frames(self, frame_idxs: ArrayType, channel: int = 0) -> NumpyArray:
        pass

    @abstractmethod
    def get_image_size(self) -> ArrayType:
        pass

    @abstractmethod
    def get_num_frames(self) -> int:
        pass

    @abstractmethod
    def get_sampling_frequency(self) -> float:
        pass

    @abstractmethod
    def get_channel_names(self) -> list:
        """List of  channels in the recoding.

        Returns
        -------
        channel_names: list
            List of strings of channel names
        """
        pass

    @abstractmethod
    def get_num_channels(self) -> int:
        """Total number of active channels in the recording

        Returns
        -------
        no_of_channels: int
            integer count of number of channels
        """
        pass

    def get_dtype(self) -> DtypeType:
        return self.get_frames(0, 0).dtype

    @check_get_videos_args
    def get_video(
        self, start_frame: int = None, end_frame: int = None, channel: int = 0
    ) -> NumpyArray:
        return self.get_frames(range(start_frame, end_frame), channel)

    def frame_to_time(
        self, frames: Union[FloatType, NumpyArray]
    ) -> Union[FloatType, NumpyArray]:
        """This function converts user-inputted frame indexes to times with units of seconds.

        Parameters
        ----------
        frames: int or array-like
            The frame or frames to be converted to times

        Returns
        -------
        times: float or array-like
            The corresponding times in seconds
        """
        # Default implementation
        if self._times is None:
            return np.round(frames / self.get_sampling_frequency(), 6)
        else:
            return self._times[frames]

    def time_to_frame(
        self, times: Union[FloatType, NumpyArray]
    ) -> Union[FloatType, NumpyArray]:
        """This function converts a user-inputted times (in seconds) to a frame indexes.

        Parameters
        -------
        times: float or array-like
            The times (in seconds) to be converted to frame indexes

        Returns
        -------
        frames: float or array-like
            The corresponding frame indexes
        """
        # Default implementation
        if self._times is None:
            return np.round(times * self.get_sampling_frequency()).astype("int64")
        else:
            return np.searchsorted(self._times, times).astype("int64")

    def set_times(self, times: NumpyArray):
        """This function sets the recording times (in seconds) for each frame

        Parameters
        ----------
        times: array-like
            The times in seconds for each frame
        """
        assert (
            len(times) == self.get_num_frames()
        ), "'times' should have the same length of the number of frames"
        self._times = times.astype("float64")

    def copy_times(self, extractor: BaseExtractor):
        """This function copies times from another extractor.

        Parameters
        ----------
        extractor: BaseExtractor
            The extractor from which the epochs will be copied
        """
        if extractor._times is not None:
            self.set_times(deepcopy(extractor._times))

    @staticmethod
    def write_imaging(imaging, save_path: PathType, overwrite: bool = False):
        """
        Static method to write imaging.

        Parameters
        ----------
        imaging: ImagingExtractor object
            The EXTRACT segmentation object from which an EXTRACT native format
            file has to be generated.
        save_path: str
            path to save the native format.
        overwrite: bool
            If True and save_path is existing, it is overwritten
        """
        raise NotImplementedError
