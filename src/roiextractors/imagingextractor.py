"""Base class definitions for all ImagingExtractors."""
from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple
import numpy as np
from copy import deepcopy

from spikeextractors.baseextractor import BaseExtractor

from .extraction_tools import (
    ArrayType,
    PathType,
    DtypeType,
    FloatType,
    check_get_videos_args,
)


class ImagingExtractor(ABC, BaseExtractor):
    """Abstract class that contains all the meta-data and input data from the imaging data."""

    def __init__(self) -> None:
        BaseExtractor.__init__(self)
        assert self.installed, self.installation_mesg
        self._memmapped = False

    @abstractmethod
    def get_image_size(self) -> Tuple[int, int]:
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
        return self.get_frames(frame_idxs=[0], channel=0).dtype

    @abstractmethod
    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        pass

    def get_frames(self, frame_idxs: ArrayType, channel: Optional[int] = 0) -> np.ndarray:

        return self.get_video()[frame_idxs, ..., channel]  # To-do simplify for contigious frame_idxs case

    def frame_to_time(self, frames: Union[FloatType, np.ndarray]) -> Union[FloatType, np.ndarray]:
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

    def time_to_frame(self, times: Union[FloatType, ArrayType]) -> Union[FloatType, np.ndarray]:
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

    def set_times(self, times: ArrayType) -> None:
        """This function sets the recording times (in seconds) for each frame

        Parameters
        ----------
        times: array-like
            The times in seconds for each frame
        """
        assert len(times) == self.get_num_frames(), "'times' should have the same length of the number of frames!"
        self._times = np.array(times).astype("float64")

    def copy_times(self, extractor: BaseExtractor) -> None:
        """This function copies times from another extractor.

        Parameters
        ----------
        extractor: BaseExtractor
            The extractor from which the epochs will be copied
        """
        if extractor._times is not None:
            self.set_times(deepcopy(extractor._times))

    def frame_slice(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None):
        """Return a new ImagingExtractor ranging from the start_frame to the end_frame."""
        return FrameSliceImagingExtractor(parent_imaging=self, start_frame=start_frame, end_frame=end_frame)

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


class FrameSliceImagingExtractor(ImagingExtractor):
    """
    Class to get a lazy frame slice.

    Do not use this class directly but use `.frame_slice(...)` on an ImagingExtractor object.
    """

    extractor_name = "FrameSliceImagingExtractor"
    installed = True
    is_writable = True
    installation_mesg = ""

    def __init__(
        self, parent_imaging: ImagingExtractor, start_frame: Optional[int] = None, end_frame: Optional[int] = None
    ):
        """
        Initialize an ImagingExtractor whose frames subset the parent.

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

        parent_size = self._parent_imaging.get_num_frames()
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
        assert max(frame_idxs) < self._num_frames, "'frame_idxs' range beyond number of available frames!"
        mapped_frame_idxs = np.array(frame_idxs) + self._start_frame
        return self._parent_imaging.get_frames(frame_idxs=mapped_frame_idxs, channel=channel)

    def get_video(self, start_frame=None, end_frame=None, channel: Optional[int] = 0) -> np.ndarray:
        assert max(end_frame) < self._num_frames, "'end_frame' range beyond number of available frames!"
        start_frame_shifted = start_frame + self._start_frame
        return self._parent_imaging.get_video(start_frame=start_frame_shifted, end_frame=end_frame, channel=channel)

    def get_image_size(self) -> Tuple[int, int]:
        return tuple(self._parent_imaging.get_image_size())

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return self._parent_imaging.get_sampling_frequency()

    def get_channel_names(self) -> list:
        return self._parent_imaging.get_channel_names()

    def get_num_channels(self) -> int:
        return self._parent_imaging.get_num_channels()
