"""Primary class for subsetting the frames of an ImagingExtractor."""
from typing import Optional

import numpy as np

from .imagingextractor import ImagingExtractor
from .extraction_tools import ArrayType, NumpyArray


class FrameSliceImaging(ImagingExtractor):
    """
    Class to get a lazy frame slice.

    Do not use this class directly but use `recording.frame_slice(...)`
    """

    def __init__(
        self, parent_imaging: ImagingExtractor, start_frame: Optional[int] = None, end_frame: Optional[int] = None
    ):
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
        self._parent_imaging = parent_imaging
        self._start_frame = start_frame
        self._end_frame
        self._num_frames = self.end_frame - self.start_time

    def get_frames(self, frame_idxs: ArrayType, channel: int = 0) -> NumpyArray:
        assert max(frame_idxs) < self._num_frames, "'frame_idxs' range beyond number of available frames!"
        mapped_frame_idxs = np.array(frame_idxs) + self._start_frame
        return self._parent_imaging.get_frames(frame_idxs=mapped_frame_idxs, channel=channel)

    def get_image_size(self) -> ArrayType:
        return self._parent_imaging.get_image_size()

    def get_num_frames(self) -> int:
        self._num_frames

    def get_sampling_frequency(self) -> float:
        self._parent_imaging.get_sampling_frequency()

    def get_channel_names(self) -> list:
        self._parent_imaging.get_channel_names()

    def get_num_channels(self) -> int:
        self._parent_imaging.get_num_channels()
