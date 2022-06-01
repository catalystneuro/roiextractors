from abc import ABC
from array import ArrayType
from typing import Tuple

import numpy as np

from .extraction_tools import NumpyArray
from .imagingextractor import ImagingExtractor


class MultiImagingExtractor(ImagingExtractor, ABC):
    """
    This class is used to combine multiple ImagingExtractor objects by frame.
    """

    extractor_name = "MultiImagingExtractor"
    installed = True
    installation_mesg = ""

    def __init__(self, imaging_extractors: list):
        """
        Parameters
        ----------
        imaging_extractors: list of ImagingExtractor
            list of imaging extractor objects
        """
        super().__init__()
        assert isinstance(imaging_extractors, list), "Enter a list of imaging extractor objects as argument"
        assert all(isinstance(IX, ImagingExtractor) for IX in imaging_extractors)
        self._imaging_extractors = imaging_extractors

        # Num channels and sampling frequency based off the initial extractor
        self._first_imaging_extractor = self._imaging_extractors[0]
        self._num_channels = self._first_imaging_extractor.get_num_channels()
        self._channel_names = self._first_imaging_extractor.get_channel_names()
        self._sampling_frequency = self._first_imaging_extractor.get_sampling_frequency()
        self._image_size = self._first_imaging_extractor.get_image_size()

        self._start_frames, self._end_frames = [], []
        num_frames = 0.0
        for imaging_extractor in self._imaging_extractors:
            self._start_frames.append(num_frames)
            num_frames = num_frames + imaging_extractor.get_num_frames()
            self._end_frames.append(num_frames)

            # Check consistency between extractors
            sampling_frequency = imaging_extractor.get_sampling_frequency()
            assert (
                self._sampling_frequency == sampling_frequency
            ), f"Inconsistent sampling frequency ({sampling_frequency}) for {imaging_extractor.file_path}"
            image_size = imaging_extractor.get_image_size()
            assert (
                self._image_size == image_size
            ), f"Inconsistent image size ({image_size}) for {imaging_extractor.file_path}"

        self._num_frames = int(num_frames)

    def get_frames(self, frame_idxs: ArrayType, channel: int = 0) -> NumpyArray:
        extractor_indices = np.searchsorted(self._end_frames, frame_idxs, side="right")

        frames_to_concatenate = []
        # Extract frames for each extractor and concatenate
        for extractor_index in extractor_indices:
            frames_for_each_extractor = self._get_frames_from_an_imaging_extractor(
                extractor_index=extractor_index,
                frame_idxs=frame_idxs,
            )
            frames_to_concatenate.append(frames_for_each_extractor[np.newaxis, ...])

        frames = np.concatenate(frames_to_concatenate, axis=0)
        return frames

    def _get_frames_from_an_imaging_extractor(self, extractor_index: int, frame_idxs: ArrayType):
        relative_frame_indices = (np.array(frame_idxs) - self._start_frames[extractor_index]).astype(int)
        imaging_extractor = self._imaging_extractors[extractor_index]

        frames = imaging_extractor.get_frames(frame_idxs=relative_frame_indices)
        return frames

    def get_image_size(self) -> Tuple:
        return self._image_size

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_channel_names(self) -> list:
        return self._channel_names

    def get_num_channels(self) -> int:
        return self._num_channels
