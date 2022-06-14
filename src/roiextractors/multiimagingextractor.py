from abc import ABC
from array import ArrayType
from typing import Tuple, List, Iterable

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

    def __init__(self, imaging_extractors: List[ImagingExtractor]):
        """
        Parameters
        ----------
        imaging_extractors: list of ImagingExtractor
            list of imaging extractor objects
        """
        super().__init__()
        assert isinstance(imaging_extractors, list), "Enter a list of ImagingExtractor objects as argument"
        assert all(isinstance(IX, ImagingExtractor) for IX in imaging_extractors)
        self._imaging_extractors = imaging_extractors

        # Checks that properties are consistent between extractors
        self._check_consistency_between_imaging_extractors()

        # Set properties based off the initial extractor
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

        self._num_frames = int(num_frames)

    def _check_consistency_between_imaging_extractors(self):
        properties_to_check = dict(
            get_sampling_frequency="The sampling frequency",
            get_image_size="The size of a frame",
            get_num_channels="The number of channels",
            get_channel_names="The name of the channels",
        )
        for method, property_message in properties_to_check.items():
            values = [getattr(extractor, method, None)() for extractor in self._imaging_extractors]
            unique_values = set(tuple(v) if isinstance(v, Iterable) else v for v in values)
            assert (
                len(unique_values) == 1
            ), f"{property_message} is not consistent over the files (found {unique_values})."

    def get_frames(self, frame_idxs: ArrayType, channel: int = 0) -> NumpyArray:
        assert max(frame_idxs) < self._num_frames, "'frame_idxs' range beyond number of available frames!"
        extractor_indices = np.searchsorted(self._end_frames, frame_idxs, side="right")
        # Match frame_idxs to imaging extractors
        extractors_dict = {}
        for extractor_index, frame_index in zip(extractor_indices, frame_idxs):
            extractors_dict.setdefault(extractor_index, []).append(frame_index)

        frames_to_concatenate = []
        # Extract frames for each extractor and concatenate
        for extractor_index, frame_indices in extractors_dict.items():
            frames_for_each_extractor = self._get_frames_from_an_imaging_extractor(
                extractor_index=extractor_index,
                frame_idxs=frame_indices,
            )
            if len(frame_indices) == 1:
                frames_for_each_extractor = frames_for_each_extractor[np.newaxis, ...]
            frames_to_concatenate.append(frames_for_each_extractor)

        frames = np.concatenate(frames_to_concatenate, axis=0).squeeze()
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
