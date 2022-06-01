from abc import ABC
from array import ArrayType
from typing import Tuple

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
        assert isinstance(
            imaging_extractors, list
        ), "Enter a list of imaging extractor objects as argument"
        assert all(isinstance(IX, ImagingExtractor) for IX in imaging_extractors)
        self._imaging_extractors = imaging_extractors

        # Num channels and sampling frequency based off the initial extractor
        self._first_imaging_extractor = self._imaging_extractors[0]
        self._num_channels = self._first_imaging_extractor.get_num_channels()
        self._channel_names = self._first_imaging_extractor.get_channel_names()
        self._sampling_frequency = (
            self._first_imaging_extractor.get_sampling_frequency()
        )
        self._image_size = self._first_imaging_extractor.get_image_size()

        self._start_frames, self._end_frames = [], []
        num_frames = 0.0
        for extractor_index, imaging_extractor in enumerate(self._imaging_extractors):
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
        extractor_index = self._get_imaging_extractor_index(frame_idxs[0])
        rel_frame_index = frame_idxs[0] - int(self._start_frames[extractor_index])
        imaging_extractor = self._imaging_extractors[extractor_index]

        return imaging_extractor.get_frames(frame_idxs=[rel_frame_index])

    def _get_imaging_extractor_index(self, frame_index: int):
        for ind, (start, end) in enumerate(zip(self._start_frames, self._end_frames)):
            if start <= frame_index < end:
                return ind

        return len(self._start_frames) - 1

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
