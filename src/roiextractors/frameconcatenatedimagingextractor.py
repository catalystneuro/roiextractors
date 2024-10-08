"""Defines the FrameConcatenatedImagingExtractor class.

Classes
-------
FrameConcatenatedImagingExtractor
    This class is used to combine multiple ImagingExtractor objects by frames.
"""

from collections import defaultdict
from typing import Tuple, List, Iterable, Optional

import numpy as np

from .tools.typing import ArrayType
from .imagingextractor import ImagingExtractor


class FrameConcatenatedImagingExtractor(ImagingExtractor):
    """Class to combine multiple ImagingExtractor objects by frames."""

    extractor_name = "FrameConcatenatedImagingExtractor"
    installed = True
    installation_mesg = ""

    def __init__(self, imaging_extractors: List[ImagingExtractor]):
        """Initialize a FrameConcatenatedImagingExtractor object from a list of ImagingExtractors.

        Parameters
        ----------
        imaging_extractors: list of ImagingExtractor
            list of imaging extractor objects
        """
        super().__init__()
        assert isinstance(imaging_extractors, list), "Enter a list of ImagingExtractor objects as argument"
        assert all(isinstance(imaging_extractor, ImagingExtractor) for imaging_extractor in imaging_extractors)
        self._check_consistency_between_imaging_extractors(imaging_extractors=imaging_extractors)
        self._imaging_extractors = imaging_extractors

        self._start_frames, self._end_frames = [], []
        num_frames = 0
        for imaging_extractor in self._imaging_extractors:
            self._start_frames.append(num_frames)
            num_frames = num_frames + imaging_extractor.get_num_frames()
            self._end_frames.append(num_frames)
        self._start_frames = np.array(self._start_frames)
        self._end_frames = np.array(self._end_frames)
        self._num_frames = num_frames

        if any((getattr(imaging_extractor, "_times") is not None for imaging_extractor in self._imaging_extractors)):
            times = self._get_times()
            self.set_times(times=times)

    @staticmethod
    def _check_consistency_between_imaging_extractors(imaging_extractors: List[ImagingExtractor]):
        """Check that essential properties are consistent between extractors so that they can be combined appropriately.

        Parameters
        ----------
        imaging_extractors: list of ImagingExtractor
            list of imaging extractor objects

        Raises
        ------
        AssertionError
            If any of the properties are not consistent between extractors.

        Notes
        -----
        This method checks the following properties:
            - sampling frequency
            - image size
            - data type
        """
        properties_to_check = dict(
            get_sampling_frequency="The sampling frequency",
            get_image_size="The size of a frame",
            get_dtype="The data type.",
        )
        for method, property_message in properties_to_check.items():
            values = [getattr(extractor, method)() for extractor in imaging_extractors]
            unique_values = set(tuple(v) if isinstance(v, Iterable) else v for v in values)
            assert (
                len(unique_values) == 1
            ), f"{property_message} is not consistent over the files (found {unique_values})."

    def _get_times(self) -> np.ndarray:
        """Get all the times from the imaging extractors and combine them into a single array.

        Returns
        -------
        times: numpy.ndarray
            Array of times.
        """
        frame_indices = np.arange(self._num_frames)
        times = self.frame_to_time(frames=frame_indices)

        for extractor_index, extractor in enumerate(self._imaging_extractors):
            if getattr(extractor, "_times") is not None:
                to_replace = np.arange(self._start_frames[extractor_index], self._end_frames[extractor_index])
                times[to_replace] = extractor._times

        return times

    def get_dtype(self):
        return self._imaging_extractors[0].get_dtype()

    def get_frames(self, frame_idxs: ArrayType) -> np.ndarray:
        self._validate_get_frames_arguments(frame_idxs=frame_idxs)
        extractor_indices = np.searchsorted(self._end_frames, frame_idxs, side="right")
        relative_frame_indices = frame_idxs - self._start_frames[extractor_indices]

        # Match frame_idxs to imaging extractors
        extractor_index_to_relative_frame_indices = defaultdict(list)
        for extractor_index, frame_index in zip(extractor_indices, relative_frame_indices):
            extractor_index_to_relative_frame_indices[extractor_index].append(frame_index)

        frames_to_concatenate = []
        # Extract frames for each extractor and concatenate
        for extractor_index, frame_indices in extractor_index_to_relative_frame_indices.items():
            imaging_extractor = self._imaging_extractors[extractor_index]
            frames = imaging_extractor.get_frames(frame_idxs=frame_indices)
            frames_to_concatenate.append(frames)

        frames = np.concatenate(frames_to_concatenate, axis=0)
        return frames

    def get_video(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> np.ndarray:
        start_frame, end_frame = self._validate_get_video_arguments(start_frame=start_frame, end_frame=end_frame)
        extractors_range = np.searchsorted(self._end_frames, (start_frame, end_frame - 1), side="right")
        extractors_spanned = list(
            range(extractors_range[0], min(extractors_range[-1] + 1, len(self._imaging_extractors)))
        )

        # Early return with simple relative indexing; preserves native return class of that extractor
        if len(extractors_spanned) == 1:
            extractor_index = extractors_spanned[0]
            relative_start = start_frame - self._start_frames[extractor_index]
            relative_stop = end_frame - start_frame + relative_start

            return self._imaging_extractors[extractors_spanned[0]].get_video(
                start_frame=relative_start, end_frame=relative_stop
            )

        video_shape = (end_frame - start_frame, *self._imaging_extractors[0].get_image_size())
        video = np.empty(shape=video_shape, dtype=self.get_dtype())
        current_frame = 0

        # Left endpoint; since more than one extractor is spanned, only care about indexing first start frame
        extractor_index = extractors_spanned[0]
        relative_start = start_frame - self._start_frames[extractor_index]
        relative_span = self._end_frames[extractor_index] - start_frame
        array_frame_slice = slice(current_frame, relative_span)
        imaging_extractor = self._imaging_extractors[extractor_index]
        video[array_frame_slice, ...] = imaging_extractor.get_video(start_frame=relative_start)
        current_frame += relative_span

        # All inner spans can be written knowing only how long each section is
        for extractor_index in extractors_spanned[1:-1]:
            relative_span = self._end_frames[extractor_index] - self._start_frames[extractor_index]
            array_frame_slice = slice(current_frame, current_frame + relative_span)
            imaging_extractor = self._imaging_extractors[extractor_index]
            video[array_frame_slice, ...] = imaging_extractor.get_video()
            current_frame += relative_span

        # Right endpoint; since more than one extractor is spanned, only care about indexing final end frame
        relative_stop = end_frame - self._start_frames[extractors_spanned[-1]]
        array_frame_slice = slice(current_frame, None)
        imaging_extractor = self._imaging_extractors[extractors_spanned[-1]]
        video[array_frame_slice, ...] = imaging_extractor.get_video(end_frame=relative_stop)

        return video

    def get_image_size(self) -> Tuple[int, int]:
        return self._imaging_extractors[0].get_image_size()

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return self._imaging_extractors[0].get_sampling_frequency()

    def get_channel_names(self) -> list:
        return self._imaging_extractors[0].get_channel_names()

    def get_num_channels(self) -> int:
        return self._imaging_extractors[0].get_num_channels()
