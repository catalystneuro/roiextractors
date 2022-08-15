from collections import defaultdict
from typing import Tuple, List, Iterable, Optional

import numpy as np

from .extraction_tools import ArrayType, NumpyArray, check_get_frames_args
from .imagingextractor import ImagingExtractor


class MultiImagingExtractor(ImagingExtractor):
    """
    This class is used to combine multiple ImagingExtractor objects by frames.
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

        self._dtype = self._imaging_extractors[0].get_dtype()
        self._start_frames, self._end_frames = [], []
        num_frames = 0
        for imaging_extractor in self._imaging_extractors:
            self._start_frames.append(num_frames)
            num_frames = num_frames + imaging_extractor.get_num_frames()
            self._end_frames.append(num_frames)
        self._num_frames = num_frames

        if any((getattr(imaging_extractor, "_times") is not None for imaging_extractor in self._imaging_extractors)):
            times = self._get_times()
            self.set_times(times=times)

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

    def _get_times(self):
        frame_indices = np.array([*range(self._start_frames[0], self._end_frames[-1])])
        times = self.frame_to_time(frames=frame_indices)

        for extractor_index, extractor in enumerate(self._imaging_extractors):
            if getattr(extractor, "_times") is not None:
                to_replace = [*range(self._start_frames[extractor_index], self._end_frames[extractor_index])]
                times[to_replace] = extractor._times

        return times

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        if channel != 0:
            raise NotImplementedError(
                f"MultiImagingExtractors for multiple channels have not yet been implemented! (Received '{channel}'."
            )

        start = start_frame if start_frame is not None else 0
        stop = end_frame if end_frame is not None else self.get_num_frames()
        total_range = list(range(start, stop))
        extractors_spanned = np.unique(np.searchsorted(self._end_frames, total_range, side="right"))

        video_shape = (stop - start,) + self._imaging_extractors[0].get_image_size()
        video = np.empty(shape=video_shape, dtype=self._dtype)

        current_frame = 0
        for extractor_index in extractors_spanned:
            relative_start = self._start_frames[extractor_index] - start
            relative_stop = min(self._end_frames[extractor_index], stop) - start
            relative_span = relative_stop - relative_start
            total_selection = slice(current_frame, current_frame + relative_span)
            print(f"relative_start = {relative_start}")
            print(f"relative_stop = {relative_stop}")
            print(f"relative_span = {relative_span}")
            print(f"total_selection = {total_selection}")
            video[total_selection, ...] = self._imaging_extractors[extractor_index].get_video(
                start_frame=relative_start, end_frame=relative_stop
            )
            current_frame += relative_span
        return video

    def get_image_size(self) -> Tuple:
        return self._imaging_extractors[0].get_image_size()

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return self._imaging_extractors[0].get_sampling_frequency()

    def get_channel_names(self) -> list:
        return self._imaging_extractors[0].get_channel_names()

    def get_num_channels(self) -> int:
        return self._imaging_extractors[0].get_num_channels()
