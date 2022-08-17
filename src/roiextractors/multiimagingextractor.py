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

    def _get_frames_from_an_imaging_extractor(self, extractor_index: int, frame_idxs: ArrayType) -> NumpyArray:
        imaging_extractor = self._imaging_extractors[extractor_index]
        frames = imaging_extractor.get_frames(frame_idxs=frame_idxs)
        return frames

    def get_frames(self, frame_idxs: ArrayType, channel: Optional[int] = 0) -> NumpyArray:
        if isinstance(frame_idxs, (int, np.integer)):
            frame_idxs = [frame_idxs]
        frame_idxs = np.array(frame_idxs)
        assert np.all(frame_idxs < self.get_num_frames()), "'frame_idxs' exceed number of frames"
        extractor_indices = np.searchsorted(self._end_frames, frame_idxs, side="right")
        relative_frame_indices = frame_idxs - np.array(self._start_frames)[extractor_indices]
        # Match frame_idxs to imaging extractors
        extractors_dict = defaultdict(list)
        for extractor_index, frame_index in zip(extractor_indices, relative_frame_indices):
            extractors_dict[extractor_index].append(frame_index)

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

        frames = np.concatenate(frames_to_concatenate, axis=0)
        return frames

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

        print(f"self._start_frames = {self._start_frames}")
        print(f"self._end_frames = {self._end_frames}")
        print(f"start = {start}")
        print(f"stop = {stop}")
        print(f"extractors_spanned = {extractors_spanned}")

        # Early return with simple relative indexing; preserves native return class of that extractor
        if len(extractors_spanned) == 1:
            relative_start = start - self._start_frames[extractors_spanned[0]]
            relative_stop = stop - start + relative_start

            print(f"relative_start = {relative_start}")
            print(f"relative_stop = {relative_stop}")
            return self._imaging_extractors[extractors_spanned[0]].get_video(
                start_frame=relative_start, end_frame=relative_stop
            )

        video_shape = (stop - start,) + self._imaging_extractors[0].get_image_size()
        video = np.empty(shape=video_shape, dtype=self._dtype)
        current_frame = 0

        print(f"video_shape = {video_shape}")

        # Left endpoint; since more than one extractor is spanned, only care about indexing first start frame
        relative_start = start - self._start_frames[extractors_spanned[0]]
        relative_span = self._end_frames[extractors_spanned[0]] - start
        array_frame_slice = slice(current_frame, relative_span)
        print("Left")
        print(f"array_frame_slice = {array_frame_slice}")
        print(f"relative_start = {relative_start}")
        print(f"relative_span = {relative_span}")

        video[array_frame_slice, ...] = self._imaging_extractors[extractors_spanned[0]].get_video(
            start_frame=relative_start
        )
        current_frame += relative_span

        # All inner spans can be written knowing only how long each section is
        for extractor_index in extractors_spanned[1:-1]:
            relative_span = self._end_frames[extractor_index] - self._start_frames[extractor_index]
            array_frame_slice = slice(current_frame, current_frame + relative_span)
            print(f"array_frame_slice = {array_frame_slice}")
            print(f"extractor_index = {extractor_index}")
            print(f"relative_span = {relative_span}")
            video[array_frame_slice, ...] = self._imaging_extractors[extractor_index].get_video()
            current_frame += relative_span

        # Right endpoint; since more than one extractor is spanned, only care about indexing final end frame
        relative_stop = stop - self._start_frames[extractors_spanned[-1]]
        array_frame_slice = slice(current_frame, None)
        print("Right")
        print(f"array_frame_slice = {array_frame_slice}")
        print(f"relative_stop = {relative_stop}")
        video[array_frame_slice, ...] = self._imaging_extractors[extractors_spanned[-1]].get_video(
            end_frame=relative_stop
        )
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
