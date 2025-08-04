"""Defines the MultiImagingExtractor class.

Classes
-------
MultiImagingExtractor
    This class is used to combine multiple ImagingExtractor objects by frames.
"""

import warnings
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .extraction_tools import ArrayType, NumpyArray
from .imagingextractor import ImagingExtractor


class MultiImagingExtractor(ImagingExtractor):
    """Class to combine multiple ImagingExtractor objects by frames."""

    extractor_name = "MultiImagingExtractor"
    installation_mesg = ""

    def __init__(self, imaging_extractors: List[ImagingExtractor]):
        """Initialize a MultiImagingExtractor object from a list of ImagingExtractors.

        Parameters
        ----------
        imaging_extractors: list of ImagingExtractor
            list of imaging extractor objects
        """
        super().__init__()
        assert isinstance(imaging_extractors, list), "Enter a list of ImagingExtractor objects as argument"
        assert all(isinstance(imaging_extractor, ImagingExtractor) for imaging_extractor in imaging_extractors)
        self._imaging_extractors = imaging_extractors

        # Checks that properties are consistent between extractors
        self._check_consistency_between_imaging_extractors()

        self._start_frames, self._end_frames = [], []
        num_samples = 0
        for imaging_extractor in self._imaging_extractors:
            self._start_frames.append(num_samples)
            num_samples = num_samples + imaging_extractor.get_num_samples()
            self._end_frames.append(num_samples)
        self._num_samples = num_samples

        if any((getattr(imaging_extractor, "_times") is not None for imaging_extractor in self._imaging_extractors)):
            times = self._get_times()
            self.set_times(times=times)

    def _check_consistency_between_imaging_extractors(self):
        """Check that essential properties are consistent between extractors so that they can be combined appropriately.

        Raises
        ------
        AssertionError
            If any of the properties are not consistent between extractors.

        Notes
        -----
        This method checks the following properties:
            - sampling frequency
            - image size
            - number of channels
            - channel names
            - data type
        """
        properties_to_check = dict(
            get_sampling_frequency="The sampling frequency",
            get_sample_shape="The shape of a sample",
            get_channel_names="The name of the channels",
            get_dtype="The data type",
            get_num_samples="The number of samples",
        )
        for method, property_message in properties_to_check.items():
            values = [getattr(extractor, method)() for extractor in self._imaging_extractors]
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
        frame_indices = np.array([*range(self._start_frames[0], self._end_frames[-1])])
        times = self.frame_to_time(frames=frame_indices)

        for extractor_index, extractor in enumerate(self._imaging_extractors):
            if getattr(extractor, "_times") is not None:
                to_replace = [*range(self._start_frames[extractor_index], self._end_frames[extractor_index])]
                times[to_replace] = extractor._times

        return times

    def _get_frames_from_an_imaging_extractor(self, extractor_index: int, frame_idxs: ArrayType) -> NumpyArray:
        """Get frames from a single imaging extractor.

        Parameters
        ----------
        extractor_index: int
            Index of the imaging extractor to use.
        frame_idxs: array_like
            Indices of the frames to get.

        Returns
        -------
        frames: numpy.ndarray
            Array of frames.
        """
        imaging_extractor = self._imaging_extractors[extractor_index]
        frames = imaging_extractor.get_frames(frame_idxs=frame_idxs)
        return frames

    def get_dtype(self):
        return self._imaging_extractors[0].get_dtype()

    def get_frames(self, frame_idxs: ArrayType, channel: Optional[int] = 0) -> NumpyArray:
        """Get specific video frames from indices.

        Parameters
        ----------
        frame_idxs: array-like
            Indices of frames to return.
        channel: int, optional
            Channel index. Deprecated: This parameter will be removed in August 2025.

        Returns
        -------
        frames: numpy.ndarray
            The video frames.
        """
        if channel != 0:
            warnings.warn(
                "The 'channel' parameter in get_frames() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
        if isinstance(frame_idxs, (int, np.integer)):
            frame_idxs = [frame_idxs]
        frame_idxs = np.array(frame_idxs)
        assert np.all(frame_idxs < self.get_num_samples()), "'frame_idxs' exceed number of samples"
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

    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
        """Get the video frames.

        Parameters
        ----------
        start_sample: int, optional
            Start sample index (inclusive).
        end_sample: int, optional
            End sample index (exclusive).

        Returns
        -------
        series: numpy.ndarray
            The video frames.
        """
        start = start_sample if start_sample is not None else 0
        stop = end_sample if end_sample is not None else self.get_num_samples()
        extractors_range = np.searchsorted(self._end_frames, (start, stop - 1), side="right")
        extractors_spanned = list(
            range(extractors_range[0], min(extractors_range[-1] + 1, len(self._imaging_extractors)))
        )

        # Early return with simple relative indexing; preserves native return class of that extractor
        if len(extractors_spanned) == 1:
            relative_start = start - self._start_frames[extractors_spanned[0]]
            relative_stop = stop - start + relative_start

            return self._imaging_extractors[extractors_spanned[0]].get_series(
                start_sample=relative_start, end_sample=relative_stop
            )

        series_shape = (stop - start,) + self._imaging_extractors[0].get_image_size()
        series = np.empty(shape=series_shape, dtype=self.get_dtype())
        current_frame = 0

        # Left endpoint; since more than one extractor is spanned, only care about indexing first start frame
        relative_start = start - self._start_frames[extractors_spanned[0]]
        relative_span = self._end_frames[extractors_spanned[0]] - start
        array_frame_slice = slice(current_frame, relative_span)
        series[array_frame_slice, ...] = self._imaging_extractors[extractors_spanned[0]].get_series(
            start_sample=relative_start
        )
        current_frame += relative_span

        # All inner spans can be written knowing only how long each section is
        for extractor_index in extractors_spanned[1:-1]:
            relative_span = self._end_frames[extractor_index] - self._start_frames[extractor_index]
            array_frame_slice = slice(current_frame, current_frame + relative_span)
            series[array_frame_slice, ...] = self._imaging_extractors[extractor_index].get_series()
            current_frame += relative_span

        # Right endpoint; since more than one extractor is spanned, only care about indexing final end frame
        relative_stop = stop - self._start_frames[extractors_spanned[-1]]
        array_frame_slice = slice(current_frame, None)
        series[array_frame_slice, ...] = self._imaging_extractors[extractors_spanned[-1]].get_series(
            end_sample=relative_stop
        )

        return series

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        """Get the video frames.

        Parameters
        ----------
        start_frame: int, optional
            Start frame index (inclusive).
        end_frame: int, optional
            End frame index (exclusive).
        channel: int, optional
            Channel index. Deprecated: This parameter will be removed in August 2025.

        Returns
        -------
        video: numpy.ndarray
            The video frames.

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_series() instead.
        """
        warnings.warn(
            "get_video() is deprecated and will be removed in or after September 2025. " "Use get_series() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if channel != 0:
            warnings.warn(
                "The 'channel' parameter in get_video() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
            raise NotImplementedError(
                f"MultiImagingExtractors for multiple channels have not yet been implemented! (Received '{channel}'."
            )

        return self.get_series(start_sample=start_frame, end_sample=end_frame)

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._imaging_extractors[0].get_image_shape()

    def get_image_size(self) -> Tuple[int, int]:
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._imaging_extractors[0].get_image_size()

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_num_frames(self) -> int:
        """Get the number of frames in the video.

        Returns
        -------
        num_frames: int
            Number of frames in the video.

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_num_samples() instead.
        """
        warnings.warn(
            "get_num_frames() is deprecated and will be removed in or after September 2025. "
            "Use get_num_samples() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_num_samples()

    def get_sampling_frequency(self) -> float:
        return self._imaging_extractors[0].get_sampling_frequency()

    def get_channel_names(self) -> list:
        return self._imaging_extractors[0].get_channel_names()

    def get_num_channels(self) -> int:
        warnings.warn(
            "get_num_channels() is deprecated and will be removed in or after August 2025.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._imaging_extractors[0].get_num_channels()

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # MultiImagingExtractor combines multiple extractors with potentially different timestamp behaviors.
        # Implementing native timestamp concatenation is complex due to potential timestamp overlaps,
        # different sampling rates, and mixed native/calculated timestamps across child extractors.
        # For now, return None to use calculated timestamps based on sampling frequency.
        return None
