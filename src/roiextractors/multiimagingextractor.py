"""Defines the MultiImagingExtractor class.

Classes
-------
MultiImagingExtractor
    This class is used to combine multiple ImagingExtractor objects by frames.
"""

import warnings
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from .imagingextractor import ImagingExtractor


class MultiImagingExtractor(ImagingExtractor):
    """Class to combine multiple ImagingExtractor objects by frames."""

    extractor_name = "MultiImagingExtractor"

    def __init__(self, imaging_extractors: list[ImagingExtractor]):
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
            - sample shape (image size and number of channels)
            - channel names
            - data type
        """
        properties_to_check = dict(
            get_sampling_frequency="The sampling frequency",
            get_sample_shape="The shape of a sample",
            get_channel_names="The name of the channels",
            get_dtype="The data type",
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
        start_sample = self._start_frames[0]
        end_sample = self._end_frames[-1]
        times = self.get_timestamps(start_sample=start_sample, end_sample=end_sample)

        for extractor_index, extractor in enumerate(self._imaging_extractors):
            if getattr(extractor, "_times") is not None:
                to_replace = [*range(self._start_frames[extractor_index], self._end_frames[extractor_index])]
                times[to_replace] = extractor._times

        return times

    def _get_frames_from_an_imaging_extractor(self, extractor_index: int, sample_indices: ArrayLike) -> np.ndarray:
        """Get samples from a single imaging extractor.

        Parameters
        ----------
        extractor_index: int
            Index of the imaging extractor to use.
        sample_indices: array_like
            Indices of the samples to get.

        Returns
        -------
        samples: numpy.ndarray
            Array of samples.
        """
        imaging_extractor = self._imaging_extractors[extractor_index]
        samples = imaging_extractor.get_samples(sample_indices=sample_indices)
        return samples

    def get_dtype(self):
        return self._imaging_extractors[0].get_dtype()

    def get_series(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
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

        series_shape = (stop - start,) + self._imaging_extractors[0].get_image_shape()
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

    def get_image_shape(self) -> tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._imaging_extractors[0].get_image_shape()

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_sampling_frequency(self) -> float:
        return self._imaging_extractors[0].get_sampling_frequency()

    def get_channel_names(self) -> list:
        warnings.warn(
            "get_channel_names is deprecated and will be removed in May 2026 or after.",
            category=FutureWarning,
            stacklevel=2,
        )
        return self._imaging_extractors[0].get_channel_names()

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        # MultiImagingExtractor combines multiple extractors with potentially different timestamp behaviors.
        # Implementing native timestamp concatenation is complex due to potential timestamp overlaps,
        # different sampling rates, and mixed native/calculated timestamps across child extractors.
        # For now, return None to use calculated timestamps based on sampling frequency.
        return None
