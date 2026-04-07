"""Defines the base class for memmapable imaging extractors.

Classes
-------
MemmapImagingExtractor
    The base class for memmapable imaging extractors.
"""

import warnings

import numpy as np

from ...imagingextractor import ImagingExtractor


class MemmapImagingExtractor(ImagingExtractor):
    """Abstract class for memmapable imaging extractors."""

    extractor_name = "MemmapImagingExtractor"

    def __init__(
        self,
        video,
    ) -> None:
        """Create a MemmapImagingExtractor instance.

        Parameters
        ----------
        video: numpy.ndarray
            The video data.
        """
        self._video = video
        super().__init__()

    def get_series(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self._num_samples
        sample_indices = range(start_sample, end_sample)
        # Use channel=None to preserve the channel dimension
        return self._video.take(indices=list(sample_indices), axis=0)

    def get_image_shape(self) -> tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return (self._num_rows, self._num_columns)

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_dtype(self) -> np.dtype:
        return self.dtype

    def get_channel_names(self) -> list:
        warnings.warn(
            "get_channel_names is deprecated and will be removed in or after October 2026.",
            FutureWarning,
            stacklevel=2,
        )
        return [f"channel_{i}" for i in range(self._num_channels)]

    def get_volume_shape(self) -> tuple[int, int, int, int]:
        """Return the shape of the video data.

        Returns
        -------
        video_shape: tuple[int, int, int, int]
            The shape of the video data (num_samples, num_rows, num_columns, num_channels).
        """
        return (self._num_samples, self._num_rows, self._num_columns, self._num_channels)

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        # Memory-mapped imaging data does not have native timestamps
        return None
