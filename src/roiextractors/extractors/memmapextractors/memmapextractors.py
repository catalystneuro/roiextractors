"""Defines the base class for memmapable imaging extractors.

Classes
-------
MemmapImagingExtractor
    The base class for memmapable imaging extractors.
"""

from warnings import warn

import numpy as np

from ...extraction_tools import DtypeType
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

    def get_frames(self, frame_idxs=None, channel: int | None = 0) -> np.ndarray:
        """Get specific video frames from indices.

        Parameters
        ----------
        frame_idxs: array-like, optional
            Indices of frames to return. If None, returns all frames.
        channel: int, optional
            Channel index. Deprecated: This parameter will be removed in August 2025.

        Returns
        -------
        frames: numpy.ndarray
            The video frames.
        """
        if channel != 0:
            warn(
                "The 'channel' parameter in get_frames() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )

        if frame_idxs is None:
            frame_idxs = [frame for frame in range(self.get_num_samples())]

        frames = self._video.take(indices=frame_idxs, axis=0)
        if channel is not None:
            frames = frames[..., channel]

        return frames

    def get_series(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self._num_samples
        frame_idxs = range(start_sample, end_sample)
        # Use channel=None to preserve the channel dimension
        return self._video.take(indices=list(frame_idxs), axis=0)

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

    def get_channel_names(self):
        pass

    def get_dtype(self) -> DtypeType:
        return self.dtype

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
