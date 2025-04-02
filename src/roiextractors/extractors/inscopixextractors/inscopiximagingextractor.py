"""Inscopix Imaging Extractor."""

import warnings
from typing import Optional, Tuple

import numpy as np

from ...imagingextractor import ImagingExtractor
from ...extraction_tools import PathType


class InscopixImagingExtractor(ImagingExtractor):
    """Extracts imaging data from Inscopix recordings."""

    extractor_name = "InscopixImaging"

    def __init__(self, file_path: PathType):
        """
        Create an InscopixImagingExtractor instance from a single .isx file.

        Parameters
        ----------
        file_path : PathType
            Path to the Inscopix file.
        """
        import isx

        super().__init__(file_path=file_path)
        self.movie = isx.Movie.read(str(file_path))

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        num_pixels = self.movie.spacing.num_pixels
        return num_pixels

    def get_image_size(self) -> Tuple[int, int]:
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        num_pixels = self.movie.spacing.num_pixels
        return num_pixels

    def get_num_frames(self) -> int:
        return self.movie.timing.num_samples

    def get_sampling_frequency(self) -> float:
        return 1 / self.movie.timing.period.secs_float

    def get_channel_names(self) -> list[str]:
        warnings.warn("isx only supports single channel videos.")
        return ["channel_0"]

    def get_num_channels(self) -> int:
        warnings.warn("isx only supports single channel videos.")
        return 1

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: Optional[int] = 0
    ) -> np.ndarray:

        if channel != 0:
            warnings.warn(
                "The 'channel' parameter in get_video() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )

        start_frame = start_frame or 0
        end_frame = end_frame or self.get_num_frames()
        return np.array([self.movie.get_frame_data(i) for i in range(start_frame, end_frame)])

    def get_dtype(self) -> np.dtype:
        return np.dtype(self.movie.data_type)

    def get_raw_data(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> np.ndarray:
        """Get raw data from the video frames.

        Returns
        -------
        np.ndarray : The raw data from the specified frames.
        """
        start_frame = start_frame or 0
        end_frame = end_frame or self.get_num_frames()
        raw_data = np.array([self.movie.get_frame_data(i) for i in range(start_frame, end_frame)])
        return raw_data
