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

    def get_image_size(self) -> Tuple[int, int]:
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
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        start_frame = start_frame or 0
        end_frame = end_frame or self.get_num_frames()
        return np.array([self.movie.get_frame_data(i) for i in range(start_frame, end_frame)])

    def get_dtype(self) -> np.dtype:
        return np.dtype(self.movie.data_type)
