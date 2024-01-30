"""NumpyMemmapImagingExtractor class.

Classes
-------
NumpyMemmapImagingExtractor
    The class for reading optical imaging data stored in a binary format with numpy.memmap.
"""

import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from tqdm import tqdm

from ...imagingextractor import ImagingExtractor
from typing import Tuple, Dict
from roiextractors.extraction_tools import read_numpy_memmap_video, VideoStructure, DtypeType, PathType
from .memmapextractors import MemmapImagingExtractor


class NumpyMemmapImagingExtractor(MemmapImagingExtractor):
    """An ImagingExtractor class for reading optical imaging data stored in a binary format with numpy.memmap."""

    extractor_name = "NumpyMemmapImagingExtractor"

    def __init__(
        self,
        file_path: PathType,
        video_structure: VideoStructure,
        sampling_frequency: float,
        dtype: DtypeType,
        offset: int = 0,
    ):
        """Create an instance of NumpyMemmapImagingExtractor.

        Parameters
        ----------
        file_path : PathType
            the file_path where the data resides.
        video_structure : VideoStructure
            A VideoStructure instance describing the structure of the image to read. This includes parameters
            such as the number of rows, columns and channels plus which axis (i.e. dimension) of the
            image corresponds to each of them.

            As an example you create one of these structures in the following way:

            from roiextractors.extraction_tools import VideoStructure

            num_rows = 10
            num_columns = 5
            num_channels = 3
            frame_axis = 0
            rows_axis = 1
            columns_axis = 2
            channel_axis = 3

            video_structure = VideoStructure(
                num_rows=num_rows,
                columns=columns,
                num_channels=num_channels,
                rows_axis=rows_axis,
                columns_axis=columns_axis,
                channel_axis=channel_axis,
                frame_axis=frame_axis,
            )

        sampling_frequency : float, optional
            The sampling frequency.
        dtype : DtypeType
            The type of the data to be loaded (int, float, etc.)
        offset : int, optional
            The offset in bytes. Usually corresponds to the number of bytes occupied by the header. 0 by default.
        """
        self.installed = True

        self.file_path = Path(file_path)
        self.video_structure = video_structure
        self._sampling_frequency = float(sampling_frequency)
        self.offset = offset
        self.dtype = dtype

        # Extract video
        self._video = read_numpy_memmap_video(
            file_path=file_path, video_structure=video_structure, dtype=dtype, offset=offset
        )
        self._video = video_structure.transform_video_to_canonical_form(self._video)
        self._num_frames, self._num_rows, self._num_columns, self._num_channels = self._video.shape

        super().__init__(video=self._video)
