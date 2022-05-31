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

    extractor_name = "NumpyMemmapImagingExtractor"

    def __init__(
        self,
        file_path: PathType,
        video_structure: VideoStructure,
        sampling_frequency: float,
        dtype: DtypeType,
        offset: int = 0,
    ):
        """Class for reading optical imaging data stored in a binary format with np.memmap


        Parameters
        ----------
        file_path : PathType
            the file_path where the data resides.
        video_structure : VideoStructure
            A VideoStructure instance describing the structure of the image to read. This includes parameters
            such as the rows, columns and number of channels of the images plus which axis (i.e. dimension) of the
            image corresponds to each of them.

            As an example you create one of these structures in the following way:

            from roiextractors.extraction_tools import VideoStructure

            rows = 10
            columns = 5
            num_channels = 3
            frame_axis = 0
            rows_axis = 1
            columns_axis = 2
            num_channels_axis = 3

            video_structure = VideoStructure(
                rows=rows,
                columns=columns,
                num_channels=num_channels,
                rows_axis=rows_axis,
                columns_axis=columns_axis,
                num_channels_axis=num_channels_axis,
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
        super().__init__()

        self.file_path = Path(file_path)
        self.video_structure = video_structure
        self._sampling_frequency = sampling_frequency
        self.offset = offset
        self.dtype = dtype

        # Extract video
        self._video = read_numpy_memmap_video(
            file_path=file_path, video_structure=video_structure, dtype=dtype, offset=offset
        )

        # Get the image structure as attributes
        self._rows = self.video_structure.rows
        self._columns = self.video_structure.columns
        self._num_channels = self.video_structure.num_channels

        self.frame_axis = self.video_structure.frame_axis
        self._num_frames = self._video.shape[self.frame_axis]

    def get_frames(self, frame_idxs=None):
        if frame_idxs is None:
            frame_idxs = [frame for frame in range(self.get_num_frames())]

        frames = self._video.take(indices=frame_idxs, axis=self.frame_axis)

        return frames

    def get_video(self, start_frame: int = None, end_frame: int = None) -> np.array:
        frame_idxs = range(start_frame, end_frame)
        return self.get_frames(frame_idxs=frame_idxs)

    def get_image_size(self):
        return (self._rows, self._columns)

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_channel_names(self):
        """List of  channels in the recoding.

        Returns
        -------
        channel_names: list
            List of strings of channel names
        """
        pass

    def get_num_channels(self):
        """Total number of active channels in the recording

        Returns
        -------
        no_of_channels: int
            integer count of number of channels
        """
        return self._num_channels

    def get_dtype(self) -> DtypeType:
        return self.dtype
