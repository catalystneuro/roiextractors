import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from tqdm import tqdm

from ...imagingextractor import ImagingExtractor
from typing import Tuple, Dict
from roiextractors.extraction_tools import read_numpy_memmap_video, VideoStructure

from ...extraction_tools import (
    PathType,
    DtypeType,
)


class NumpyMemmapImagingExtractor(ImagingExtractor):

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
        frame_shape : tuple
            The frame shape of the image determines how each frame looks. Examples:
            (n_channels, rows, columns), (rows, columns, n_channels), (n_channels, columns, rows), etc.
            Note that n_channels is 1 for grayscale and 3 for color images.
        dtype : DtypeType
            The type of the data to be loaded (int, float, etc.)
        offset : int, optional
            The offset in bytes. Usually corresponds to the number of bytes occupied by the header. 0 by default.
        sampling_frequency : float, optional
            The sampling frequency.
        image_structure_to_axis : dict, optional
            A dictionary indicating what axis corresponds to what in the memmap. The default values are:
            dict(frame_axis=0, num_channels=1, rows=2, columns=3)
            frame_axis=0 indicates that the first axis corresponds to the frames (usually time)
            num_channels=1 here indicates that the first axis corresponds to the  n_channels.
            rows=2 indicates that the rows in the image are in the second axis
            columns=3 indicates that columns is the last axis or dimension in this structure.

            Notice that this should correspond with frame_shape.
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

    @staticmethod
    def write_imaging(imaging_extractor: ImagingExtractor, save_path: PathType = None, verbose: bool = False):
        """
        Static method to write imaging.

        Parameters
        ----------
        imaging: ImagingExtractor object
        save_path: str
            path to save the native format.
        overwrite: bool
            If True and save_path is existing, it is overwritten
        """
        imaging = imaging_extractor
        video_data_to_save = imaging.get_frames()[:]
        memmap_shape = video_data_to_save.shape
        video_memmap = np.memmap(
            save_path,
            shape=memmap_shape,
            dtype=imaging.get_dtype(),
            mode="w+",
        )

        video_memmap[:] = video_data_to_save
        video_memmap.flush()
