import os
import shutil
import tempfile
from pathlib import Path
from typing import Union, Optional

import numpy as np
from tqdm import tqdm

from .extraction_tools import PathType, check_get_frames_args
from .imagingextractor import ImagingExtractor

PathType = Union[str, Path]
DtypeType = Union[str, np.dtype]
OptionalDtypeType = Optional[DtypeType]
DtypeType = Union[str, np.dtype]


class MemmapImagingExtractor(ImagingExtractor):

    extractor_name = "MemmapImagingExtractor"

    def __init__(
        self,
        file_path: PathType,
        frame_shape: tuple,
        dtype: DtypeType,
        offset: int = 0,
        sampling_frequency: float = 0,
        image_structure_to_axis: dict = None,
    ):
        """Class for reading binary data.


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

        self.file_path = file_path
        self._sampling_frequency = sampling_frequency
        self.offset = offset
        self.dtype = dtype

        # Get the structure, apply default if not available
        self.frame_shape = frame_shape
        image_structure_to_axis = dict() if image_structure_to_axis is None else image_structure_to_axis
        self.image_structure_to_axis = dict(frame_axis=0, num_channels=1, rows=2, columns=3)
        self.image_structure_to_axis.update(image_structure_to_axis)
        self.frame_axis = self.image_structure_to_axis["frame_axis"]

        # Extract video
        self._video = self.read_binary_video()

        # Get the image structure as attributes
        self._rows = self._video.shape[self.image_structure_to_axis["rows"]]
        self._columns = self._video.shape[self.image_structure_to_axis["columns"]]
        self._num_channels = self._video.shape[self.image_structure_to_axis["num_channels"]]
        self._num_frames = self._video.shape[self.image_structure_to_axis["frame_axis"]]

    def read_binary_video(self):
        file = self.file_path.open()
        file_descriptor = file.fileno()
        file_size_bytes = os.fstat(file_descriptor).st_size

        pixels_per_frame = np.prod(self.frame_shape)
        type_size = np.dtype(self.dtype).itemsize
        frame_size_bytes = pixels_per_frame * type_size

        bytes_available = file_size_bytes - self.offset
        number_of_frames = bytes_available // frame_size_bytes

        memmap_shape = list(self.frame_shape)
        memmap_shape.insert(self.frame_axis, number_of_frames)
        memmap_shape = tuple(memmap_shape)

        video_memap = np.memmap(self.file_path, offset=self.offset, dtype=self.dtype, mode="r", shape=memmap_shape)

        return video_memap

    def get_frames(self, frame_idxs=None):
        if frame_idxs is None:
            frame_idxs = [frame for frame in range(self.get_num_frames())]
        return self._video.take(indices=frame_idxs, axis=self.frame_axis)

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
