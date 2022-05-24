import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from tqdm import tqdm

from ...imagingextractor import ImagingExtractor
from typing import Tuple, Dict

from ...extraction_tools import (
    PathType,
    DtypeType,
)


class NumpyMemmapImagingExtractor(ImagingExtractor):

    extractor_name = "NumpyMemmapImagingExtractor"

    def __init__(
        self,
        file_path: PathType,
        frame_shape: Tuple[int, int],
        sampling_frequency: float,
        dtype: DtypeType,
        offset: int = 0,
        image_structure_to_axis: Dict[str, int] = None,
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

    @staticmethod
    def write_imaging(imaging_extractor: ImagingExtractor, save_path: PathType = None, verbose: bool = False):
        """
        Static method to write imaging.

        Parameters
        ----------
        imaging: ImagingExtractor object
            The EXTRACT segmentation object from which an EXTRACT native format
            file has to be generated.
        save_path: str
            path to save the native format.
        overwrite: bool
            If True and save_path is existing, it is overwritten
        """
        imaging = imaging_extractor
        video_to_save = np.memmap(
            save_path,
            shape=(
                imaging.get_num_frames(),
                imaging.get_num_channels(),
                imaging.get_image_size()[0],
                imaging.get_image_size()[1],
            ),
            dtype=imaging.get_dtype(),
            mode="w+",
        )

        if verbose:
            for ch in range(imaging.get_num_channels()):
                print(f"Saving channel {ch}")
                for i in tqdm(range(imaging.get_num_frames())):
                    plane = imaging.get_frames(i, channel=ch)
                    video_to_save[ch, i] = plane
        else:
            for ch in range(imaging.get_num_channels()):
                for i in range(imaging.get_num_frames()):
                    plane = imaging.get_frames(i, channel=ch)
                    video_to_save[ch, i] = plane

        video_to_save.flush()
