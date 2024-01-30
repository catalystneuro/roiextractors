"""Defines the base class for memmapable imaging extractors.

Classes
-------
MemmapImagingExtractor
    The base class for memmapable imaging extractors.
"""

from pathlib import Path

import numpy as np
import psutil
from tqdm import tqdm

from ...imagingextractor import ImagingExtractor
from typing import Tuple, Dict, Optional

from ...extraction_tools import (
    PathType,
    DtypeType,
    NumpyArray,
)


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

    def get_frames(self, frame_idxs=None, channel: Optional[int] = 0) -> np.ndarray:
        if frame_idxs is None:
            frame_idxs = [frame for frame in range(self.get_num_frames())]

        frames = self._video.take(indices=frame_idxs, axis=0)
        if channel is not None:
            frames = frames[..., channel]

        return frames

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: Optional[int] = 0
    ) -> np.ndarray:
        frame_idxs = range(start_frame, end_frame)
        return self.get_frames(frame_idxs=frame_idxs, channel=channel)

    def get_image_size(self) -> Tuple[int, int]:
        return (self._num_rows, self._num_columns)

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_channel_names(self):
        pass

    def get_num_channels(self) -> int:
        return self._num_channels

    def get_dtype(self) -> DtypeType:
        return self.dtype

    def get_video_shape(self) -> Tuple[int, int, int, int]:
        """Return the shape of the video data.

        Returns
        -------
        video_shape: Tuple[int, int, int, int]
            The shape of the video data (num_frames, num_rows, num_columns, num_channels).
        """
        return (self._num_frames, self._num_rows, self._num_columns, self._num_channels)

    @staticmethod
    def write_imaging(
        imaging_extractor: ImagingExtractor,
        save_path: PathType,
        verbose: bool = False,
        buffer_size_in_gb: Optional[float] = None,
    ) -> None:
        """Write imaging by flushing to disk.

        Parameters
        ----------
        imaging_extractor: ImagingExtractor
            An ImagingExtractor object that inherited from MemmapImagingExtractor
        save_path: str
            path to save the native format to.
        verbose: bool
            Displays a progress bar.
        buffer_size_in_gb: float
            The size of the buffer in Gigabytes. The default of None results in buffering over one frame at a time.
        """
        # The base and default case is to load one image at a time.
        if buffer_size_in_gb is None:
            buffer_size_in_gb = 0

        imaging = imaging_extractor
        file_size_in_bytes = Path(imaging.file_path).stat().st_size
        available_memory_in_bytes = psutil.virtual_memory().available
        buffer_size_in_bytes = int(buffer_size_in_gb * 1e9)
        if available_memory_in_bytes < buffer_size_in_bytes:
            raise f"Not enough memory available, {available_memory_in_bytes* 1e9} for buffer size {buffer_size_in_gb}"

        num_frames = imaging.get_num_frames()
        memmap_shape = imaging.get_video_shape()
        dtype = imaging.get_dtype()

        # Load the memmap
        video_memmap = np.memmap(
            save_path,
            shape=memmap_shape,
            dtype=dtype,
            mode="w+",
        )

        if file_size_in_bytes < buffer_size_in_bytes:
            video_data_to_save = imaging.get_frames(channel=None)
            video_memmap[:] = video_data_to_save

        else:
            buffer_size_in_bytes = int(buffer_size_in_bytes)
            type_size = np.dtype(dtype).itemsize

            n_channels = imaging.get_num_channels()
            pixels_per_frame = n_channels * np.product(imaging.get_image_size())
            bytes_per_frame = type_size * pixels_per_frame
            frames_in_buffer = buffer_size_in_bytes // bytes_per_frame

            # If the buffer size is smaller than the size of one image, the iterator goes over one image only.
            frames_in_buffer = max(frames_in_buffer, 1)
            iterator = range(0, num_frames, frames_in_buffer)
            if verbose:
                iterator = tqdm(iterator, ascii=True, desc="Writing to .dat file")

            for frame in iterator:
                start_frame = frame
                end_frame = min(frame + frames_in_buffer, num_frames)

                # Get the video chunk
                video_chunk = imaging.get_video(start_frame=start_frame, end_frame=end_frame, channel=None)

                # Fit the video chunk in the memmap array
                video_memmap[start_frame:end_frame, ...] = video_chunk

        # Flush the video and delete it
        video_memmap.flush()
        del video_memmap
