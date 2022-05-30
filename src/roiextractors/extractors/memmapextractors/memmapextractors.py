from pathlib import Path

import numpy as np
import psutil
from tqdm import tqdm

from ...imagingextractor import ImagingExtractor
from typing import Tuple, Dict

from ...extraction_tools import (
    PathType,
    DtypeType,
)


class MemmapImagingExtractor(ImagingExtractor):

    extractor_name = "MemmapImagingExtractor"

    def __init__(
        self,
        video,
    ):
        """
        Abstract class for memmapable imaging extractors.
        """
        self._video = video
        super().__init__()

    def get_frames(self, frame_idxs=None):
        if frame_idxs is None:
            frame_idxs = [frame for frame in range(self.get_num_frames())]

        frames = self._video.take(indices=frame_idxs, axis=0)

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

    def get_video_shape(self):

        return (self._num_frames, self._rows, self._columns, self._num_channels)

    @staticmethod
    def write_imaging(
        imaging_extractor: ImagingExtractor,
        save_path: PathType,
        verbose: bool = False,
        buffer_gb: int = 0,
    ):
        """
        Static method to write imaging.

        Parameters
        ----------
        imaging: An ImagingExtractor object that inherited from MemmapImagingExtractor
        save_path: str
            path to save the native format to.
        verbose: bool
            Displays a progress bar.
        buffer_gb: int
            The size of the buffer in Gigabytes. Default of 0 forces chunks that cover only one image.
        """
        imaging = imaging_extractor
        file_size_in_bytes = Path(imaging.file_path).stat().st_size
        available_memory_in_bytes = psutil.virtual_memory().available
        buffer_size_in_bytes = buffer_gb * 10**9
        if available_memory_in_bytes < buffer_size_in_bytes:
            raise f"Not enough memory available memory {available_memory_in_bytes* 10**9} for buffer size {buffer_gb}"

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
            video_data_to_save = imaging.get_frames()
            video_memmap[:] = video_data_to_save

        else:
            chunk_size_in_bytes = int(buffer_size_in_bytes)
            type_size = np.dtype(dtype).itemsize

            n_channels = imaging.get_num_channels()
            pixels_per_frame = n_channels * np.product(imaging.get_image_size())
            bytes_per_frame = type_size * pixels_per_frame
            frames_per_chunk = chunk_size_in_bytes // bytes_per_frame

            # If the chunk size is smaller than one image, the iterator goes over one image only.
            frames_per_chunk = max(frames_per_chunk, 1)
            iterator = range(0, num_frames, frames_per_chunk)
            if verbose:
                iterator = tqdm(iterator, ascii=True, desc="Writing to .dat file")

            for frame in iterator:
                start_frame = frame
                end_frame = min(frame + frames_per_chunk, num_frames)

                # Get the video chunk
                video_chunk = imaging.get_video(start_frame=start_frame, end_frame=end_frame)

                # Fit the video chunk in the memmap array
                video_memmap[start_frame:end_frame, ...] = video_chunk

        # Flush the video and delete it
        video_memmap.flush()
        del video_memmap
