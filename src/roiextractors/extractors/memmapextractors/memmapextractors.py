from pathlib import Path

import numpy as np
import psutil
from tqdm import tqdm

from ...imagingextractor import ImagingExtractor
from typing import Tuple, Dict, Optional

from ...extraction_tools import (
    PathType,
    DtypeType,
)


class MemmapImagingExtractor(ImagingExtractor):

    extractor_name = "MemmapImagingExtractor"

    def __init__(
        self,
    ):
        """
        Abstract class for memmapable imaging extractors.
        """
        super().__init__()

        pass

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
    def write_imaging(
        imaging_extractor: ImagingExtractor,
        save_path: PathType,
        verbose: bool = False,
        buffer_size_in_gb: Optional[float] = None,
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
        memmap_shape = imaging.video_structure.build_video_shape(n_frames=num_frames)
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
                end_frame = min(frame + frames_in_buffer, num_frames)

                # Get the video in buffer
                video_in_buffer = imaging.get_video(start_frame=frame, end_frame=end_frame)

                # Fit the video buffer in the memmap array
                indices = np.arange(start=frame, stop=end_frame)
                axis_to_expand = (
                    imaging.video_structure.rows_axis,
                    imaging.video_structure.columns_axis,
                    imaging.video_structure.num_channels_axis,
                )
                indices = np.expand_dims(indices, axis=axis_to_expand)
                frame_axis = imaging.video_structure.frame_axis
                np.put_along_axis(arr=video_memmap, indices=indices, values=video_in_buffer, axis=frame_axis)

        # Flush the video and delete it
        video_memmap.flush()
        del video_memmap
