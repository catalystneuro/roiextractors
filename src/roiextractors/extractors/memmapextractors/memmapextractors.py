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
        save_path: PathType = None,
        verbose: bool = False,
        buffer_data: bool = False,
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
        buffer_data: bool
            Forces chunk to occur even if memmory is available
        """
        imaging = imaging_extractor
        memory_safety_margin = 0.80  # Accept a file smaller than 80 per cent of available memory
        file_size_in_bytes = Path(imaging.file_path).stat().st_size
        available_memory_in_bytes = psutil.virtual_memory().available

        memory_limit = available_memory_in_bytes * memory_safety_margin
        if file_size_in_bytes < memory_limit and not buffer_data:
            video_data_to_save = imaging.get_frames()
            memmap_shape = video_data_to_save.shape
            video_memmap = np.memmap(
                save_path,
                shape=memmap_shape,
                dtype=imaging.get_dtype(),
                mode="w+",
            )

            video_memmap[:] = video_data_to_save
            video_memmap.flush()

        else:
            chunk_size_in_bytes = int(available_memory_in_bytes * memory_safety_margin)
            dtype = imaging.get_dtype()
            type_size = np.dtype(dtype).itemsize

            n_channels = imaging.get_num_channels()
            pixels_per_frame = n_channels * np.product(imaging.get_image_size())
            bytes_per_frame = type_size * pixels_per_frame
            frames_per_chunk = chunk_size_in_bytes // bytes_per_frame

            num_frames = imaging.get_num_frames()
            memmap_shape = imaging.video_structure.build_video_shape(n_frames=num_frames)

            iterator = range(0, num_frames, frames_per_chunk)
            if verbose:
                iterator = tqdm(iterator, ascii=True, desc="Writing to .dat file")

            for frame in iterator:
                end_frame = min(frame + frames_per_chunk, num_frames)

                # Get the video chunk
                video_chunk = imaging.get_video(start_frame=frame, end_frame=end_frame)

                # Load the memmap
                video_memmap = np.memmap(
                    save_path,
                    shape=memmap_shape,
                    dtype=dtype,
                    mode="w+",
                )

                # Fit the video chunk in the memmap array
                indices = np.arange(start=frame, stop=end_frame)
                axis_to_expand = (
                    imaging.video_structure.rows_axis,
                    imaging.video_structure.columns_axis,
                    imaging.video_structure.num_channels_axis,
                )
                indices = np.expand_dims(indices, axis=axis_to_expand)
                frame_axis = imaging.video_structure.frame_axis
                np.put_along_axis(arr=video_memmap, indices=indices, values=video_chunk, axis=frame_axis)

                # Flush to liberate memory for next iteration
                video_memmap.flush()
                del video_memmap
