"""Defines the base class for memmapable imaging extractors.

Classes
-------
MemmapImagingExtractor
    The base class for memmapable imaging extractors.
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple
from warnings import warn

import numpy as np
import psutil
from tqdm import tqdm

from ...extraction_tools import DtypeType, PathType
from ...imagingextractor import ImagingExtractor


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
        """Get specific video frames from indices.

        Parameters
        ----------
        frame_idxs: array-like, optional
            Indices of frames to return. If None, returns all frames.
        channel: int, optional
            Channel index. Deprecated: This parameter will be removed in August 2025.

        Returns
        -------
        frames: numpy.ndarray
            The video frames.
        """
        if channel != 0:
            warn(
                "The 'channel' parameter in get_frames() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )

        if frame_idxs is None:
            frame_idxs = [frame for frame in range(self.get_num_frames())]

        frames = self._video.take(indices=frame_idxs, axis=0)
        if channel is not None:
            frames = frames[..., channel]

        return frames

    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self._num_samples
        frame_idxs = range(start_sample, end_sample)
        # Use channel=None to preserve the channel dimension
        return self._video.take(indices=list(frame_idxs), axis=0)

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: Optional[int] = 0
    ) -> np.ndarray:
        warnings.warn(
            "get_video() is deprecated and will be removed in or after September 2025. " "Use get_series() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if channel != 0:
            warn(
                "The 'channel' parameter in get_video() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.get_series(start_sample=start_frame, end_sample=end_frame)

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return (self._num_rows, self._num_columns)

    def get_image_size(self) -> Tuple[int, int]:
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        return (self._num_rows, self._num_columns)

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_num_frames(self) -> int:
        """Get the number of frames in the video.

        Returns
        -------
        num_frames: int
            Number of frames in the video.

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_num_samples() instead.
        """
        warnings.warn(
            "get_num_frames() is deprecated and will be removed in or after September 2025. "
            "Use get_num_samples() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_num_samples()

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_channel_names(self):
        pass

    def get_num_channels(self) -> int:
        warn(
            "get_num_channels() is deprecated and will be removed in or after August 2025.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._num_channels

    def get_dtype(self) -> DtypeType:
        return self.dtype

    def get_volume_shape(self) -> Tuple[int, int, int, int]:
        """Return the shape of the video data.

        Returns
        -------
        video_shape: Tuple[int, int, int, int]
            The shape of the video data (num_samples, num_rows, num_columns, num_channels).
        """
        return (self._num_samples, self._num_rows, self._num_columns, self._num_channels)

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # Memory-mapped imaging data does not have native timestamps
        return None

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
        warnings.warn(
            "The write_imaging function is deprecated and will be removed on or after September 2025. ROIExtractors is no longer supporting write operations.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        memmap_shape = imaging.get_volume_shape()
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
            pixels_per_frame = n_channels * np.prod(imaging.get_image_size())
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
