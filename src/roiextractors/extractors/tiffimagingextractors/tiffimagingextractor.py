"""A TIFF imaging extractor for TIFF files.

Classes
-------
TiffImagingExtractor
    A TIFF imaging extractor for TIFF files.
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple
from warnings import warn

import numpy as np
from tqdm import tqdm

from ...extraction_tools import (
    FloatType,
    PathType,
    get_package,
    raise_multi_channel_or_depth_not_implemented,
)
from ...imagingextractor import ImagingExtractor


class TiffImagingExtractor(ImagingExtractor):
    """A ImagingExtractor for TIFF files."""

    extractor_name = "TiffImaging"
    mode = "file"

    def __init__(self, file_path: PathType, sampling_frequency: FloatType):
        """Create a TiffImagingExtractor instance from a TIFF file.

        Parameters
        ----------
        file_path : PathType
            Path to the TIFF file.
        sampling_frequency : float
            The frequency at which the frames were sampled, in Hz.
        """
        tifffile = get_package(package_name="tifffile")

        super().__init__()
        self.file_path = Path(file_path)
        self._sampling_frequency = sampling_frequency
        if self.file_path.suffix not in [".tiff", ".tif", ".TIFF", ".TIF"]:
            warn(
                "File suffix ({self.file_path.suffix}) is not one of .tiff, .tif, .TIFF, or .TIF! "
                "The TiffImagingExtractor may not be appropriate."
            )

        with tifffile.TiffFile(self.file_path) as tif:
            self._num_channels = len(tif.series)

        try:
            self._video = tifffile.memmap(self.file_path, mode="r")
        except Exception as e:

            try:
                with tifffile.TiffFile(self.file_path) as tif:
                    self._video = tif.asarray()
                warn(
                    f"memmap of TIFF file could not be established due to the following error: {e}. "
                    "Reading entire matrix into memory. Consider using the ScanImageTiffSinglePlaneImagingExtractor or ScanImageTiffMultiPlaneImagingExtractor for lazy data access.",
                    stacklevel=2,
                )
            except Exception as e2:
                raise RuntimeError(
                    f"Memory mapping failed: {e}. \n"
                    f"Attempt to read the TIFF file directly also failed: {e2}. \n"
                    f"Consider using ScanImageTiffSinglePlaneImagingExtractor or ScanImageTiffMultiPlaneImagingExtractor for lazy data access, check the file integrity. \n"
                    f"If problems persist, please report an issue at roiextractors/issues."
                )

        shape = self._video.shape
        if len(shape) == 3:
            self._num_samples, self._num_rows, self._num_columns = shape
            self._num_channels = 1
        else:
            raise_multi_channel_or_depth_not_implemented(extractor_name=self.extractor_name)

        self._kwargs = {
            "file_path": str(Path(file_path).absolute()),
            "sampling_frequency": sampling_frequency,
        }

    def get_frames(self, frame_idxs, channel: int = 0):
        """Get specific video frames from indices.

        Parameters
        ----------
        frame_idxs: array-like
            Indices of frames to return.
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
        return self._video[frame_idxs, ...]

    def get_series(self, start_sample=None, end_sample=None) -> np.ndarray:
        return self._video[start_sample:end_sample, ...]

    def get_video(self, start_frame=None, end_frame=None, channel: Optional[int] = 0) -> np.ndarray:
        """Get the video frames.

        Parameters
        ----------
        start_frame: int, optional
            Start frame index (inclusive).
        end_frame: int, optional
            End frame index (exclusive).
        channel: int, optional
            Channel index. Deprecated: This parameter will be removed in August 2025.

        Returns
        -------
        video: numpy.ndarray
            The video frames.

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_series() instead.
        """
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

    def get_num_samples(self):
        return self._num_samples

    def get_num_frames(self):
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

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_num_channels(self):
        warn(
            "get_num_channels() is deprecated and will be removed in or after August 2025.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._num_channels

    def get_channel_names(self):
        pass

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # Basic TIFF files do not have native timestamps
        return None

    @staticmethod
    def write_imaging(imaging, save_path, overwrite: bool = False, chunk_size=None, verbose=True):
        """Write a TIFF file from an ImagingExtractor.

        Parameters
        ----------
        imaging : ImagingExtractor
            The ImagingExtractor to be written to a TIFF file.
        save_path : str or PathType
            The path to save the TIFF file.
        overwrite : bool
            If True, will overwrite the file if it exists. Otherwise will raise an error if the file exists.
        chunk_size : int or None
            If None, will write the entire video to a single TIFF file. Otherwise will write the video
            in chunk_size frames at a time.
        verbose : bool
            If True, will print progress bar.
        """
        warn(
            "The write_imaging function is deprecated and will be removed on or after September 2025. ROIExtractors is no longer supporting write operations.",
            DeprecationWarning,
            stacklevel=2,
        )
        tifffile = get_package(package_name="tifffile")

        save_path = Path(save_path)
        assert save_path.suffix in [
            ".tiff",
            ".tif",
            ".TIFF",
            ".TIF",
        ], "'save_path' file is not an .tiff file"

        if save_path.is_file():
            if not overwrite:
                raise FileExistsError("The specified path exists! Use overwrite=True to overwrite it.")
            else:
                save_path.unlink()

        if chunk_size is None:
            tifffile.imwrite(save_path, imaging.get_video())
        else:
            num_samples = imaging.get_num_samples()
            # chunk size is not None
            n_chunk = num_samples // chunk_size
            if num_samples % chunk_size > 0:
                n_chunk += 1
            if verbose:
                chunks = tqdm(range(n_chunk), ascii=True, desc="Writing to .tiff file")
            else:
                chunks = range(n_chunk)
            with tifffile.TiffWriter(save_path) as tif:
                for i in chunks:
                    video = imaging.get_video(
                        start_frame=i * chunk_size,
                        end_frame=min((i + 1) * chunk_size, num_samples),
                    )
                    chunk_frames = np.squeeze(video)
                    tif.save(chunk_frames, contiguous=True, metadata=None)
