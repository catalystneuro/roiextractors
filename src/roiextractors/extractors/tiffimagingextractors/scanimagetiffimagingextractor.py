"""Specialized extractor for reading TIFF files produced via ScanImage."""
from pathlib import Path
from typing import Optional, Tuple
from warnings import warn

import numpy as np

from ...extraction_tools import PathType, FloatType, ArrayType, get_package
from ...imagingextractor import ImagingExtractor


def _get_scanimage_reader() -> type:
    return get_package(
        package_name="ScanImageTiffReader", installation_instructions="pip install scanimage-tiff-reader"
    ).ScanImageTiffReader


class ScanImageTiffImagingExtractor(ImagingExtractor):
    """Specialized extractor for reading TIFF files produced via ScanImage."""

    extractor_name = "ScanImageTiffImaging"
    is_writable = True
    mode = "file"

    def __init__(
        self,
        file_path: PathType,
        sampling_frequency: FloatType,
    ):
        """
        Specialized extractor for reading TIFF files produced via ScanImage.

        This extractor allows for lazy accessing of slices, unlike
        :py:class:`~roiextractors.extractors.tiffimagingextractors.TiffImagingExtractor`.
        However, direct slicing of the underlying data structure is not equivalent to a numpy memory map.

        Parameters
        ----------
        file_path : PathType
            Path to the TIFF file.
        sampling_frequency : float
            The frequency at which the frames were sampled, in Hz.
        """
        ScanImageTiffReader = _get_scanimage_reader()

        super().__init__()
        self.file_path = Path(file_path)
        self._sampling_frequency = sampling_frequency
        valid_suffixes = [".tiff", ".tif", ".TIFF", ".TIF"]
        if self.file_path.suffix not in valid_suffixes:
            suffix_string = ", ".join(valid_suffixes[:-1]) + f", or {valid_suffixes[-1]}"
            warn(
                f"Suffix ({self.file_path.suffix}) is not of type {suffix_string}! "
                f"The {self.extractor_name}Extractor may not be appropriate for the file."
            )

        with ScanImageTiffReader(str(self.file_path)) as io:
            shape = io.shape()  # [frames, rows, columns]
        if len(shape) == 3:
            self._num_frames, self._num_rows, self._num_columns = shape
            self._num_channels = 1
        else:  # no example file for multiple color channels or depths
            raise NotImplementedError(
                "Extractor cannot handle 4D TIFF data. Please raise an issue to request this feature: "
                "https://github.com/catalystneuro/roiextractors/issues "
            )

    def get_frames(self, frame_idxs: ArrayType, channel: int = 0) -> np.ndarray:
        ScanImageTiffReader = _get_scanimage_reader()

        squeeze_data = False
        if isinstance(frame_idxs, int):
            squeeze_data = True
            frame_idxs = [frame_idxs]

        if not all(np.diff(frame_idxs) == 1):
            return np.concatenate([self._get_single_frame(idx=idx) for idx in frame_idxs])
        else:
            with ScanImageTiffReader(filename=str(self.file_path)) as io:
                frames = io.data(beg=frame_idxs[0], end=frame_idxs[-1] + 1)
                if squeeze_data:
                    frames = frames.squeeze()
            return frames

    def _get_single_frame(self, idx: int) -> np.ndarray:
        """
        Data accessed through an open ScanImageTiffReader io gets scrambled if there are multiple calls.

        Thus, open fresh io in context each time something is needed.
        """
        ScanImageTiffReader = _get_scanimage_reader()

        with ScanImageTiffReader(str(self.file_path)) as io:
            return io.data(beg=idx, end=idx + 1)

    def get_video(self, start_frame=None, end_frame=None, channel: Optional[int] = 0) -> np.ndarray:
        ScanImageTiffReader = _get_scanimage_reader()

        with ScanImageTiffReader(filename=str(self.file_path)) as io:
            return io.data(beg=start_frame, end=end_frame)

    def get_image_size(self) -> Tuple[int, int]:
        return (self._num_rows, self._num_columns)

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_num_channels(self) -> int:
        return self._num_channels

    def get_channel_names(self) -> list:
        pass
