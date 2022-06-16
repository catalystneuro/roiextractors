"""Specialized extractor for reading TIFF files produced via ScanImage."""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ...extraction_tools import (
    PathType,
    check_get_frames_args,
    FloatType,
    ArrayType,
)
from ...imagingextractor import ImagingExtractor

try:
    from ScanImageTiffReader import ScanImageTiffReader

    HAVE_SCAN_IMAGE_TIFF = True
except ImportError:
    HAVE_SCAN_IMAGE_TIFF = False


class ScanImageTiffImagingExtractor(ImagingExtractor):
    """Specialized extractor for reading TIFF files produced via ScanImage."""

    extractor_name = "ScanImageTiffImaging"
    installed = HAVE_SCAN_IMAGE_TIFF
    is_writable = True
    mode = "file"
    installation_mesg = (
        "To use the ScanImageTiffImagingExtractor install scanimage-tiff-reader: "
        "\n\n pip install scanimage-tiff-reader\n\n"
    )

    def __init__(
        self,
        file_path: PathType,
        sampling_frequency: FloatType,
    ):
        """
        Specialized extractor for reading TIFF files produced via ScanImage.

        The generic TiffImagingExtractor has issues loading these types of TIFF files in a lazy (memory mapped) manner.

        The API calls to this extractor, however, have fully lazy behavior; however, direct slicing of the underlying
        data structure is not equivalent to a numpy memory map.

        Parameters
        ----------
        file_path : PathType
            Path to the TIFF file.
        sampling_frequency : float
            The frequency at which the frames were sampled, in Hz.
        """
        assert self.installed, self.installation_mesg
        super().__init__()
        self.file_path = Path(file_path)
        self._sampling_frequency = sampling_frequency
        assert self.file_path.suffix in [".tiff", ".tif", ".TIFF", ".TIF"]

        with ScanImageTiffReader(str(self.file_path)) as io:
            shape = io.shape()  # [frames, rows, cols]
        if len(shape) == 3:
            self._num_frames, self._num_rows, self._num_cols = shape
            self._num_channels = 1
        else:  # no example file for multiple color channels or depths
            raise NotImplementedError(
                "Extractor cannot handle 4D TIFF data. Please raise an issue to request this feature: "
                "https://github.com/catalystneuro/roiextractors/issues "
            )

    @check_get_frames_args
    def get_frames(self, frame_idxs: ArrayType, channel: int = 0) -> np.ndarray:
        if not all(np.diff(frame_idxs) == 1):
            return np.concatenate([self._get_single_frame(idx=idx) for idx in frame_idxs])
        else:
            with ScanImageTiffReader(filename=str(self.file_path)) as io:
                return io.data(beg=frame_idxs[0], end=frame_idxs[-1] + 1)

    def _get_single_frame(self, idx: int) -> np.ndarray:
        """
        Data accessed through an open ScanImageTiffReader io gets scrambled if there are multiple calls.

        Thus, open fresh io in context each time something is needed.
        """
        with ScanImageTiffReader(str(self.file_path)) as io:
            return io.data(beg=idx, end=idx + 1)

    def get_image_size(self) -> Tuple[int, int]:
        return (self._num_rows, self._num_cols)

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_channel_names(self) -> Optional[ArrayType]:
        return self._channel_names

    def get_num_channels(self) -> int:
        return self._num_channels
