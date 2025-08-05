import platform
import unittest
from pathlib import Path
from shutil import copy, rmtree
from tempfile import mkdtemp

import pytest
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal

from roiextractors import ScanImageLegacyImagingExtractor, TiffImagingExtractor
from roiextractors.extractors.tiffimagingextractors.scanimagetiff_utils import (
    _get_scanimage_reader,
)

from .setup_paths import OPHYS_DATA_PATH

is_m_series_mac = platform.system() == "Darwin" and platform.machine() == "arm64"
if (
    is_m_series_mac
):  # Remove this check once scanimage tiff reader is available on ARM -- see https://gitlab.com/vidriotech/scanimagetiffreader-python/-/issues/31
    pytest.skip("ScanImageTiffReader does not support M-series Macs", allow_module_level=True)


class TestScanImageTiffExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_path = OPHYS_DATA_PATH / "imaging_datasets" / "Tif" / "sample_scanimage.tiff"
        cls.tmpdir = Path(mkdtemp())
        cls.imaging_extractor = ScanImageLegacyImagingExtractor(file_path=cls.file_path, sampling_frequency=30.0)
        ScanImageTiffReader = _get_scanimage_reader()
        with ScanImageTiffReader(filename=str(cls.imaging_extractor.file_path)) as io:
            cls.data = io.data()

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.tmpdir)

    def test_tiff_non_memmap_warning(self):
        with self.assertWarnsWith(
            warn_type=UserWarning,
            exc_msg=(
                "memmap of TIFF file could not be established due to the following error: image data are not memory-mappable. "
                "Reading entire matrix into memory. Consider using the ScanImageTiffSinglePlaneImagingExtractor or ScanImageTiffMultiPlaneImagingExtractor for lazy data access."
            ),
        ):
            TiffImagingExtractor(file_path=self.file_path, sampling_frequency=30.0)

    def test_tiff_suffix_warning(self):
        different_suffix_file_path = self.tmpdir / f"{self.file_path.stem}.jpg"
        # different_suffix_file_path.symlink_to(self.file_path)
        copy(src=self.file_path, dst=different_suffix_file_path)
        with self.assertWarnsWith(
            warn_type=UserWarning,
            exc_msg=(
                "Suffix (.jpg) is not of type .tiff, .tif, .TIFF, or .TIF! "
                "The ScanImageLegacyImagingExtractor may not be appropriate for the file."
            ),
        ):
            ScanImageLegacyImagingExtractor(file_path=different_suffix_file_path, sampling_frequency=30.0)

    def test_scan_image_tiff_consecutive_frames(self):
        frame_idxs = [6, 8]
        assert_array_equal(self.imaging_extractor.get_frames(frame_idxs=frame_idxs), self.data[frame_idxs])

    def test_scan_image_tiff_nonconsecutive_frames(self):
        frame_idxs = [3, 6]

        assert_array_equal(self.imaging_extractor.get_frames(frame_idxs=frame_idxs), self.data[frame_idxs, ...])

    def test_scan_image_get_video(self):
        assert_array_equal(self.imaging_extractor.get_video(), self.data)

    def test_scan_image_tiff_sampling_frequency(self):
        assert self.imaging_extractor.get_sampling_frequency() == 30.0

    def test_scan_image_tiff_num_frames(self):
        assert self.imaging_extractor.get_num_frames() == 10

    def test_scan_image_tiff_image_size(self):
        assert self.imaging_extractor.get_image_size() == (256, 256)


if __name__ == "__main__":
    unittest.main()
