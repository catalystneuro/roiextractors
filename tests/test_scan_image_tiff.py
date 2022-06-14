import unittest

from hdmf.testing import TestCase
from numpy.testing import assert_array_equal

from roiextractors import TiffImagingExtractor, ScanImageTiffImagingExtractor

from .setup_paths import OPHYS_DATA_PATH


class TestScanImageTiffExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_path = OPHYS_DATA_PATH / "imaging_datasets" / "Tif" / "sample_scanimage.tiff"
        cls.imaging_extractor = ScanImageTiffImagingExtractor(file_path=cls.file_path)
        cls.data = cls.imaging_extractor._scan_image_io.data()

    def test_tiff_non_memmap_warning(self):
        with self.assertWarnsWith(
            warn_type=UserWarning,
            exc_msg=(
                "memmap of TIFF file could not be established. Reading entire matrix into memory. "
                "Consider using the ScanImageTiffExtractor for lazy data access."
            ),
        ):
            TiffImagingExtractor(file_path=self.file_path, sampling_frequency=60.0)

    def test_scan_image_tiff_consecutive_frames(self):
        frame_idxs = [12, 13]
        assert_array_equal(self.imaging_extractor.get_frames(frame_idxs=frame_idxs), self.data[frame_idxs])

    def test_scan_image_tiff_nonconsecutive_frames(self):
        frame_idxs = [12, 16]

        assert_array_equal(self.imaging_extractor.get_frames(frame_idxs=frame_idxs), self.data[frame_idxs, ...])

    def test_scan_image_get_video(self):
        assert_array_equal(self.imaging_extractor.get_video(), self.data)

    def test_scan_image_tiff_sampling_frequency_from_file(self):
        assert self.imaging_extractor.get_sampling_frequency() == 60.0


if __name__ == "__main__":
    unittest.main()
