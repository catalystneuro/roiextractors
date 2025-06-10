import shutil
import tempfile
from pathlib import Path

import numpy as np
import zarr
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal

from roiextractors import MinianSegmentationExtractor
from tests.setup_paths import OPHYS_DATA_PATH


class TestMinianSegmentationExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        folder_path = str(OPHYS_DATA_PATH / "segmentation_datasets" / "minian")

        cls.folder_path = Path(folder_path)
        extractor = MinianSegmentationExtractor(folder_path=cls.folder_path)
        cls.extractor = extractor

        # denoised traces
        dataset = zarr.open(folder_path + "/C.zarr")
        cls.denoised_traces = np.transpose(dataset["C"])
        cls.num_samples = 100
        # deconvolved traces
        dataset = zarr.open(folder_path + "/S.zarr")
        cls.deconvolved_traces = np.transpose(dataset["S"])
        # baseline traces
        dataset = zarr.open(folder_path + "/b0.zarr")
        cls.baseline_traces = np.transpose(dataset["b0"])
        # neuropil trace
        dataset = zarr.open(folder_path + "/f.zarr")
        cls.neuropil_trace = np.expand_dims(dataset["f"], axis=1)

        # ROIs masks
        dataset = zarr.open(folder_path + "/A.zarr")
        cls.image_masks = np.transpose(dataset["A"], (1, 2, 0))
        cls.frame_shape = (608, 608)
        cls.num_rois = 3
        # background mask
        dataset = zarr.open(folder_path + "/b.zarr")
        cls.background_image_mask = np.expand_dims(dataset["b"], axis=2)
        # summary image: maximum projection
        cls.maximum_projection_image = np.array(zarr.open(folder_path + "/max_proj.zarr/max_proj"))

    def test_incomplete_extractor_load(self):
        """Check extractor can be initialized when not all traces are available."""
        # Use temporary directory context manager for automatic cleanup
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # temporary directory for testing assertion when some of the files are missing
            folders_to_copy = [
                "A.zarr",
                "C.zarr",
                "b0.zarr",
                "b.zarr",
                "f.zarr",
                "max_proj.zarr",
                ".zgroup",
                "timeStamps.csv",
            ]

            for folder in folders_to_copy:
                src = Path(self.folder_path) / folder
                dst = tmp_path / folder
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy(src, dst)

            extractor = MinianSegmentationExtractor(folder_path=tmp_path)
            traces_dict = extractor.get_traces_dict()
            self.assertEqual(traces_dict["deconvolved"], None)

    def test_frame_shape(self):
        self.assertEqual(self.extractor.get_frame_shape(), self.frame_shape)

    def test_num_samples(self):
        self.assertEqual(self.extractor.get_num_samples(), self.num_samples)

    def test_frame_to_time(self):
        self.assertEqual(self.extractor.frame_to_time(frames=[0]), [0.329])

    def test_num_rois(self):
        self.assertEqual(self.extractor.get_num_rois(), self.num_rois)

    def test_extractor_denoised_traces(self):
        assert_array_equal(self.extractor.get_traces(name="denoised"), self.denoised_traces)

    def test_extractor_neuropil_trace(self):
        assert_array_equal(self.extractor.get_traces(name="neuropil"), self.neuropil_trace)

    def test_extractor_image_masks(self):
        """Test that the image masks are correctly extracted."""
        assert_array_equal(self.extractor.get_roi_image_masks(), self.image_masks)

    def test_extractor_background_image_masks(self):
        """Test that the image masks are correctly extracted."""
        assert_array_equal(self.extractor.get_background_image_masks(), self.background_image_mask)

    def test_maximum_projection_image(self):
        """Test that the mean image is correctly loaded from the extractor."""
        images_dict = self.extractor.get_images_dict()
        assert_array_equal(images_dict["maximum_projection"], self.maximum_projection_image)

    def test_read_timestamps_from_csv(self):
        """Test that timestamps are correctly read from CSV file."""
        # Get timestamps using the extractor
        timestamps = self.extractor._read_timestamps_from_csv()

        # First timestamp should match the expected value
        self.assertEqual(timestamps[0], [0.329])

        # Length should match number of frames
        self.assertEqual(len(timestamps), self.num_samples)
