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

        cls.test_dir = Path(tempfile.mkdtemp())

        # denoised traces
        dataset = zarr.open(folder_path + "/C.zarr")
        cls.denoised_traces = np.transpose(dataset["C"])
        cls.num_frames = len(dataset["frame"][:])
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
        cls.image_size = (dataset["height"].shape[0], dataset["width"].shape[0])
        cls.num_rois = dataset["unit_id"].shape[0]
        # background mask
        dataset = zarr.open(folder_path + "/b.zarr")
        cls.background_image_mask = np.expand_dims(dataset["b"], axis=2)
        # summary image: maximum projection
        cls.maximum_projection_image = np.array(zarr.open(folder_path + "/max_proj.zarr/max_proj"))

    @classmethod
    def tearDownClass(cls):
        # remove the temporary directory and its contents
        shutil.rmtree(cls.test_dir)

    def test_incomplete_extractor_load(self):
        """Check extractor can be initialized when not all traces are available."""
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
        self.test_dir.mkdir(exist_ok=True)

        for folder in folders_to_copy:
            src = Path(self.folder_path) / folder
            dst = self.test_dir / folder
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy(src, dst)

        extractor = MinianSegmentationExtractor(folder_path=self.test_dir)
        traces_dict = extractor.get_traces_dict()
        self.assertEqual(traces_dict["deconvolved"], None)

    def test_image_size(self):
        self.assertEqual(self.extractor.get_image_size(), self.image_size)

    def test_num_frames(self):
        self.assertEqual(self.extractor.get_num_frames(), self.num_frames)

    def test_frame_to_time(self):
        self.assertEqual(self.extractor.frame_to_time(frames=[0]), [0.329])

    def test_num_channels(self):
        self.assertEqual(self.extractor.get_num_channels(), 1)

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
        self.assertEqual(len(timestamps), self.num_frames)
