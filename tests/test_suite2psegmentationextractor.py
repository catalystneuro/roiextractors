import shutil
import tempfile
from pathlib import Path

import numpy as np
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal

from roiextractors import Suite2pSegmentationExtractor
from tests.setup_paths import OPHYS_DATA_PATH


class TestSuite2pSegmentationExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        folder_path = str(OPHYS_DATA_PATH / "segmentation_datasets" / "suite2p")
        cls.channel_names = ["chan1", "chan2"]
        cls.plane_names = ["plane0", "plane1"]

        cls.folder_path = Path(folder_path)

        extractor = Suite2pSegmentationExtractor(folder_path=folder_path, channel_name="chan1", plane_name="plane0")
        cls.extractor = extractor

        cls.test_dir = Path(tempfile.mkdtemp())

        cls.first_channel_raw_traces = np.load(cls.folder_path / "plane0" / "F.npy").T
        cls.second_channel_raw_traces = np.load(cls.folder_path / "plane0" / "F_chan2.npy").T

        options = np.load(cls.folder_path / "plane0" / "ops.npy", allow_pickle=True).item()
        cls.first_channel_mean_image = options["meanImg"]
        cls.second_channel_mean_image = options["meanImg_chan2"]

        cls.image_size = (128, 128)
        cls.num_rois = 15

        pixel_masks = cls.extractor.get_roi_pixel_masks()
        image_masks = np.zeros(shape=(*cls.image_size, cls.num_rois))
        for roi_ind, pixel_mask in enumerate(pixel_masks):
            for y, x, wt in pixel_mask:
                image_masks[int(y), int(x), roi_ind] = wt
        cls.image_masks = image_masks

    @classmethod
    def tearDownClass(cls):
        # remove the temporary directory and its contents
        shutil.rmtree(cls.test_dir)

    def test_available_channel_names(self):
        self.assertEqual(
            Suite2pSegmentationExtractor.get_available_channels(folder_path=self.folder_path), self.channel_names
        )

    def test_plane_names(self):
        self.assertEqual(
            Suite2pSegmentationExtractor.get_available_planes(folder_path=self.folder_path), self.plane_names
        )

    def test_multi_channel_warns(self):
        exc_msg = "More than one channel is detected! Please specify which channel you wish to load with the `channel_name` argument. To see what channels are available, call `Suite2pSegmentationExtractor.get_available_channels(folder_path=...)`."
        with self.assertWarnsWith(warn_type=UserWarning, exc_msg=exc_msg):
            Suite2pSegmentationExtractor(folder_path=self.folder_path)

    def test_multi_plane_warns(self):
        exc_msg = "More than one plane is detected! Please specify which plane you wish to load with the `plane_name` argument. To see what planes are available, call `Suite2pSegmentationExtractor.get_available_planes(folder_path=...)`."
        with self.assertWarnsWith(warn_type=UserWarning, exc_msg=exc_msg):
            Suite2pSegmentationExtractor(folder_path=self.folder_path, channel_name="chan2")

    def test_incorrect_plane_name_raises(self):
        exc_msg = "The selected plane 'plane2' is not a valid plane name. To see what planes are available, call `Suite2pSegmentationExtractor.get_available_planes(folder_path=...)`."
        with self.assertRaisesWith(exc_type=ValueError, exc_msg=exc_msg):
            Suite2pSegmentationExtractor(folder_path=self.folder_path, plane_name="plane2")

    def test_incorrect_channel_name_raises(self):
        exc_msg = "The selected channel 'test' is not a valid channel name. To see what channels are available, call `Suite2pSegmentationExtractor.get_available_channels(folder_path=...)`."
        with self.assertRaisesWith(exc_type=ValueError, exc_msg=exc_msg):
            Suite2pSegmentationExtractor(folder_path=self.folder_path, channel_name="test")

    def test_incomplete_extractor_load(self):
        """Check extractor can be initialized when not all traces are available."""
        # temporary directory for testing assertion when some of the files are missing
        files_to_copy = ["stat.npy", "ops.npy", "iscell.npy", "Fneu.npy"]
        (self.test_dir / "plane0").mkdir(exist_ok=True)
        [
            shutil.copy(Path(self.folder_path) / "plane0" / file, self.test_dir / "plane0" / file)
            for file in files_to_copy
        ]

        extractor = Suite2pSegmentationExtractor(folder_path=self.test_dir)
        traces_dict = extractor.get_traces_dict()
        self.assertEqual(traces_dict["raw"], None)
        self.assertEqual(traces_dict["dff"], None)
        self.assertEqual(traces_dict["deconvolved"], None)

    def test_image_size(self):
        self.assertEqual(self.extractor.get_frame_shape(), self.image_size)

    def test_num_frames(self):
        self.assertEqual(self.extractor.get_num_samples(), 250)

    def test_sampling_frequency(self):
        self.assertEqual(self.extractor.get_sampling_frequency(), 10.0)

    def test_optical_channel_names(self):
        self.assertEqual(self.extractor.get_channel_names(), ["Chan1"])

    def test_num_channels(self):
        self.assertEqual(self.extractor.get_num_channels(), 1)

    def test_num_rois(self):
        self.assertEqual(self.extractor.get_num_rois(), self.num_rois)

    def test_extractor_first_channel_raw_traces(self):
        assert_array_equal(self.extractor.get_traces(name="raw"), self.first_channel_raw_traces)

    def test_extractor_second_channel(self):
        extractor = Suite2pSegmentationExtractor(folder_path=self.folder_path, channel_name="chan2")
        self.assertEqual(extractor.get_channel_names(), ["Chan2"])
        traces = extractor.get_traces_dict()
        self.assertEqual(traces["deconvolved"], None)
        assert_array_equal(traces["raw"], self.second_channel_raw_traces)

    def test_extractor_image_masks(self):
        """Test that the image masks are correctly extracted."""
        assert_array_equal(self.extractor.get_roi_image_masks(), self.image_masks)

    def test_extractor_image_masks_selected_rois(self):
        """Test that the image masks are correctly extracted for a subset of ROIs."""
        selected_roi_ids = self.extractor.get_roi_ids()[:5]
        roi_indices = list(range(5))
        assert_array_equal(
            self.extractor.get_roi_image_masks(roi_ids=selected_roi_ids), self.image_masks[..., roi_indices]
        )

    def test_first_channel_mean_image(self):
        """Test that the mean image is correctly loaded from the extractor."""
        images_dict = self.extractor.get_images_dict()
        assert_array_equal(images_dict["mean"], self.first_channel_mean_image)

    def test_second_channel_mean_image(self):
        """Test that the mean image for the second channel is correctly loaded from the extractor."""
        extractor = Suite2pSegmentationExtractor(folder_path=self.folder_path, channel_name="chan2")
        images_dict = extractor.get_images_dict()
        assert_array_equal(images_dict["mean"], self.second_channel_mean_image)
