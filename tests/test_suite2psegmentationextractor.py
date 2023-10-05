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

    @classmethod
    def tearDownClass(cls):
        # remove the temporary directory and its contents
        shutil.rmtree(cls.test_dir)

    def test_channel_names(self):
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
        self.assertEqual(self.extractor.get_image_size(), (128, 128))

    def test_num_frames(self):
        self.assertEqual(self.extractor.get_num_frames(), 250)

    def test_sampling_frequency(self):
        self.assertEqual(self.extractor.get_sampling_frequency(), 10.0)

    def test_channel_names(self):
        self.assertEqual(self.extractor.get_channel_names(), ["Chan1"])

    def test_num_channels(self):
        self.assertEqual(self.extractor.get_num_channels(), 1)

    def test_num_rois(self):
        self.assertEqual(self.extractor.get_num_rois(), 15)

    def test_extractor_first_channel_raw_traces(self):
        assert_array_equal(self.extractor.get_traces(name="raw"), self.first_channel_raw_traces)

    def test_extractor_second_channel(self):
        extractor = Suite2pSegmentationExtractor(folder_path=self.folder_path, channel_name="chan2")
        self.assertEqual(extractor.get_channel_names(), ["Chan2"])
        traces = extractor.get_traces_dict()
        self.assertEqual(traces["deconvolved"], None)
        assert_array_equal(traces["raw"], self.second_channel_raw_traces)
