import shutil
import tempfile
from pathlib import Path
from warnings import warn

import numpy as np
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal
from tifffile import tifffile

from roiextractors import MicroManagerTiffImagingExtractor
from tests.setup_paths import OPHYS_DATA_PATH


class TestMicroManagerTiffExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        folder_path = str(OPHYS_DATA_PATH / "imaging_datasets" / "MicroManagerTif" / "TS12_20220407_20hz_noteasy_1")
        cls.folder_path = Path(folder_path)
        file_paths = [
            "TS12_20220407_20hz_noteasy_1_MMStack_Default.ome.tif",
            "TS12_20220407_20hz_noteasy_1_MMStack_Default_1.ome.tif",
            "TS12_20220407_20hz_noteasy_1_MMStack_Default_2.ome.tif",
        ]
        cls.file_paths = file_paths
        extractor = MicroManagerTiffImagingExtractor(folder_path=folder_path)
        cls.extractor = extractor
        cls.video = cls._get_test_video()

        # temporary directory for testing assertion when xml file is missing
        test_dir = tempfile.mkdtemp()
        cls.test_dir = test_dir
        shutil.copy(Path(folder_path) / file_paths[0], Path(test_dir) / file_paths[0])

    @classmethod
    def _get_test_video(cls):
        frames = []
        for file_path in cls.file_paths:
            with tifffile.TiffFile(str(cls.folder_path / file_path)) as tif:
                frames.append(tif.asarray(key=range(5)))
        return np.concatenate(frames, axis=0)

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(cls.test_dir)
        except PermissionError:  # Windows
            warn(f"Unable to cleanup testing data at {cls.test_dir}! Please remove it manually.")

    def test_tif_files_are_missing_assertion(self):
        folder_path = "not a tiff path"
        exc_msg = f"The TIF image files are missing from '{folder_path}'."
        with self.assertRaisesWith(AssertionError, exc_msg=exc_msg):
            MicroManagerTiffImagingExtractor(folder_path=folder_path)

    def test_json_file_is_missing_assertion(self):
        folder_path = self.test_dir
        exc_msg = f"The 'DisplaySettings.json' file is not found at '{folder_path}'."
        with self.assertRaisesWith(AssertionError, exc_msg=exc_msg):
            MicroManagerTiffImagingExtractor(folder_path=folder_path)

    def test_list_of_missing_tif_files_assertion(self):
        shutil.copy(Path(self.folder_path) / "DisplaySettings.json", Path(self.test_dir) / "DisplaySettings.json")
        exc_msg = f"Some of the TIF image files at '{self.test_dir}' are missing. The list of files that are missing: {self.file_paths[1:]}"
        with self.assertRaisesWith(AssertionError, exc_msg=exc_msg):
            MicroManagerTiffImagingExtractor(folder_path=self.test_dir)

    def test_micromanagertiffextractor_image_size(self):
        self.assertEqual(self.extractor.get_image_size(), (1024, 1024))

    def test_micromanagertiffextractor_num_frames(self):
        self.assertEqual(self.extractor.get_num_frames(), 15)

    def test_micromanagertiffextractor_sampling_frequency(self):
        self.assertEqual(self.extractor.get_sampling_frequency(), 20.0)

    def test_micromanagertiffextractor_channel_names(self):
        self.assertEqual(self.extractor.get_channel_names(), ["Default"])

    def test_micromanagertiffextractor_num_channels(self):
        self.assertEqual(self.extractor.get_num_channels(), 1)

    def test_micromanagertiffextractor_dtype(self):
        self.assertEqual(self.extractor.get_dtype(), np.uint16)

    def test_micromanagertiffextractor_get_video(self):
        assert_array_equal(self.extractor.get_video(), self.video)

    def test_micromanagertiffextractor_get_single_frame(self):
        assert_array_equal(self.extractor.get_frames(frames=[0]), self.video[0][np.newaxis, ...])

    def test_private_micromanagertiffextractor_num_frames(self):
        for sub_extractor in self.extractor._imaging_extractors:
            self.assertEqual(sub_extractor.get_num_frames(), 5)

    def test_private_micromanagertiffextractor_num_channels(self):
        self.assertEqual(self.extractor._imaging_extractors[0].get_num_channels(), 1)

    def test_private_micromanagertiffextractor_sampling_frequency(self):
        sub_extractor = self.extractor._imaging_extractors[0]
        exc_msg = f"The {sub_extractor.extractor_name}Extractor does not support retrieving the imaging rate."
        with self.assertRaisesWith(NotImplementedError, exc_msg=exc_msg):
            self.extractor._imaging_extractors[0].get_sampling_frequency()

    def test_private_micromanagertiffextractor_channel_names(self):
        sub_extractor = self.extractor._imaging_extractors[0]
        exc_msg = f"The {sub_extractor.extractor_name}Extractor does not support retrieving the name of the channels."
        with self.assertRaisesWith(NotImplementedError, exc_msg=exc_msg):
            self.extractor._imaging_extractors[0].get_channel_names()

    def test_private_micromanagertiffextractor_dtype(self):
        """Test that the dtype of the private extractor is the same as the dtype of the main extractor."""
        sub_extractor = self.extractor._imaging_extractors[0]
        self.assertEqual(self.extractor.get_dtype(), sub_extractor.get_dtype())

    def test_private_micromanagertiffextractor_get_video(self):
        """Test that the dtype of the video is uint16."""
        sub_extractor = self.extractor._imaging_extractors[0]
        expected_dtype = np.uint16
        sub_extractor_video_dtype = sub_extractor.get_video().dtype
        self.assertEqual(sub_extractor_video_dtype, expected_dtype)
