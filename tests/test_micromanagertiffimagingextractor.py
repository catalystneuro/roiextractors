import shutil
import tempfile
from pathlib import Path

import numpy as np
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal
from tifffile import tifffile

from roiextractors import MicroManagerTiffImagingExtractor
from tests.setup_paths import OPHYS_DATA_PATH


class TestMicroManagerTiffExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        folder_path = str(
            OPHYS_DATA_PATH / "imaging_datasets" / "MicroManagerTif" / "TS12_20220407_20hz_noteasy_1"
        )
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
        pass
        # remove the temporary directory and its contents
        #shutil.rmtree(cls.test_dir)

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

    def test_micromanagertiffextractor_image_size(self):
        self.assertEqual(self.extractor.get_image_size(), (1024, 1024))

    def test_brukertiffextractor_num_frames(self):
        self.assertEqual(self.extractor.get_num_frames(), 15)

    def test_brukertiffextractor_sampling_frequency(self):
        self.assertEqual(self.extractor.get_sampling_frequency(), 20.0)

    def test_brukertiffextractor_channel_names(self):
        self.assertEqual(self.extractor.get_channel_names(), ["Default"])

    def test_brukertiffextractor_num_channels(self):
        self.assertEqual(self.extractor.get_num_channels(), 1)

    def test_brukertiffextractor_dtype(self):
        self.assertEqual(self.extractor.get_dtype(), np.uint16)

    def test_brukertiffextractor_get_video(self):
        assert_array_equal(self.extractor.get_video(), self.video)

    def test_brukertiffextractor_get_single_frame(self):
        assert_array_equal(self.extractor.get_frames(frame_idxs=[0]),
                           self.video[0][np.newaxis, ...])
