import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal

from roiextractors import MiniscopeImagingExtractor
from .setup_paths import OPHYS_DATA_PATH


class TestMiniscopeExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        # TODO: upload test data (waiting on approval)
        cls.folder_path = Path(OPHYS_DATA_PATH / "imaging_datasets" / "Miniscope" / "C6-J588_Disc5")
        cls.file_paths = [
            "15_03_28/Miniscope/0.avi",
            "15_06_28/Miniscope/0.avi",
            "15_07_58/Miniscope/0.avi",
        ]
        cls.image_size = (480, 752)
        cls.num_frames = 15

        # temporary directory for testing assertion when json file is missing
        test_dir = tempfile.mkdtemp()
        cls.test_dir = Path(test_dir)
        os.makedirs(cls.test_dir / cls.file_paths[0])
        shutil.copy(cls.folder_path / cls.file_paths[0], cls.test_dir / cls.file_paths[0])

        cls.extractor = MiniscopeImagingExtractor(folder_path=cls.folder_path)

        cls.video = cls._get_test_video()

    @classmethod
    def _get_test_video(cls):
        video = np.empty((cls.num_frames, *cls.image_size))

        for file_num, file_path in enumerate(cls.file_paths):
            cap = cv2.VideoCapture(str(cls.folder_path / file_path))

            for frame_num in range(5):
                ret, frame = cap.read()
                grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                video[(file_num * 5) + frame_num] = grayscale_frame

            cap.release()
        return video

    def test_avi_files_are_missing_assertion(self):
        folder_path = "test"
        exc_msg = f"The Miniscope movies (.avi files) are missing from '{folder_path}'."
        with self.assertRaisesWith(AssertionError, exc_msg=exc_msg):
            MiniscopeImagingExtractor(folder_path=folder_path)

    def test_json_files_are_missing_assertion(self):
        exc_msg = f"The configuration files (metaData.json files) are missing from '{self.test_dir}'."
        with self.assertRaisesWith(AssertionError, exc_msg=exc_msg):
            MiniscopeImagingExtractor(folder_path=self.test_dir)

    def test_miniscopeextractor_num_frames(self):
        self.assertEqual(self.extractor.get_num_frames(), self.num_frames)

    def test_miniscopeextractor_image_size(self):
        self.assertEqual(self.extractor.get_image_size(), self.image_size)

    def test_miniscopeextractor_num_channels(self):
        self.assertEqual(self.extractor.get_num_channels(), 1)

    def test_miniscopeextractor_channel_names(self):
        self.assertEqual(self.extractor.get_channel_names(), ["OpticalChannel"])

    def test_miniscopeextractor_sampling_frequency(self):
        self.assertEqual(self.extractor.get_sampling_frequency(), 15.0)

    def test_miniscopeextractor_dtype(self):
        self.assertEqual(self.extractor.get_dtype(), np.uint8)

    def test_private_miniscopeextractor_num_frames(self):
        for sub_extractor in self.extractor._imaging_extractors:
            self.assertEqual(sub_extractor.get_num_frames(), 5)

    def test_private_miniscopeextractor_sampling_frequency(self):
        sub_extractor = self.extractor._imaging_extractors[0]
        self.assertEqual(sub_extractor.get_sampling_frequency(), 15.0)

    def test_private_miniscopeextractors_get_video(self):
        num_frames_per_extractor = 5
        for num_extractor, sub_extractor in enumerate(self.extractor._imaging_extractors):
            start = num_extractor * num_frames_per_extractor
            video_range = np.arange(start, start + num_frames_per_extractor)
            assert_array_equal(sub_extractor.get_video(), self.video[video_range])

    def test_miniscopeextractor_get_consecutive_frames(self):
        assert_array_equal(self.extractor.get_video(start_frame=4, end_frame=11), self.video[4:11])

    def test_miniscopeextractor_get_video(self):
        assert_array_equal(self.extractor.get_video(), self.video)
