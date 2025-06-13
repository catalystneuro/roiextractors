import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal

from roiextractors import MiniscopeImagingExtractor
from roiextractors.extractors.miniscopeimagingextractor import (
    MiniscopeMultiRecordingImagingExtractor,
    get_miniscope_files_from_multi_recordings_subfolders,
    get_miniscope_files_from_direct_folder,
    validate_miniscope_files,
    load_miniscope_config,
)
from .setup_paths import OPHYS_DATA_PATH


class TestMiniscopeExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder_path = Path(OPHYS_DATA_PATH / "imaging_datasets" / "Miniscope" / "C6-J588_Disc5")
        cls.relative_file_paths = [
            "15_03_28/Miniscope/0.avi",
            "15_06_28/Miniscope/0.avi",
            "15_07_58/Miniscope/0.avi",
        ]
        cls.image_size = (480, 752)
        cls.num_frames = 15

        # Get absolute file paths and configuration file using utility function
        cls.file_paths, cls.configuration_file_path = get_miniscope_files_from_multi_recordings_subfolders(
            cls.folder_path
        )

        # temporary directory for testing assertion when json file is missing
        test_dir = tempfile.mkdtemp()
        cls.test_dir = Path(test_dir)
        os.makedirs(cls.test_dir / cls.relative_file_paths[0])
        shutil.copy(cls.folder_path / cls.relative_file_paths[0], cls.test_dir / cls.relative_file_paths[0])

        # Create extractors using both old and new interfaces
        cls.multi_recording_extractor = MiniscopeMultiRecordingImagingExtractor(cls.folder_path)
        cls.extractor = MiniscopeImagingExtractor(cls.file_paths, cls.configuration_file_path)

        cls.video = cls._get_test_video()

    @classmethod
    def _get_test_video(cls):
        video = np.empty((cls.num_frames, *cls.image_size))

        for file_num, file_path in enumerate(cls.relative_file_paths):
            cap = cv2.VideoCapture(str(cls.folder_path / file_path))

            for frame_num in range(5):
                ret, frame = cap.read()
                grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                video[(file_num * 5) + frame_num] = grayscale_frame

            cap.release()
        return video

    def test_avi_files_are_missing_assertion(self):
        with self.assertRaises(AssertionError):
            get_miniscope_files_from_multi_recordings_subfolders("test")

    def test_json_files_are_missing_assertion(self):
        with self.assertRaises(AssertionError):
            get_miniscope_files_from_multi_recordings_subfolders(self.test_dir)

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
        video = self.extractor.get_video()
        assert_array_equal(video, self.video)
        self.assertEqual(video.shape, (self.num_frames, *self.image_size))
        self.assertEqual(video.dtype, self.extractor.get_dtype())

    # Tests for the new MiniscopeMultiRecordingImagingExtractor
    def test_multi_recording_extractor_num_samples(self):
        self.assertEqual(self.multi_recording_extractor.get_num_samples(), self.num_frames)

    def test_multi_recording_extractor_image_shape(self):
        self.assertEqual(self.multi_recording_extractor.get_image_shape(), self.image_size)

    def test_multi_recording_extractor_num_channels(self):
        self.assertEqual(self.multi_recording_extractor.get_num_channels(), 1)

    def test_multi_recording_extractor_channel_names(self):
        self.assertEqual(self.multi_recording_extractor.get_channel_names(), ["OpticalChannel"])

    def test_multi_recording_extractor_sampling_frequency(self):
        self.assertEqual(self.multi_recording_extractor.get_sampling_frequency(), 15.0)

    def test_multi_recording_extractor_dtype(self):
        self.assertEqual(self.multi_recording_extractor.get_dtype(), np.uint8)

    def test_multi_recording_extractor_get_series(self):
        series = self.multi_recording_extractor.get_series()
        assert_array_equal(series, self.video)
        self.assertEqual(series.shape, (self.num_frames, *self.image_size))
        self.assertEqual(series.dtype, self.multi_recording_extractor.get_dtype())

    def test_multi_recording_extractor_get_consecutive_series(self):
        assert_array_equal(self.multi_recording_extractor.get_series(start_sample=4, end_sample=11), self.video[4:11])

    # Tests for utility functions
    def test_validate_miniscope_files_valid(self):
        # Should not raise any exception for valid files
        validate_miniscope_files(self.file_paths, self.configuration_file_path)

    def test_validate_miniscope_files_empty_list(self):
        with self.assertRaises(ValueError):
            validate_miniscope_files([], self.configuration_file_path)

    def test_validate_miniscope_files_missing_config(self):
        with self.assertRaises(FileNotFoundError):
            validate_miniscope_files(self.file_paths, "nonexistent.json")

    def test_validate_miniscope_files_invalid_config_extension(self):
        with self.assertRaises(FileNotFoundError):
            validate_miniscope_files(self.file_paths, "config.txt")

    def test_validate_miniscope_files_invalid_video_extension(self):
        invalid_files = [Path("video.mp4")]
        with self.assertRaises(FileNotFoundError):
            validate_miniscope_files(invalid_files, self.configuration_file_path)

    def test_load_miniscope_config(self):
        config = load_miniscope_config(self.configuration_file_path)
        self.assertIsInstance(config, dict)
        self.assertIn("frameRate", config)

    def test_load_miniscope_config_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_miniscope_config("nonexistent.json")

    def test_get_miniscope_files_from_multi_recordings_subfolders(self):
        file_paths, config_path = get_miniscope_files_from_multi_recordings_subfolders(self.folder_path)
        self.assertEqual(len(file_paths), 3)
        self.assertTrue(all(str(fp).endswith(".avi") for fp in file_paths))
        self.assertTrue(str(config_path).endswith("metaData.json"))

    def test_multi_recording_extractor_file_validation_invalid_avi(self):
        # Test with non-existent file
        invalid_files = [Path("/nonexistent/file.avi")]
        with self.assertRaises(FileNotFoundError):
            validate_miniscope_files(invalid_files, self.configuration_file_path)

    def test_multi_recording_extractor_equivalence_with_old(self):
        # Test that both extractors produce the same results
        old_video = self.extractor.get_video()
        new_series = self.multi_recording_extractor.get_series()
        assert_array_equal(old_video, new_series)

    def test_direct_folder_structure_utility(self):
        # Test the direct folder utility function with a mock structure
        # This test would need actual data in direct folder format to work properly
        # For now, we just test that the function exists and can be called
        try:
            get_miniscope_files_from_direct_folder(self.folder_path)
        except AssertionError:
            # Expected since our test data is not in direct folder format
            pass

    def test_private_miniscopeextractors_get_series(self):
        num_frames_per_extractor = 5
        for num_extractor, sub_extractor in enumerate(self.multi_recording_extractor._imaging_extractors):
            start = num_extractor * num_frames_per_extractor
            video_range = np.arange(start, start + num_frames_per_extractor)
            assert_array_equal(sub_extractor.get_series(), self.video[video_range])

    def test_deprecation_warnings(self):
        # Test that deprecation warnings are raised for old methods
        with self.assertWarns(DeprecationWarning):
            MiniscopeImagingExtractor(self.file_paths, self.configuration_file_path)

        # Test deprecated methods on individual extractor
        single_extractor = self.multi_recording_extractor._imaging_extractors[0]
        with self.assertWarns(DeprecationWarning):
            single_extractor.get_num_frames()

        with self.assertWarns(DeprecationWarning):
            single_extractor.get_image_size()

        with self.assertWarns(DeprecationWarning):
            single_extractor.get_video()
