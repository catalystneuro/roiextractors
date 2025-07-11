import datetime
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal

from roiextractors import MiniscopeImagingExtractor, MiniscopeMultiRecordingImagingExtractor
from roiextractors.extractors.miniscopeimagingextractor.miniscope_utils import (
    get_miniscope_files_from_multi_recordings_subfolders,
    get_miniscope_files_from_direct_folder,
    get_recording_start_time,
    get_recording_start_times_for_multi_recordings,
    get_timestamps_for_multi_recordings,
    load_miniscope_config,
    read_timestamps_from_csv_file,
    validate_miniscope_files,
)

from .setup_paths import OPHYS_DATA_PATH


class TestMiniscopeImagingExtractor(TestCase):
    """Test class for MiniscopeImagingExtractor using Ca_EEG3-4_FC data."""

    @classmethod
    def setUpClass(cls):
        cls.folder_path = Path(OPHYS_DATA_PATH / "imaging_datasets" / "Miniscope" / "Ca_EEG3-4_FC")
        cls.direct_folder_path = cls.folder_path / "2022_09_19" / "09_18_41" / "miniscope"
        cls.relative_file_paths = [
            "0.avi",
            "1.avi",
            "2.avi",
        ]

        # Get files from direct folder structure (using lowercase "miniscope")
        cls.file_paths, cls.configuration_file_path = get_miniscope_files_from_direct_folder(cls.direct_folder_path)

        # Create extractor
        cls.extractor = MiniscopeImagingExtractor(cls.file_paths, cls.configuration_file_path)

        # Get video properties
        cls.num_samples = 15
        cls.image_shape = (608, 608)

        # Load test video for comparison
        cls.video = cls._get_test_video()

    @classmethod
    def _get_test_video(cls):
        """Load video data for testing."""
        video = np.empty((cls.num_samples, *cls.image_shape), dtype=cls.extractor.get_dtype())

        frame_idx = 0
        for file_path in cls.file_paths:
            cap = cv2.VideoCapture(str(file_path))

            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= cls.num_samples:
                    break

                grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                video[frame_idx] = grayscale_frame
                frame_idx += 1

            cap.release()

        return video

    def test_miniscopeextractor_num_samples(self):
        """Test that the extractor returns the correct number of samples."""
        self.assertEqual(self.extractor.get_num_samples(), self.num_samples)

    def test_miniscopeextractor_image_shape(self):
        """Test that the extractor returns the correct image shape."""
        self.assertEqual(len(self.extractor.get_image_shape()), 2)
        self.assertEqual(self.extractor.get_image_shape()[0], self.image_shape[0])
        self.assertEqual(self.extractor.get_image_shape()[1], self.image_shape[1])

    def test_miniscopeextractor_num_channels(self):
        """Test that the extractor returns the correct number of channels."""
        self.assertEqual(self.extractor.get_num_channels(), 1)

    def test_miniscopeextractor_channel_names(self):
        """Test that the extractor returns the correct channel names."""
        self.assertEqual(self.extractor.get_channel_names(), ["OpticalChannel"])

    def test_miniscopeextractor_sampling_frequency(self):
        """Test that the extractor returns the correct sampling frequency."""
        self.assertIsInstance(self.extractor.get_sampling_frequency(), float)
        self.assertGreater(self.extractor.get_sampling_frequency(), 0)

    def test_miniscopeextractor_dtype(self):
        """Test that the extractor returns the correct data type."""
        self.assertEqual(self.extractor.get_dtype(), np.uint8)

    def test_miniscopeextractor_get_series(self):
        """Test that the extractor can retrieve video series."""
        series = self.extractor.get_series()
        self.assertEqual(series.shape, (self.num_samples, *self.image_shape))
        self.assertEqual(series.dtype, self.extractor.get_dtype())

    def test_miniscopeextractor_get_series_slice(self):
        """Test that the extractor can retrieve a slice of video series."""
        start_sample = 5
        end_sample = 10
        series_slice = self.extractor.get_series(start_sample=start_sample, end_sample=end_sample)
        self.assertEqual(series_slice.shape, (end_sample - start_sample, *self.image_shape))

    def test_private_miniscopeextractor_properties(self):
        """Test properties of individual sub-extractors."""
        for sub_extractor in self.extractor._imaging_extractors:
            self.assertGreater(sub_extractor.get_num_samples(), 0)
            self.assertEqual(sub_extractor.get_sampling_frequency(), self.extractor.get_sampling_frequency())


class TestMiniscopeMultiRecordingImagingExtractor(TestCase):
    """Test class for MiniscopeMultiRecordingImagingExtractor using C6-J588_Disc5 data."""

    @classmethod
    def setUpClass(cls):
        cls.folder_path = Path(OPHYS_DATA_PATH / "imaging_datasets" / "Miniscope" / "C6-J588_Disc5")
        cls.relative_file_paths = [
            "15_03_28/Miniscope/0.avi",
            "15_06_28/Miniscope/0.avi",
            "15_07_58/Miniscope/0.avi",
        ]
        cls.image_shape = (480, 752)
        cls.num_samples = 15

        # temporary directory for testing assertion when json file is missing
        test_dir = tempfile.mkdtemp()
        cls.test_dir = Path(test_dir)
        os.makedirs(cls.test_dir / cls.relative_file_paths[0])
        shutil.copy(cls.folder_path / cls.relative_file_paths[0], cls.test_dir / cls.relative_file_paths[0])

        # Create extractors using both old and new interfaces
        cls.multi_recording_extractor = MiniscopeMultiRecordingImagingExtractor(cls.folder_path)

        cls.video = cls._get_test_video()

    @classmethod
    def _get_test_video(cls):
        video = np.empty((cls.num_samples, *cls.image_shape))

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

    # Tests for the MiniscopeMultiRecordingImagingExtractor
    def test_multi_recording_extractor_num_samples(self):
        self.assertEqual(self.multi_recording_extractor.get_num_samples(), self.num_samples)

    def test_multi_recording_extractor_image_shape(self):
        self.assertEqual(self.multi_recording_extractor.get_image_shape(), self.image_shape)

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
        self.assertEqual(series.shape, (self.num_samples, *self.image_shape))
        self.assertEqual(series.dtype, self.multi_recording_extractor.get_dtype())

    def test_multi_recording_extractor_get_consecutive_series(self):
        assert_array_equal(self.multi_recording_extractor.get_series(start_sample=4, end_sample=11), self.video[4:11])

    def test_private_miniscopeextractors_get_series(self):
        num_samples_per_extractor = 5
        for num_extractor, sub_extractor in enumerate(self.multi_recording_extractor._imaging_extractors):
            start = num_extractor * num_samples_per_extractor
            video_range = np.arange(start, start + num_samples_per_extractor)
            assert_array_equal(sub_extractor.get_series(), self.video[video_range])


class TestMiniscopeUtilityFunctions(TestCase):
    """Test class for all utility functions in miniscope_utils.py."""

    @classmethod
    def setUpClass(cls):
        # Set up paths for different test scenarios
        cls.multi_recording_folder = Path(OPHYS_DATA_PATH / "imaging_datasets" / "Miniscope" / "C6-J588_Disc5")
        cls.direct_folder = Path(
            OPHYS_DATA_PATH
            / "imaging_datasets"
            / "Miniscope"
            / "Ca_EEG3-4_FC"
            / "2022_09_19"
            / "09_18_41"
            / "miniscope"
        )

        # Metadata files for testing get_recording_start_time
        cls.metadata_file_1 = Path(
            OPHYS_DATA_PATH
            / "imaging_datasets"
            / "Miniscope"
            / "Ca_EEG2-1_FC"
            / "2021_10_14"
            / "10_11_24"
            / "metaData.json"
        )
        cls.metadata_file_2 = Path(
            OPHYS_DATA_PATH
            / "imaging_datasets"
            / "Miniscope"
            / "Ca_EEG3-4_FC"
            / "2022_09_19"
            / "09_18_41"
            / "metaData.json"
        )

        # Get valid file paths for testing
        cls.valid_file_paths, cls.valid_config_path = get_miniscope_files_from_multi_recordings_subfolders(
            cls.multi_recording_folder
        )

        # Create temporary directory for negative tests
        cls.temp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary directory
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_get_miniscope_files_from_folder_valid(self):
        """Test successful file discovery from multi-recording folder structure."""
        file_paths, config_path = get_miniscope_files_from_multi_recordings_subfolders(self.multi_recording_folder)

        self.assertIsInstance(file_paths, list)
        self.assertGreater(len(file_paths), 0)
        self.assertTrue(all(str(fp).endswith(".avi") for fp in file_paths))
        self.assertTrue(str(config_path).endswith("metaData.json"))
        self.assertTrue(Path(config_path).exists())

    def test_get_miniscope_files_from_folder_no_avi(self):
        """Test assertion when no .avi files are found."""
        with self.assertRaises(AssertionError):
            get_miniscope_files_from_direct_folder(self.temp_dir)

    def test_get_miniscope_files_from_folder_nonexistent(self):
        """Test assertion when folder doesn't exist."""
        with self.assertRaises(AssertionError):
            get_miniscope_files_from_direct_folder(Path("nonexistent_folder"))

    def test_get_miniscope_files_from_folder_direct_structure(self):
        """Test successful file discovery from direct folder structure."""
        file_paths, config_path = get_miniscope_files_from_direct_folder(self.direct_folder)

        self.assertIsInstance(file_paths, list)
        self.assertGreater(len(file_paths), 0)
        self.assertTrue(all(str(fp).endswith(".avi") for fp in file_paths))
        self.assertTrue(str(config_path).endswith("metaData.json"))
        self.assertTrue(Path(config_path).exists())

    def test_validate_miniscope_files_valid(self):
        """Test validation with valid files."""
        # Should not raise any exception
        validate_miniscope_files(self.valid_file_paths, self.valid_config_path)

    def test_validate_miniscope_files_empty_list(self):
        """Test validation with empty file list."""
        with self.assertRaises(ValueError):
            validate_miniscope_files([], self.valid_config_path)

    def test_validate_miniscope_files_missing_config(self):
        """Test validation with missing configuration file."""
        with self.assertRaises(FileNotFoundError):
            validate_miniscope_files(self.valid_file_paths, Path("nonexistent.json"))

    def test_validate_miniscope_files_invalid_config_extension(self):
        """Test validation with invalid configuration file extension."""
        with self.assertRaises(ValueError):
            # Create a temporary file with wrong extension
            temp_file = self.temp_dir / "config.txt"
            temp_file.touch()
            validate_miniscope_files(self.valid_file_paths, temp_file)

    def test_validate_miniscope_files_missing_video(self):
        """Test validation with missing video file."""
        invalid_files = [Path("nonexistent.avi")]
        with self.assertRaises(FileNotFoundError):
            validate_miniscope_files(invalid_files, self.valid_config_path)

    def test_validate_miniscope_files_invalid_video_extension(self):
        """Test validation with invalid video file extension."""
        # Create a temporary file with wrong extension
        temp_file = self.temp_dir / "video.mp4"
        temp_file.touch()
        with self.assertRaises(ValueError):
            validate_miniscope_files([temp_file], self.valid_config_path)

    def test_load_miniscope_config_valid(self):
        """Test loading valid configuration file."""
        config = load_miniscope_config(self.valid_config_path)

        self.assertIsInstance(config, dict)
        self.assertIn("frameRate", config)

    def test_load_miniscope_config_missing_file(self):
        """Test loading missing configuration file."""
        with self.assertRaises(FileNotFoundError):
            load_miniscope_config(Path("nonexistent.json"))

    def test_load_miniscope_config_invalid_json(self):
        """Test loading invalid JSON file."""
        # Create invalid JSON file
        invalid_json = self.temp_dir / "invalid.json"
        with open(invalid_json, "w") as f:
            f.write("{ invalid json")

        with self.assertRaises(json.JSONDecodeError):
            load_miniscope_config(invalid_json)

    def test_get_recording_start_time_format_1(self):
        """Test get_recording_start_time with first JSON format (Ca_EEG2-1_FC)."""
        start_time = get_recording_start_time(self.metadata_file_1)

        self.assertIsInstance(start_time, datetime.datetime)
        self.assertEqual(start_time.year, 2021)
        self.assertEqual(start_time.month, 10)
        self.assertEqual(start_time.day, 14)
        self.assertEqual(start_time.hour, 10)
        self.assertEqual(start_time.minute, 11)
        self.assertEqual(start_time.second, 24)
        self.assertEqual(start_time.microsecond, 779000)  # 779 ms converted to microseconds

    def test_get_recording_start_time_format_2(self):
        """Test get_recording_start_time with second JSON format (Ca_EEG3-4_FC)."""
        start_time = get_recording_start_time(self.metadata_file_2)

        self.assertIsInstance(start_time, datetime.datetime)
        self.assertEqual(start_time.year, 2022)
        self.assertEqual(start_time.month, 9)
        self.assertEqual(start_time.day, 19)
        self.assertEqual(start_time.hour, 9)
        self.assertEqual(start_time.minute, 18)
        self.assertEqual(start_time.second, 41)
        self.assertEqual(start_time.microsecond, 7000)  # 7 ms converted to microseconds

    def test_get_recording_start_time_missing_file(self):
        """Test get_recording_start_time with missing file."""
        with self.assertRaises((FileNotFoundError, OSError)):
            get_recording_start_time(Path("nonexistent.json"))

    def test_get_recording_start_time_missing_keys(self):
        """Test get_recording_start_time with missing required keys."""
        # Create JSON file missing required keys
        incomplete_json = self.temp_dir / "incomplete.json"
        with open(incomplete_json, "w") as f:
            json.dump({"year": 2021, "month": 10}, f)

        with self.assertRaises(KeyError):
            get_recording_start_time(incomplete_json)

    def test_get_recording_start_times_for_multi_recordings(self):
        """Test getting start times for multiple recordings."""

        start_times = get_recording_start_times_for_multi_recordings(self.multi_recording_folder)
        self.assertIsInstance(start_times, list)
        self.assertGreater(len(start_times), 0)
        self.assertTrue(all(isinstance(st, datetime.datetime) for st in start_times))

    def test_get_recording_start_times_for_multi_recordings_no_files(self):
        """Test getting start times when no config files exist."""
        with self.assertRaises(AssertionError):
            get_recording_start_times_for_multi_recordings(self.temp_dir)

    def test_read_timestamps_from_csv_file(self):
        """Test reading timestamps from CSV file."""
        # Find a timestamps CSV file in the test data
        csv_files = list(self.multi_recording_folder.glob("*/Miniscope/timeStamps.csv"))
        if csv_files:
            timestamps = read_timestamps_from_csv_file(csv_files[0])

            self.assertIsInstance(timestamps, np.ndarray)
            self.assertGreater(len(timestamps), 0)
            self.assertTrue(all(isinstance(t, (int, float, np.number)) for t in timestamps))

    def test_get_timestamps_for_multi_recordings(self):
        """Test getting timestamps for multiple recordings."""
        timestamps = get_timestamps_for_multi_recordings(self.multi_recording_folder)
        self.assertIsInstance(timestamps, list)
        self.assertGreater(len(timestamps), 0)

    def test_get_timestamps_for_multi_recordings_no_files(self):
        """Test getting timestamps when no CSV files exist."""
        with self.assertRaises(AssertionError):
            get_timestamps_for_multi_recordings(self.temp_dir)
