import datetime
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from roiextractors import (
    MiniscopeImagingExtractor,
    MiniscopeMultiRecordingImagingExtractor,
)
from roiextractors.extractors.miniscopeimagingextractor.miniscope_utils import (
    get_recording_start_time,
    get_recording_start_times_for_multi_recordings,
)
from roiextractors.extractors.miniscopeimagingextractor.miniscopeimagingextractor import (
    read_timestamps_from_csv_file,
)

from .setup_paths import OPHYS_DATA_PATH


class TestMiniscopeImagingExtractor:
    """Test class for MiniscopeImagingExtractor using Ca_EEG3-4_FC data."""

    @classmethod
    def setup_class(cls):
        cls.folder_path = Path(OPHYS_DATA_PATH / "imaging_datasets" / "Miniscope" / "Ca_EEG3-4_FC")
        cls.direct_folder_path = cls.folder_path / "2022_09_19" / "09_18_41" / "miniscope"
        cls.relative_file_paths = [
            "0.avi",
            "1.avi",
            "2.avi",
        ]

        # Get files from direct folder structure (using lowercase "miniscope")
        cls.file_paths, cls.configuration_file_path, cls.timestamp_path = (
            MiniscopeImagingExtractor._get_miniscope_files_from_direct_folder(cls.direct_folder_path)
        )

        # Create extractor
        cls.extractor = MiniscopeImagingExtractor(
            file_paths=cls.file_paths,
            configuration_file_path=cls.configuration_file_path,
            timestamps_path=cls.timestamp_path,
        )

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
        assert self.extractor.get_num_samples() == self.num_samples

    def test_miniscopeextractor_image_shape(self):
        """Test that the extractor returns the correct image shape."""
        assert len(self.extractor.get_image_shape()) == 2
        assert self.extractor.get_image_shape()[0] == self.image_shape[0]
        assert self.extractor.get_image_shape()[1] == self.image_shape[1]

    def test_miniscopeextractor_channel_names(self):
        """Test that the extractor returns the correct channel names."""
        assert self.extractor.get_channel_names() == ["OpticalChannel"]

    def test_miniscopeextractor_sampling_frequency(self):
        """Test that the extractor returns the correct sampling frequency."""
        assert isinstance(self.extractor.get_sampling_frequency(), float)
        assert self.extractor.get_sampling_frequency() > 0

    # We suppress the warning here because this test is specifically for error handling
    # So the warning is non-informative for this test case
    @pytest.mark.filterwarnings("ignore:`timeStamps\\.csv` file not found:UserWarning")
    def test_frame_rate_extraction_error(self, tmp_path):
        """Test error when frame rate cannot be extracted from configuration."""
        temp_frame_rate_dir = tmp_path / "frame_rate_test"
        temp_frame_rate_dir.mkdir()

        test_avi_file = temp_frame_rate_dir / "0.avi"
        test_avi_file.touch()

        invalid_config = temp_frame_rate_dir / "metaData.json"
        config_data = {"frameRate": "invalid_frame_rate_format", "deviceName": "Miniscope"}
        with open(invalid_config, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(
            ValueError, match="Could not extract frame rate from configuration: invalid_frame_rate_format"
        ):
            MiniscopeImagingExtractor(file_paths=[test_avi_file], configuration_file_path=invalid_config)

    def test_miniscopeextractor_dtype(self):
        """Test that the extractor returns the correct data type."""
        assert self.extractor.get_dtype() == np.uint8

    def test_miniscopeextractor_get_series(self):
        """Test that the extractor can retrieve video series."""
        series = self.extractor.get_series()
        assert series.shape == (self.num_samples, *self.image_shape)
        assert series.dtype == self.extractor.get_dtype()

    def test_miniscopeextractor_get_series_slice(self):
        """Test that the extractor can retrieve a slice of video series."""
        start_sample = 5
        end_sample = 10
        series_slice = self.extractor.get_series(start_sample=start_sample, end_sample=end_sample)
        assert series_slice.shape == (end_sample - start_sample, *self.image_shape)

    def test_private_miniscopeextractor_properties(self):
        """Test properties of individual sub-extractors."""
        for sub_extractor in self.extractor._imaging_extractors:
            assert sub_extractor.get_num_samples() > 0
            assert sub_extractor.get_sampling_frequency() == self.extractor.get_sampling_frequency()

    def test_miniscopeextractor_timestamps(self):
        """Test that the extractor can retrieve native timestamps."""
        timestamps = self.extractor.get_native_timestamps()
        assert isinstance(timestamps, np.ndarray)
        assert len(timestamps) == self.num_samples
        assert all(isinstance(ts, (int, float, np.number)) for ts in timestamps)


class TestMiniscopeMultiRecordingImagingExtractor:
    """Test class for MiniscopeMultiRecordingImagingExtractor using C6-J588_Disc5 data."""

    @classmethod
    def setup_class(cls):
        cls.folder_path = Path(OPHYS_DATA_PATH / "imaging_datasets" / "Miniscope" / "C6-J588_Disc5")
        cls.relative_file_paths = [
            "15_03_28/Miniscope/0.avi",
            "15_06_28/Miniscope/0.avi",
            "15_07_58/Miniscope/0.avi",
        ]
        cls.image_shape = (480, 752)
        cls.num_samples = 15

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
        with pytest.raises(AssertionError, match="No Miniscope .avi files found at"):
            MiniscopeMultiRecordingImagingExtractor._get_miniscope_files_from_multi_recordings_subfolders("test")

    def test_json_files_are_missing_assertion(self, tmp_path):
        first_relative_path = Path(self.relative_file_paths[0])
        destination = tmp_path / first_relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.folder_path / self.relative_file_paths[0], destination)

        with pytest.raises(AssertionError, match="No Miniscope configuration files found at"):
            MiniscopeMultiRecordingImagingExtractor._get_miniscope_files_from_multi_recordings_subfolders(tmp_path)

    # Tests for the MiniscopeMultiRecordingImagingExtractor
    def test_multi_recording_extractor_num_samples(self):
        assert self.multi_recording_extractor.get_num_samples() == self.num_samples

    def test_multi_recording_extractor_image_shape(self):
        assert self.multi_recording_extractor.get_image_shape() == self.image_shape

    def test_multi_recording_extractor_channel_names(self):
        assert self.multi_recording_extractor.get_channel_names() == ["OpticalChannel"]

    def test_multi_recording_extractor_sampling_frequency(self):
        assert self.multi_recording_extractor.get_sampling_frequency() == 15.0

    def test_multi_recording_extractor_dtype(self):
        assert self.multi_recording_extractor.get_dtype() == np.uint8

    def test_multi_recording_extractor_get_series(self):
        series = self.multi_recording_extractor.get_series()
        assert_array_equal(series, self.video)
        assert series.shape == (self.num_samples, *self.image_shape)
        assert series.dtype == self.multi_recording_extractor.get_dtype()

    def test_multi_recording_extractor_get_consecutive_series(self):
        assert_array_equal(self.multi_recording_extractor.get_series(start_sample=4, end_sample=11), self.video[4:11])

    def test_private_miniscopeextractors_get_series(self):
        num_samples_per_extractor = 5
        for num_extractor, sub_extractor in enumerate(self.multi_recording_extractor._imaging_extractors):
            start = num_extractor * num_samples_per_extractor
            video_range = np.arange(start, start + num_samples_per_extractor)
            assert_array_equal(sub_extractor.get_series(), self.video[video_range])

    def test_multi_recording_extractor_timestamps(self):
        timestamps = self.multi_recording_extractor.get_native_timestamps()
        assert isinstance(timestamps, np.ndarray)
        assert len(timestamps) == self.num_samples
        assert all(isinstance(ts, (int, float, np.number)) for ts in timestamps)


class TestMiniscopeUtilityFunctions:
    """Test class for all utility functions in miniscope_utils.py."""

    @classmethod
    def setup_class(cls):
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
        cls.valid_file_paths, cls.valid_config_path = (
            MiniscopeMultiRecordingImagingExtractor._get_miniscope_files_from_multi_recordings_subfolders(
                cls.multi_recording_folder
            )
        )

    @pytest.fixture(autouse=True)
    def _set_temp_dir(self, tmp_path):
        self.temp_dir = tmp_path

    def test_get_miniscope_files_from_folder_valid(self):
        """Test successful file discovery from multi-recording folder structure."""
        file_paths, config_path = (
            MiniscopeMultiRecordingImagingExtractor._get_miniscope_files_from_multi_recordings_subfolders(
                self.multi_recording_folder
            )
        )

        assert isinstance(file_paths, list)
        assert len(file_paths) > 0
        assert all(str(fp).endswith(".avi") for fp in file_paths)
        assert str(config_path).endswith("metaData.json")
        assert Path(config_path).exists()

    def test_get_miniscope_files_from_folder_no_avi(self):
        """Test assertion when no .avi files are found."""
        with pytest.raises(AssertionError, match="No .avi files found in direct folder structure at"):
            MiniscopeImagingExtractor._get_miniscope_files_from_direct_folder(self.temp_dir)

    def test_get_miniscope_files_from_folder_nonexistent(self):
        """Test assertion when folder doesn't exist."""
        with pytest.raises(AssertionError, match="No .avi files found in direct folder structure at"):
            MiniscopeImagingExtractor._get_miniscope_files_from_direct_folder(Path("nonexistent_folder"))

    def test_get_miniscope_files_from_folder_direct_structure(self):
        """Test successful file discovery from direct folder structure."""
        file_paths, config_path, timestamps_path = MiniscopeImagingExtractor._get_miniscope_files_from_direct_folder(
            self.direct_folder
        )

        assert isinstance(file_paths, list)
        assert len(file_paths) > 0
        assert all(str(fp).endswith(".avi") for fp in file_paths)
        assert str(config_path).endswith("metaData.json")
        assert Path(config_path).exists()
        assert str(timestamps_path).endswith("timeStamps.csv")
        assert Path(timestamps_path).exists()

    def test_validate_miniscope_files_valid(self):
        """Test validation with valid files."""
        # Should not raise any exception
        MiniscopeImagingExtractor.validate_miniscope_files(self.valid_file_paths, self.valid_config_path)

    def test_validate_miniscope_files_empty_list(self):
        """Test validation with empty file list."""
        with pytest.raises(ValueError):
            MiniscopeImagingExtractor.validate_miniscope_files([], self.valid_config_path)

    def test_validate_miniscope_files_missing_config(self):
        """Test validation with missing configuration file."""
        with pytest.raises(FileNotFoundError):
            MiniscopeImagingExtractor.validate_miniscope_files(self.valid_file_paths, Path("nonexistent.json"))

    def test_validate_miniscope_files_invalid_config_extension(self):
        """Test validation with invalid configuration file extension."""
        with pytest.raises(ValueError):
            # Create a temporary file with wrong extension
            temp_file = self.temp_dir / "config.txt"
            temp_file.touch()
            MiniscopeImagingExtractor.validate_miniscope_files(self.valid_file_paths, temp_file)

    def test_validate_miniscope_files_missing_video(self):
        """Test validation with missing video file."""
        invalid_files = [Path("nonexistent.avi")]
        with pytest.raises(FileNotFoundError):
            MiniscopeImagingExtractor.validate_miniscope_files(invalid_files, self.valid_config_path)

    def test_validate_miniscope_files_invalid_video_extension(self):
        """Test validation with invalid video file extension."""
        # Create a temporary file with wrong extension
        temp_file = self.temp_dir / "video.mp4"
        temp_file.touch()
        with pytest.raises(ValueError):
            MiniscopeImagingExtractor.validate_miniscope_files([temp_file], self.valid_config_path)

    def test_load_miniscope_config_valid(self):
        """Test loading valid configuration file."""
        config = MiniscopeImagingExtractor.load_miniscope_config(self.valid_config_path)

        assert isinstance(config, dict)
        assert "frameRate" in config

    def test_load_miniscope_config_missing_file(self):
        """Test loading missing configuration file."""
        with pytest.raises(FileNotFoundError):
            MiniscopeImagingExtractor.load_miniscope_config(Path("nonexistent.json"))

    def test_load_miniscope_config_invalid_json(self):
        """Test loading invalid JSON file."""
        # Create invalid JSON file
        invalid_json = self.temp_dir / "invalid.json"
        with open(invalid_json, "w") as f:
            f.write("{ invalid json")

        with pytest.raises(json.JSONDecodeError):
            MiniscopeImagingExtractor.load_miniscope_config(invalid_json)

    def test_get_recording_start_time_format_1(self):
        """Test get_recording_start_time with first JSON format (Ca_EEG2-1_FC)."""
        start_time = get_recording_start_time(self.metadata_file_1)
        expected_start_time = datetime.datetime(2021, 10, 14, 10, 11, 24, 779000)

        assert start_time == expected_start_time

    def test_get_recording_start_time_format_2(self):
        """Test get_recording_start_time with second JSON format (Ca_EEG3-4_FC)."""
        start_time = get_recording_start_time(self.metadata_file_2)
        expected_start_time = datetime.datetime(2022, 9, 19, 9, 18, 41, 7000)

        assert start_time == expected_start_time

    def test_get_recording_start_time_missing_file(self):
        """Test get_recording_start_time with missing file."""
        with pytest.raises((FileNotFoundError, OSError)):
            get_recording_start_time(Path("nonexistent.json"))

    def test_get_recording_start_time_missing_keys(self):
        """Test get_recording_start_time with missing required keys."""
        # Create JSON file missing required keys
        incomplete_json = self.temp_dir / "incomplete.json"
        with open(incomplete_json, "w") as f:
            json.dump({"year": 2021, "month": 10}, f)

        with pytest.raises(KeyError):
            get_recording_start_time(incomplete_json)

    def test_get_recording_start_times_for_multi_recordings(self):
        """Test getting start times for multiple recordings."""

        start_times = get_recording_start_times_for_multi_recordings(self.multi_recording_folder)
        assert isinstance(start_times, list)
        assert len(start_times) > 0
        assert all(isinstance(st, datetime.datetime) for st in start_times)

    def test_get_recording_start_times_for_multi_recordings_no_files(self):
        """Test getting start times when no config files exist."""
        no_metadata_dir = self.temp_dir / "no_metadata"
        no_metadata_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(AssertionError, match="No Miniscope configuration files found at"):
            get_recording_start_times_for_multi_recordings(no_metadata_dir)

    def test_read_timestamps_from_csv_file(self):
        """Test reading timestamps from CSV file."""
        # Find a timestamps CSV file in the test data
        csv_files = list(self.multi_recording_folder.glob("*/Miniscope/timeStamps.csv"))
        if csv_files:
            timestamps = read_timestamps_from_csv_file(csv_files[0])

            assert isinstance(timestamps, np.ndarray)
            assert len(timestamps) > 0
            assert all(isinstance(t, (int, float, np.number)) for t in timestamps)

    def test_get_miniscope_files_from_direct_folder_valid(self):
        """Test successful file discovery from direct folder structure."""
        file_paths, config_path, timestamps_path = MiniscopeImagingExtractor._get_miniscope_files_from_direct_folder(
            self.direct_folder
        )

        assert isinstance(file_paths, list)
        assert len(file_paths) > 0
        assert all(str(fp).endswith(".avi") for fp in file_paths)
        assert str(config_path).endswith("metaData.json")
        assert Path(config_path).exists()
        assert str(timestamps_path).endswith("timeStamps.csv")
        assert Path(timestamps_path).exists()

        # Test files are sorted naturally (0.avi, 1.avi, 2.avi, ...)
        expected_names = [f"{i}.avi" for i in range(len(file_paths))]
        actual_names = [fp.name for fp in file_paths]
        assert actual_names == expected_names

    def test_get_miniscope_files_from_direct_folder_no_avi_files(self):
        """Test assertion when no .avi files are found in direct folder."""
        with pytest.raises(AssertionError, match="No .avi files found in direct folder structure at"):
            MiniscopeImagingExtractor._get_miniscope_files_from_direct_folder(self.temp_dir)

    def test_get_miniscope_files_from_direct_folder_no_config_file(self):
        """Test assertion when no metaData.json file is found in direct folder."""
        # Create temporary directory with .avi files but no config
        temp_avi_dir = self.temp_dir / "avi_only"
        temp_avi_dir.mkdir()
        (temp_avi_dir / "0.avi").touch()
        (temp_avi_dir / "1.avi").touch()

        with pytest.raises(AssertionError, match="No configuration file found at"):
            MiniscopeImagingExtractor._get_miniscope_files_from_direct_folder(temp_avi_dir)

    def test_get_miniscope_files_from_direct_folder_invalid_avi_naming(self):
        """Test error when .avi files don't follow expected naming convention."""
        temp_bad_naming_dir = self.temp_dir / "bad_naming"
        temp_bad_naming_dir.mkdir()
        (temp_bad_naming_dir / "video1.avi").touch()  # Should be 0.avi
        (temp_bad_naming_dir / "metaData.json").touch()

        with pytest.raises(ValueError, match="Unexpected file name 'video1.avi'. Expected '0.avi'"):
            MiniscopeImagingExtractor._get_miniscope_files_from_direct_folder(temp_bad_naming_dir)

    def test_get_miniscope_files_from_direct_folder_missing_timestamps_warning(self):
        """Test warning when timeStamps.csv file is missing."""
        temp_no_timestamps_dir = self.temp_dir / "no_timestamps"
        temp_no_timestamps_dir.mkdir()
        (temp_no_timestamps_dir / "0.avi").touch()
        (temp_no_timestamps_dir / "1.avi").touch()
        (temp_no_timestamps_dir / "metaData.json").touch()

        with pytest.warns(UserWarning) as warning_context:
            file_paths, config_path, timestamps_path = (
                MiniscopeImagingExtractor._get_miniscope_files_from_direct_folder(temp_no_timestamps_dir)
            )

        assert "No timestamps file found" in str(warning_context[0].message)
        assert timestamps_path is None
        assert len(file_paths) == 2

    def test_get_miniscope_files_from_direct_folder_with_timestamps(self):
        """Test successful file discovery including timestamps file."""
        temp_with_timestamps_dir = self.temp_dir / "with_timestamps"
        temp_with_timestamps_dir.mkdir()
        (temp_with_timestamps_dir / "0.avi").touch()
        (temp_with_timestamps_dir / "1.avi").touch()
        (temp_with_timestamps_dir / "metaData.json").touch()
        (temp_with_timestamps_dir / "timeStamps.csv").touch()

        file_paths, config_path, timestamps_path = MiniscopeImagingExtractor._get_miniscope_files_from_direct_folder(
            temp_with_timestamps_dir
        )

        assert len(file_paths) == 2
        assert [fp.name for fp in file_paths] == ["0.avi", "1.avi"]
        assert config_path.name == "metaData.json"
        assert timestamps_path.name == "timeStamps.csv"

    def test_get_miniscope_files_from_direct_folder_nonexistent_path(self):
        """Test assertion when folder path doesn't exist."""
        nonexistent_path = Path("nonexistent_folder_12345")
        with pytest.raises(AssertionError, match="No .avi files found in direct folder structure at"):
            MiniscopeImagingExtractor._get_miniscope_files_from_direct_folder(nonexistent_path)
