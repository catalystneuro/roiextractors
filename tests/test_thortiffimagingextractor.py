import pytest
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from numpy.testing import assert_array_equal
import tifffile

from roiextractors import ThorTiffImagingExtractor
from .setup_paths import OPHYS_DATA_PATH


# Path to the test data
TEST_DIR = OPHYS_DATA_PATH / "imaging_datasets" / "ThorlabsTiff" / "single_channel_single_plane" / "20231018-002"
FILE_PATH = TEST_DIR / "ChanA_001_001_001_001.tif"


class TestThorTiffImagingExtractor:
    """Test ThorTiffImagingExtractor."""

    @classmethod
    def setup_class(cls):
        """Set up the test."""
        if not FILE_PATH.exists():
            pytest.skip(f"Test file {FILE_PATH} not found. Skipping tests.")

        # Create the extractor
        cls.extractor = ThorTiffImagingExtractor(file_path=FILE_PATH)

        # Load the test data for comparison
        cls.test_data = tifffile.imread(FILE_PATH)

    def test_thor_tiff_extractor_frame_shape(self):
        """Test the frame shape property."""
        assert self.extractor.get_frame_shape() == (self.test_data.shape[1], self.test_data.shape[2])

    def test_thor_tiff_extractor_num_samples(self):
        """Test the number of samples property."""
        assert self.extractor.get_num_samples() == self.test_data.shape[0]

    def test_thor_tiff_extractor_sampling_frequency(self):
        """Test the sampling frequency property."""
        assert self.extractor.get_sampling_frequency() is not None
        assert isinstance(self.extractor.get_sampling_frequency(), float)

    def test_thor_tiff_extractor_channel_names(self):
        """Test the channel names property."""
        assert self.extractor.get_channel_names() is not None
        assert isinstance(self.extractor.get_channel_names(), list)

    def test_thor_tiff_extractor_dtype(self):
        """Test the data type property."""
        assert self.extractor.get_dtype() == self.test_data.dtype

    def test_thor_tiff_extractor_get_series(self):
        """Test the get_series method."""
        series = self.extractor.get_series()
        assert series.shape[0] == self.test_data.shape[0]  # Same number of frames
        assert series.shape[1:] == self.test_data.shape[1:]  # Same image dimensions
        assert series.dtype == self.test_data.dtype  # Same data type

        # Compare with the entire test_data
        assert_array_equal(series, self.test_data)

        # Test with start and end frame
        start_sample = 0
        end_sample = 2
        series_slice = self.extractor.get_series(start_sample=start_sample, end_sample=end_sample)
        assert series_slice.shape[0] == end_sample - start_sample  # Correct number of frames
        assert series_slice.shape[1:] == self.test_data.shape[1:]  # Same image dimensions

        # Compare with the corresponding slice of test_data
        assert_array_equal(series_slice, self.test_data[start_sample:end_sample])

    def test_thor_tiff_extractor_get_frames(self):
        """Test the get_frames method."""
        frame_idxs = [0, 1, 2]
        frames = self.extractor.get_frames(frame_idxs=frame_idxs)
        assert frames.shape[0] == len(frame_idxs)  # Correct number of frames
        assert frames.shape[1:] == self.test_data.shape[1:]  # Same image dimensions

        # Compare with frames extracted directly from the test_data
        for i, frame_idx in enumerate(frame_idxs):
            assert_array_equal(frames[i], self.test_data[frame_idx])

        # Test with non-consecutive frames
        frame_idxs = [0, 2]
        frames = self.extractor.get_frames(frame_idxs=frame_idxs)
        assert frames.shape[0] == len(frame_idxs)  # Correct number of frames
        assert frames.shape[1:] == self.test_data.shape[1:]  # Same image dimensions

        # Compare with frames extracted directly from the test_data
        for i, frame_idx in enumerate(frame_idxs):
            assert_array_equal(frames[i], self.test_data[frame_idx])

    def test_experiment_xml(self):
        """Test parsing of Experiment.xml."""
        # Test that sampling frequency was extracted from Experiment.xml
        assert self.extractor.get_sampling_frequency() is not None

        # Test that channel names were extracted from Experiment.xml
        assert len(self.extractor.get_channel_names()) > 0

        date_value = self.extractor._experiment_xml_dict["ThorImageExperiment"]["Date"]

        unix_timestamp = int(date_value["@uTime"])
        assert unix_timestamp == 1697650759
        datetime_utc = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)

        # Assert that the date extracted from Experiment.xml matches the expected datetime
        expected_datetime = datetime(2023, 10, 18, 17, 39, 19, tzinfo=timezone.utc)
        assert datetime_utc == expected_datetime
