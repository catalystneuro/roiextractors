from datetime import datetime, timezone

import numpy as np
import pytest
import tifffile
from numpy.testing import assert_array_equal

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

    def test_thor_tiff_extractor_get_samples(self):
        """Test the get_samples method."""
        sample_indices = [0, 1, 2]
        frames = self.extractor.get_samples(sample_indices=sample_indices)
        assert frames.shape[0] == len(sample_indices)  # Correct number of frames
        assert frames.shape[1:] == self.test_data.shape[1:]  # Same image dimensions

        # Compare with frames extracted directly from the test_data
        for i, frame_idx in enumerate(sample_indices):
            assert_array_equal(frames[i], self.test_data[frame_idx])

        # Test with non-consecutive frames
        sample_indices = [0, 2]
        frames = self.extractor.get_samples(sample_indices=sample_indices)
        assert frames.shape[0] == len(sample_indices)  # Correct number of frames
        assert frames.shape[1:] == self.test_data.shape[1:]  # Same image dimensions

        # Compare with frames extracted directly from the test_data
        for i, sample_index in enumerate(sample_indices):
            assert_array_equal(frames[i], self.test_data[sample_index])

    def test_session_start_time(self):
        """Test that the acquisition start time is parsed from Experiment.xml."""
        expected = datetime(2023, 10, 18, 17, 39, 19, tzinfo=timezone.utc)
        assert self.extractor._get_session_start_time() == expected

    def test_get_available_channel_names(self):
        """Every name discovery returns can be passed back as `channel_name`."""
        channel_names = ThorTiffImagingExtractor.get_available_channel_names(file_path=FILE_PATH)
        assert channel_names == ["ChanA"]

        for channel_name in channel_names:
            extractor = ThorTiffImagingExtractor(file_path=FILE_PATH, channel_name=channel_name)
            assert extractor._get_channel_names() == channel_names


VOLUMETRIC_DIR = OPHYS_DATA_PATH / "imaging_datasets" / "ThorlabsTiff" / "multi_channel_multi_plane" / "lzw_compressed"


class TestThorTiffImagingExtractorVolumetric:
    """Test ThorTiffImagingExtractor on a two-channel, three-plane acquisition."""

    num_samples = 3
    num_planes = 3
    image_shape = (128, 128)

    @classmethod
    def setup_class(cls):
        """Set up the test."""
        cls.file_path = VOLUMETRIC_DIR / "ChanA_0001_0001_0001_0001.tif"
        if not cls.file_path.exists():
            pytest.skip(f"Test file {cls.file_path} not found. Skipping tests.")

        cls.extractor = ThorTiffImagingExtractor(file_path=cls.file_path, channel_name="ChanA")

    def test_volumetric_shape(self):
        """The depth planes are reported as a fourth dimension."""
        assert self.extractor.is_volumetric
        assert self.extractor.get_num_planes() == self.num_planes
        assert self.extractor.get_num_samples() == self.num_samples
        assert self.extractor.get_sample_shape() == (*self.image_shape, self.num_planes)

        series = self.extractor.get_series()
        assert series.shape == (self.num_samples, *self.image_shape, self.num_planes)

    def test_sample_and_plane_are_not_transposed(self):
        """Each (sample, plane) position holds the file named for that plane and timepoint.

        ThorImage names the files ``ChanA_0001_0001_<plane>_<timepoint>.tif``, so the mapping is
        checked against the files themselves. Shapes alone would pass even if the two were swapped.

        The pages are read one at a time because ``tifffile.imread`` on the first file of the
        acquisition returns the whole assembled OME series, not that file's single page.
        """
        series = self.extractor.get_series()
        for sample_index in range(self.num_samples):
            for plane_index in range(self.num_planes):
                file_path = VOLUMETRIC_DIR / f"ChanA_0001_0001_000{plane_index + 1}_000{sample_index + 1}.tif"
                with tifffile.TiffFile(file_path) as tiff_file:
                    expected = tiff_file.pages[0].asarray()
                assert_array_equal(series[sample_index, :, :, plane_index], expected)

    def test_channels_hold_different_data(self):
        """Selecting the other channel reads other pages."""
        other = ThorTiffImagingExtractor(file_path=self.file_path, channel_name="ChanB")
        assert not np.array_equal(self.extractor.get_series(), other.get_series())

    def test_get_available_channel_names(self):
        """Every name discovery returns can be passed back as `channel_name`."""
        channel_names = ThorTiffImagingExtractor.get_available_channel_names(file_path=self.file_path)
        assert channel_names == ["ChanA", "ChanB"]

        for channel_name in channel_names:
            extractor = ThorTiffImagingExtractor(file_path=self.file_path, channel_name=channel_name)
            assert extractor._get_channel_names() == channel_names
