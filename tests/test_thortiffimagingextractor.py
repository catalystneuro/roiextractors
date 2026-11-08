from datetime import datetime, timezone

import numpy as np
import pytest
import tifffile
from numpy.testing import assert_array_equal

from roiextractors import ThorTiffImagingExtractor

from .extractor_mixins import ImagingExtractorTestMixin
from .setup_paths import OPHYS_DATA_PATH

# Path to the test data
TEST_DIR = OPHYS_DATA_PATH / "imaging_datasets" / "ThorlabsTiff" / "single_channel_single_plane" / "20231018-002"
FILE_PATH = TEST_DIR / "ChanA_001_001_001_001.tif"


class TestThorTiffImagingExtractor(ImagingExtractorTestMixin):
    """Test ThorTiffImagingExtractor."""

    extractor_cls = ThorTiffImagingExtractor
    extractor_kwargs = dict(file_path=FILE_PATH)
    expected_num_samples = 3
    expected_sample_shape = (512, 512)
    expected_dtype = np.dtype("uint16")
    expected_channel_names = ["ChanA"]
    expected_has_time_vector = False  # the OME-XML has a DeltaT for the first plane only

    @classmethod
    def setup_class(cls):
        """Set up the test."""
        if not FILE_PATH.exists():
            pytest.skip(f"Test file {FILE_PATH} not found. Skipping tests.")

        # Create the extractor
        cls.extractor = ThorTiffImagingExtractor(file_path=FILE_PATH)

        # Load the test data for comparison
        cls.test_data = tifffile.imread(FILE_PATH)

    def test_thor_tiff_extractor_sampling_frequency(self):
        """Test the sampling frequency property."""
        assert self.extractor.get_sampling_frequency() is not None
        assert isinstance(self.extractor.get_sampling_frequency(), float)

    def test_thor_tiff_extractor_get_series(self):
        """Test the get_series method."""
        series = self.extractor.get_series()

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

    def test_session_start_time(self):
        """Test that the acquisition start time is parsed from Experiment.xml."""
        expected = datetime(2023, 10, 18, 17, 39, 19, tzinfo=timezone.utc)
        assert self.extractor._get_session_start_time() == expected

    def test_get_available_channel_names(self):
        """Every name discovery returns can be passed back as `channel_name`."""
        channel_names = ThorTiffImagingExtractor.get_available_channel_names(file_path=FILE_PATH)

        for channel_name in channel_names:
            extractor = ThorTiffImagingExtractor(file_path=FILE_PATH, channel_name=channel_name)
            assert extractor._get_channel_names() == channel_names


VOLUMETRIC_DIR = OPHYS_DATA_PATH / "imaging_datasets" / "ThorlabsTiff" / "multi_channel_multi_plane" / "lzw_compressed"


class TestThorTiffImagingExtractorVolumetric(ImagingExtractorTestMixin):
    """Test ThorTiffImagingExtractor on a two-channel, three-plane acquisition."""

    extractor_cls = ThorTiffImagingExtractor
    extractor_kwargs = dict(file_path=VOLUMETRIC_DIR / "ChanA_0001_0001_0001_0001.tif", channel_name="ChanA")
    expected_num_samples = 3
    expected_sample_shape = (128, 128, 3)
    expected_dtype = np.dtype("uint16")
    expected_channel_names = ["ChanA", "ChanB"]
    expected_has_time_vector = True  # Z and T capture, each file has the DeltaT of its own plane

    @classmethod
    def setup_class(cls):
        """Set up the test."""
        cls.file_path = VOLUMETRIC_DIR / "ChanA_0001_0001_0001_0001.tif"
        if not cls.file_path.exists():
            pytest.skip(f"Test file {cls.file_path} not found. Skipping tests.")

        cls.extractor = ThorTiffImagingExtractor(file_path=cls.file_path, channel_name="ChanA")

    def test_sample_and_plane_are_not_transposed(self):
        """Each (sample, plane) position holds the file named for that plane and timepoint.

        ThorImage names the files ``ChanA_0001_0001_<plane>_<timepoint>.tif``, so the mapping is
        checked against the files themselves. Shapes alone would pass even if the two were swapped.

        The pages are read one at a time because ``tifffile.imread`` on the first file of the
        acquisition returns the whole assembled OME series, not that file's single page.
        """
        series = self.extractor.get_series()
        for sample_index in range(self.expected_num_samples):
            for plane_index in range(self.expected_sample_shape[2]):
                file_path = VOLUMETRIC_DIR / f"ChanA_0001_0001_000{plane_index + 1}_000{sample_index + 1}.tif"
                with tifffile.TiffFile(file_path) as tiff_file:
                    expected = tiff_file.pages[0].asarray()
                assert_array_equal(series[sample_index, :, :, plane_index], expected)

    def test_channels_hold_different_data(self):
        """Selecting the other channel reads other pages."""
        other = ThorTiffImagingExtractor(file_path=self.file_path, channel_name="ChanB")
        assert not np.array_equal(self.extractor.get_series(), other.get_series())

    def test_native_timestamps_come_from_the_first_plane_of_each_volume(self):
        """Each sample's time is the `DeltaT` of its first plane, written in that plane's own file.

        The values are the `Plane DeltaT` of `ChanA_0001_0001_0001_000<t>.tif`. They are about 1.97 s apart,
        while `frameRate` in `Experiment.xml` is 60.087 Hz, because each plane averages 10 scans and the
        stage moves between planes.
        """
        native_timestamps = self.extractor.get_native_timestamps()
        expected_native_timestamps = np.array([0.0, 1.97315, 3.94218])
        assert_array_equal(native_timestamps, expected_native_timestamps)

    def test_select_plane_timestamps(self):
        """Each plane has the timestamps of its own frames, the `DeltaT` of `ChanA_0001_0001_000<plane>_000<t>.tif`."""
        first_plane_timestamps = self.extractor.select_plane(0).get_timestamps()
        last_plane_timestamps = self.extractor.select_plane(2).get_timestamps()
        expected_first_plane_timestamps = [0.0, 1.97315, 3.94218]
        expected_last_plane_timestamps = [0.777047, 2.74819, 4.72219]
        assert_array_equal(first_plane_timestamps, expected_first_plane_timestamps)
        assert_array_equal(last_plane_timestamps, expected_last_plane_timestamps)

    def test_get_available_channel_names(self):
        """Every name discovery returns can be passed back as `channel_name`."""
        channel_names = ThorTiffImagingExtractor.get_available_channel_names(file_path=self.file_path)

        for channel_name in channel_names:
            extractor = ThorTiffImagingExtractor(file_path=self.file_path, channel_name=channel_name)
            assert extractor._get_channel_names() == channel_names


STREAMING_AVERAGING_DIR = (
    OPHYS_DATA_PATH / "imaging_datasets" / "ThorlabsTiff" / "single_channel_single_plane" / "streaming_frame_averaging"
)


class TestThorTiffImagingExtractorStreamingFrameAveraging(ImagingExtractorTestMixin):
    """Test ThorTiffImagingExtractor on planar streaming that saves the mean of every nine scans."""

    extractor_cls = ThorTiffImagingExtractor
    extractor_kwargs = dict(file_path=STREAMING_AVERAGING_DIR / "ChanA_001_001_001_001.tif")
    expected_num_samples = 3
    expected_sample_shape = (256, 384)  # 256 lines of 384 pixels each
    expected_dtype = np.dtype("uint16")
    expected_channel_names = ["ChanA"]
    expected_has_time_vector = False  # streaming capture, where DeltaT is not the time of the scan
    expected_sampling_frequency = 90.037 / 9  # frameRate / averageNum, each saved frame is the mean of 9 scans

    def test_sampling_frequency(self, setup_extractor):
        """The rate is the scan rate divided by the scans averaged into each saved frame."""
        assert self.extractor.get_sampling_frequency() == pytest.approx(self.expected_sampling_frequency)

    def check_series(self, series):
        """Each sample is the page of the file named for its timepoint."""
        for sample_index in range(self.expected_num_samples):
            file_path = STREAMING_AVERAGING_DIR / f"ChanA_001_001_001_{sample_index + 1:03d}.tif"
            with tifffile.TiffFile(file_path) as tiff_file:
                expected_sample = tiff_file.pages[0].asarray()
            sample = series[sample_index]
            assert_array_equal(sample, expected_sample)


FAST_Z_STREAMING_DIR = (
    OPHYS_DATA_PATH / "imaging_datasets" / "ThorlabsTiff" / "single_channel_multi_plane" / "fast_z_streaming"
)


class TestThorTiffImagingExtractorFastZStreaming(ImagingExtractorTestMixin):
    """Test ThorTiffImagingExtractor on streaming with fast z, seven planes per volume and unsaved flyback frames."""

    extractor_cls = ThorTiffImagingExtractor
    extractor_kwargs = dict(file_path=FAST_Z_STREAMING_DIR / "ChanB_0001_0001_0001_0001.tif")
    expected_num_samples = 3
    expected_sample_shape = (256, 256, 7)
    expected_dtype = np.dtype("uint16")
    expected_channel_names = ["ChanB"]
    expected_has_time_vector = False  # streaming capture, where DeltaT is not the time of the scan
    expected_sampling_frequency = 58.32 / (7 + 2)  # frameRate / (steps + flybackFrames), 7 planes and 2 flyback scans

    def test_sampling_frequency(self, setup_extractor):
        """The rate is the scan rate divided by the scans of each volume, its planes plus the flyback frames."""
        assert self.extractor.get_sampling_frequency() == pytest.approx(self.expected_sampling_frequency)

    def check_series(self, series):
        """Each (sample, plane) position holds the page of the file named for that plane and volume.

        ThorImage names the files ``ChanB_0001_0001_<plane>_<volume>.tif``. The flyback frames are not saved, so
        the plane indices run from 1 to 7 in every volume.
        """
        for sample_index in range(self.expected_num_samples):
            for plane_index in range(self.expected_sample_shape[2]):
                file_path = FAST_Z_STREAMING_DIR / f"ChanB_0001_0001_{plane_index + 1:04d}_{sample_index + 1:04d}.tif"
                with tifffile.TiffFile(file_path) as tiff_file:
                    expected_plane = tiff_file.pages[0].asarray()
                plane = series[sample_index, :, :, plane_index]
                assert_array_equal(plane, expected_plane)
