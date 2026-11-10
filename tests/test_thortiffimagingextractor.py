from datetime import datetime, timezone

import numpy as np
import pytest
import tifffile
from numpy.testing import assert_array_equal

from roiextractors import ThorTiffImagingExtractor

from .extractor_mixins import ImagingExtractorTestMixin, check_metadata
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
    expected_native_timestamps = None  # the OME-XML has a DeltaT for the first plane only
    expected_sampling_frequency = 13.967  # frameRate, streaming capture without averaging
    expected_metadata = {
        "acquisition_start_datetime": datetime(2023, 10, 18, 17, 39, 19, tzinfo=timezone.utc),
        "pixel_height_um": 0.884,
        "pixel_width_um": 0.884,
        "field_of_view_height_um": 452.7,
        "field_of_view_width_um": 452.7,
        "channel_name": "ChanA",
        "acquisition_software_name": "ThorImageLS",
        "acquisition_software_version": "5.0.2023.10041",
        "objective_magnification": 16.0,
        "session_id": "20231018-002",
    }

    def check_series(self, series):
        """Each sample is the page of the file named for its timepoint, `<channel>_001_001_001_<sample + 1>.tif`."""
        channel_name = self.extractor_kwargs.get("channel_name", self.expected_channel_names[0])
        for sample_index in range(self.expected_num_samples):
            file_path = TEST_DIR / f"{channel_name}_001_001_001_{sample_index + 1:03d}.tif"
            with tifffile.TiffFile(file_path) as tiff_file:
                expected_sample = tiff_file.pages[0].asarray()
            sample = series[sample_index]
            assert_array_equal(sample, expected_sample)

    @classmethod
    def setup_class(cls):
        """Set up the test."""
        if not FILE_PATH.exists():
            pytest.skip(f"Test file {FILE_PATH} not found. Skipping tests.")

        # Create the extractor
        cls.extractor = ThorTiffImagingExtractor(file_path=FILE_PATH)

    def test_thor_tiff_extractor_sampling_frequency(self):
        """Test the sampling frequency property."""
        assert self.extractor.get_sampling_frequency() is not None
        assert isinstance(self.extractor.get_sampling_frequency(), float)

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
    # Z and T capture: the DeltaT in the OME-XML of ChanA_0001_0001_0001_<sample + 1>.tif, as Thor gives a volume
    # the time of its first plane. They are about 1.97 s apart while frameRate is 60.087 Hz, because each plane
    # averages 10 scans and the stage moves between planes
    expected_native_timestamps = [0.0, 1.97315, 3.94218]
    # DeltaT in the OME-XML of ChanA_0001_0001_<plane + 1>_<sample + 1>.tif, a list per plane
    expected_plane_native_timestamps = [
        [0.0, 1.97315, 3.94218],
        [0.384067, 2.36012, 4.33632],
        [0.777047, 2.74819, 4.72219],
    ]
    expected_metadata = {
        "acquisition_start_datetime": datetime(2019, 11, 21, 10, 15, 18, tzinfo=timezone.utc),
        "pixel_height_um": 0.593,
        "pixel_width_um": 0.593,
        "field_of_view_height_um": 75.88,
        "field_of_view_width_um": 75.88,
        "plane_z_offsets_um": [0.0, 15.0, 30.0],
        "channel_name": "ChanA",
        "acquisition_software_name": "ThorImageLS",
        "acquisition_software_version": "3.2.2018.4241",
        "objective_magnification": 22.2222222222222,
        "session_id": "Untitled_003",
    }

    def check_series(self, series):
        """Each plane of each sample is the page of `<channel>_0001_0001_<plane + 1>_<sample + 1>.tif`, in Z and T capture."""
        channel_name = self.extractor_kwargs.get("channel_name", self.expected_channel_names[0])
        for sample_index in range(self.expected_num_samples):
            for plane_index in range(self.expected_sample_shape[2]):
                file_path = (
                    VOLUMETRIC_DIR / f"{channel_name}_0001_0001_{plane_index + 1:04d}_{sample_index + 1:04d}.tif"
                )
                with tifffile.TiffFile(file_path) as tiff_file:
                    expected_plane = tiff_file.pages[0].asarray()
                plane = series[sample_index, :, :, plane_index]
                assert_array_equal(plane, expected_plane)

    @pytest.mark.skip(reason="The rate of Z and T capture is parked until the policy for data without a set rate")
    def test_sampling_frequency(self, setup_extractor):
        """It reports `frameRate`, the scan rate, instead of the rate of the saved volumes."""

    @classmethod
    def setup_class(cls):
        """Set up the test."""
        cls.file_path = VOLUMETRIC_DIR / "ChanA_0001_0001_0001_0001.tif"
        if not cls.file_path.exists():
            pytest.skip(f"Test file {cls.file_path} not found. Skipping tests.")

        cls.extractor = ThorTiffImagingExtractor(file_path=cls.file_path, channel_name="ChanA")

    def test_channels_hold_different_data(self):
        """Selecting the other channel reads other pages."""
        other = ThorTiffImagingExtractor(file_path=self.file_path, channel_name="ChanB")
        assert not np.array_equal(self.extractor.get_series(), other.get_series())
        check_metadata(other.metadata, {**self.expected_metadata, "channel_name": "ChanB"})

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
    expected_native_timestamps = None  # streaming capture, where DeltaT is not the time of the scan
    expected_sampling_frequency = 90.037 / 9  # frameRate / averageNum, each saved frame is the mean of 9 scans
    expected_metadata = {
        "acquisition_start_datetime": datetime(2025, 4, 14, 17, 7, 32, tzinfo=timezone.utc),
        "pixel_height_um": 0.304,
        "pixel_width_um": 0.304,
        "field_of_view_height_um": 77.78,
        "field_of_view_width_um": 116.67,
        "channel_name": "ChanA",
        "acquisition_software_name": "ThorImageLS",
        "acquisition_software_version": "4.1.2021.9131",
        "objective_magnification": 22.2222,
        "session_id": "250414-MB010B-2xRSET-jGC8s-9AVG-Day1-F02_001",
    }

    def check_series(self, series):
        """Each sample is the page of the file named for its timepoint."""
        channel_name = self.extractor_kwargs.get("channel_name", self.expected_channel_names[0])
        for sample_index in range(self.expected_num_samples):
            file_path = STREAMING_AVERAGING_DIR / f"{channel_name}_001_001_001_{sample_index + 1:03d}.tif"
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
    expected_native_timestamps = None  # streaming capture, where DeltaT is not the time of the scan
    expected_sampling_frequency = 58.32 / (7 + 2)  # frameRate / (steps + flybackFrames), 7 planes and 2 flyback scans
    expected_metadata = {
        "acquisition_start_datetime": datetime(2018, 2, 16, 20, 40, 49, tzinfo=timezone.utc),
        "pixel_height_um": 0.476,
        "pixel_width_um": 0.476,
        "field_of_view_height_um": 121.78,
        "field_of_view_width_um": 121.78,
        "plane_z_offsets_um": [0.0, 7.0, 14.0, 21.0, 28.0, 35.0, 42.0],
        "channel_name": "ChanB",
        "acquisition_software_name": "ThorImageLS",
        "acquisition_software_version": "3.0.2016.10131",
        "objective_magnification": 22.2,
        "session_id": "2R60F02_Gcamp6s_after_control_LH_00001",
    }

    def check_series(self, series):
        """Each (sample, plane) position holds the page of the file named for that plane and volume.

        ThorImage names the files ``ChanB_0001_0001_<plane>_<volume>.tif``. The flyback frames are not saved, so
        the plane indices run from 1 to 7 in every volume.
        """
        channel_name = self.extractor_kwargs.get("channel_name", self.expected_channel_names[0])
        for sample_index in range(self.expected_num_samples):
            for plane_index in range(self.expected_sample_shape[2]):
                file_path = (
                    FAST_Z_STREAMING_DIR / f"{channel_name}_0001_0001_{plane_index + 1:04d}_{sample_index + 1:04d}.tif"
                )
                with tifffile.TiffFile(file_path) as tiff_file:
                    expected_plane = tiff_file.pages[0].asarray()
                plane = series[sample_index, :, :, plane_index]
                assert_array_equal(plane, expected_plane)
