import platform
from datetime import datetime

import numpy as np
import pytest
from numpy import dtype
from numpy.testing import assert_array_equal

from roiextractors import InscopixImagingExtractor
from tests.extractor_mixins import ImagingExtractorTestMixin, check_metadata
from tests.setup_paths import OPHYS_DATA_PATH

# Skip all tests in this file on macOS
isx = pytest.importorskip("isx", reason="isx package is required for these tests.")
pytestmark = pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "arm64",
    reason="The isx package is currently not natively supported on macOS with Apple Silicon. "
    "Installation instructions can be found at: "
    "https://github.com/inscopix/pyisx?tab=readme-ov-file#install",
)

INSCOPIX_PATH = OPHYS_DATA_PATH / "imaging_datasets" / "inscopix"


def read_isx_movie(file_path):
    """Read every frame of an Inscopix movie with `isx`, as an array of (frames, rows, columns)."""
    movie = isx.Movie.read(str(file_path))
    return np.stack([movie.get_frame_data(frame_index) for frame_index in range(movie.timing.num_samples)])


class TestInscopixImagingExtractorMovie128x128x100Part1(ImagingExtractorTestMixin):
    """A float32 movie written with `isx.Movie.write`, which drops the acquisition metadata (isxcore 1.8.0)."""

    extractor_cls = InscopixImagingExtractor
    extractor_kwargs = dict(file_path=INSCOPIX_PATH / "movie_128x128x100_part1.isxd")
    expected_num_samples = 100
    expected_sample_shape = (128, 128)
    expected_dtype = np.dtype("float32")
    expected_native_timestamps = None
    expected_sampling_frequency = 10.0  # 1 / timing.period, 100 ms
    expected_metadata = {
        # timing.start is the Unix epoch, as the write path keeps no acquisition time
        "acquisition_start_datetime": datetime(1970, 1, 1),
        "pixel_height_um": 3.0,  # spacing, 3 µm
        "pixel_width_um": 3.0,
        "field_of_view_height_um": 384.0,  # 128 rows of 3 µm
        "field_of_view_width_um": 384.0,
        "excitation_mode": "one-photon",
        "microscope_manufacturer": "Inscopix",
    }

    def check_series(self, series):
        """The series is the frames that `isx` reads from the file."""
        expected_series = read_isx_movie(self.extractor_kwargs["file_path"])
        assert_array_equal(series, expected_series)


class TestInscopixImagingExtractorMovieLongerThan3Min(ImagingExtractorTestMixin):
    """A uint16 movie of 1248 frames from an NVista3 that keeps its acquisition info but not its frame headers.

    The recording is multiplane (170, 370 and 570 µm), and the extractor reads its planes as successive samples.
    """

    extractor_cls = InscopixImagingExtractor
    extractor_kwargs = dict(file_path=INSCOPIX_PATH / "movie_longer_than_3_min.isxd")
    expected_num_samples = 1248
    expected_sample_shape = (33, 29)
    expected_dtype = np.dtype("uint16")
    expected_native_timestamps = None
    expected_sampling_frequency = 1 / 0.179973  # 1 / timing.period, 179973 µs
    expected_metadata = {
        "acquisition_start_datetime": datetime(2019, 10, 7, 16, 22, 1, 524186),  # timing.start
        "pixel_height_um": 72.0,  # spacing, 72 µm after a spatial downsampling of 12
        "pixel_width_um": 72.0,
        "field_of_view_height_um": 2376.0,  # 33 rows of 72 µm
        "field_of_view_width_um": 2088.0,  # 29 columns of 72 µm
        "excitation_mode": "one-photon",
        "microscope_manufacturer": "Inscopix",
        # The rest are fields of get_acquisition_info()
        "exposure_time": 0.02,  # Exposure Time (ms)
        "camera_gain": 5.9,  # Microscope Gain
        "led_power_excitation_mW_per_mm2": 0.4,  # Microscope EX LED Power (mw/mm^2)
        "led_power_orange_green_mW_per_mm2": 0.2,  # Microscope OG LED Power (mw/mm^2)
        "microscope_name": "NVista3",  # Microscope Type
        "microscope_serial_number": "FA-11092903",  # Microscope Serial Number
        "acquisition_software_version": "1.3.0",  # Acquisition SW Version
        "subject_sex": "M",  # Animal Sex, m
        "session_id": "4D_SAAV_PFC_IM7_20191007",  # Session Name
    }

    def check_series(self, series):
        """The series is the frames that `isx` reads from the file."""
        expected_series = read_isx_movie(self.extractor_kwargs["file_path"])
        assert_array_equal(series, expected_series)


# TODO: move this fixture to a mixin class once the crop of `ImagingExtractorTestMixin` fits frames of 3 by 4 pixels
def test_inscopiximagingextractor_movie_u8():
    """
    Test loading a minimal valid Inscopix movie file with uint8-encoded pixel data.

    Its frames of 3 by 4 pixels are smaller than the crop of `ImagingExtractorTestMixin`, so it is not a mixin class.

    File: movie_u8.isxd
    Metadata:
    - Source: Movie file (Inscopix isxcore v1.9.5)
    - 5 samples (frames)
    - Frame rate: 20.0 Hz
    - Volumetric: False
    - Frame shape: (3, 4) pixels
    - Pixel size: 3 µm x 3 µm
    - Channels: 1 (isx format supports only single channel)
    - No acquisition info available

    This test verifies that the extractor correctly:
    1. Loads the file and extracts metadata (frame count, resolution, data type, etc.).
    2. Retrieves the video data and raw data with correct shape and type.
    """
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "inscopix" / "movie_u8.isxd"
    extractor = InscopixImagingExtractor(file_path=file_path)

    expected_metadata = {
        "acquisition_start_datetime": datetime(1970, 1, 1),
        "pixel_height_um": 3.0,
        "pixel_width_um": 3.0,
        "field_of_view_height_um": 9.0,
        "field_of_view_width_um": 12.0,
        "excitation_mode": "one-photon",
        "microscope_manufacturer": "Inscopix",
    }
    with pytest.warns(UserWarning, match="The metadata API is experimental"):
        check_metadata(extractor.metadata, expected_metadata)

    # Basic video properties
    assert extractor.get_num_samples() == 5
    assert extractor.get_image_shape() == (3, 4)
    assert extractor.get_dtype() is dtype("uint8")
    np.testing.assert_almost_equal(extractor.get_sampling_frequency(), 20.0)
    assert extractor.get_series().shape == (5, 3, 4)

    # Test retrieving multiple samples
    samples = extractor.get_samples([0, 1, 2])
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (3, 3, 4)
    assert samples.dtype == extractor.get_dtype()
    assert extractor.get_dtype().itemsize

    raw_data = extractor.get_series()
    assert raw_data.shape == (5, 3, 4)
    assert raw_data.dtype == extractor.get_dtype()


class TestInscopixImagingExtractorMultiplaneMovie(ImagingExtractorTestMixin):
    """A uint16 dual color movie whose frames cycle over 3 focal planes (0, 500 and 900 µm) (isxcore 1.9.0).

    The extractor reads the planes as successive samples.
    """

    extractor_cls = InscopixImagingExtractor
    extractor_kwargs = dict(file_path=INSCOPIX_PATH / "multiplane_movie.isxd")
    expected_num_samples = 30
    expected_sample_shape = (100, 160)
    expected_dtype = np.dtype("uint16")
    expected_native_timestamps = None
    expected_sampling_frequency = 7.002533951423664  # 1 / timing.period, about 142.8 ms
    expected_metadata = {
        "acquisition_start_datetime": datetime(2022, 6, 15, 23, 41, 43, 403000),  # timing.start
        "pixel_height_um": 24.0,  # spacing, 24 µm
        "pixel_width_um": 24.0,
        "field_of_view_height_um": 2400.0,  # 100 rows of 24 µm
        "field_of_view_width_um": 3840.0,  # 160 columns of 24 µm
        "excitation_mode": "one-photon",
        "microscope_manufacturer": "Inscopix",
        # The rest are fields of get_acquisition_info()
        "exposure_time": 0.143,  # Exposure Time (ms)
        "camera_gain": 2.0,  # Microscope Gain
        "microscope_name": "Dual Color",  # Microscope Type
        "microscope_serial_number": "11094108",  # Microscope Serial Number
        "acquisition_software_version": "1.9.0",  # Acquisition SW Version
        "subject_sex": "M",  # Animal Sex, m
        "session_id": "Session 20220614-102826",  # Session Name
    }

    def check_series(self, series):
        """The series is the frames that `isx` reads from the file."""
        expected_series = read_isx_movie(self.extractor_kwargs["file_path"])
        assert_array_equal(series, expected_series)


class TestInscopixImagingExtractorDualColorMovieWithDroppedFrames(ImagingExtractorTestMixin):
    """A uint16 single plane dual color movie whose frames 5 to 8 were dropped (isxcore 1.5.0).

    `isx` reads a dropped frame as zeros, and so does the extractor.
    """

    extractor_cls = InscopixImagingExtractor
    extractor_kwargs = dict(file_path=INSCOPIX_PATH / "dual_color_movie_with_dropped_frames.isxd")
    expected_num_samples = 25
    expected_sample_shape = (288, 480)
    expected_dtype = np.dtype("uint16")
    expected_native_timestamps = None
    expected_sampling_frequency = 6.000177005221654  # 1 / timing.period, about 166.7 ms
    expected_metadata = {
        "acquisition_start_datetime": datetime(2020, 9, 17, 15, 47, 22, 847000),  # timing.start
        "pixel_height_um": 3.0,  # spacing, 3 µm
        "pixel_width_um": 3.0,
        "field_of_view_height_um": 864.0,  # 288 rows of 3 µm
        "field_of_view_width_um": 1440.0,  # 480 columns of 3 µm
        "excitation_mode": "one-photon",
        "microscope_manufacturer": "Inscopix",
        # The rest are fields of get_acquisition_info()
        "exposure_time": 0.079,  # Exposure Time (ms)
        "camera_gain": 2.0,  # Microscope Gain
        "microscope_name": "Dual Color",  # Microscope Type
        "microscope_serial_number": "FA-1234567",  # Microscope Serial Number
        "acquisition_software_version": "1.5.0",  # Acquisition SW Version
        "subject_sex": "M",  # Animal Sex, m
        "session_id": "Session 20200917-093000",  # Session Name
    }

    def check_series(self, series):
        """The series is the frames that `isx` reads from the file, with zeros on the dropped frames."""
        expected_series = read_isx_movie(self.extractor_kwargs["file_path"])
        assert_array_equal(series, expected_series)
        dropped_frames = [5, 6, 7, 8]  # timing.dropped
        assert not series[dropped_frames].any()
