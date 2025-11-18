import numpy as np
import pytest
from pynwb import NWBHDF5IO
from pynwb.ophys import TwoPhotonSeries
from pynwb.testing.mock.file import mock_NWBFile
from pynwb.testing.mock.ophys import mock_ImagingPlane

from roiextractors import NwbImagingExtractor
from roiextractors.testing import generate_dummy_video


@pytest.fixture(scope="module")
def nwb_planar_file(tmp_path_factory):
    """Create a planar (2D) NWB file for testing."""
    tmp_path = tmp_path_factory.mktemp("nwb_planar")
    file_path = tmp_path / "test_nwb_planar_imaging_extractor.nwb"

    sampling_frequency = 30.0
    num_samples = 30
    rows = 50
    columns = 25

    nwbfile = mock_NWBFile()

    # Generate planar data: (time, rows, cols)
    video_shape = (num_samples, rows, columns)

    dtype = "uint16"
    video = generate_dummy_video(size=video_shape, dtype=dtype)

    imaging_plane = mock_ImagingPlane(nwbfile=nwbfile)

    # NWB format: (time, width, height)
    # So transpose from (time, rows, cols) to (time, cols, rows)
    image_series = TwoPhotonSeries(
        name="TwoPhotonSeries",
        data=video.transpose([0, 2, 1]),  # roiextractors -> NWB transpose
        imaging_plane=imaging_plane,
        rate=sampling_frequency,
        unit="normalized amplitude",
    )

    nwbfile.add_acquisition(image_series)

    with NWBHDF5IO(file_path, "w") as io:
        io.write(nwbfile)

    return {
        "file_path": file_path,
        "video": video,
        "frame_shape": (rows, columns),
        "num_samples": num_samples,
    }


class TestNwbImagingExtractor:
    """Tests for planar (2D) NWB imaging data."""

    def test_get_image_shape_and_num_samples(self, nwb_planar_file):
        nwb_imaging_extractor = NwbImagingExtractor(file_path=nwb_planar_file["file_path"])

        image_shape = nwb_imaging_extractor.get_image_shape()
        num_samples = nwb_imaging_extractor.get_num_samples()

        assert image_shape == nwb_planar_file["frame_shape"]
        assert num_samples == nwb_planar_file["num_samples"]

    def test_get_samples_continuous(self, nwb_planar_file):
        nwb_imaging_extractor = NwbImagingExtractor(file_path=nwb_planar_file["file_path"])
        video = nwb_planar_file["video"]

        # Test with continuous indices
        sample_indices = [0, 1, 2, 3, 4]
        samples = nwb_imaging_extractor.get_samples(sample_indices)
        expected_samples = video[sample_indices, ...]
        np.testing.assert_array_almost_equal(samples, expected_samples)

    def test_get_samples_non_continuous(self, nwb_planar_file):
        nwb_imaging_extractor = NwbImagingExtractor(file_path=nwb_planar_file["file_path"])
        video = nwb_planar_file["video"]

        # Test with non-continuous indices
        sample_indices = [0, 2, 5, 10]
        samples = nwb_imaging_extractor.get_samples(sample_indices)
        expected_samples = video[sample_indices, ...]
        np.testing.assert_array_almost_equal(samples, expected_samples)

    def test_get_series(self, nwb_planar_file):
        nwb_imaging_extractor = NwbImagingExtractor(file_path=nwb_planar_file["file_path"])
        video = nwb_planar_file["video"]

        series = nwb_imaging_extractor.get_series()
        expected_series = video

        np.testing.assert_array_almost_equal(series, expected_series)


@pytest.fixture(scope="module")
def nwb_volumetric_file(tmp_path_factory):
    """Create a volumetric (3D) NWB file for testing."""
    tmp_path = tmp_path_factory.mktemp("nwb_volumetric")
    file_path = tmp_path / "test_nwb_volumetric_imaging_extractor.nwb"

    sampling_frequency = 30.0
    num_samples = 30
    rows = 50
    columns = 25
    num_planes = 10
    starting_time = 10.0  # Non-zero starting time

    nwbfile = mock_NWBFile()

    # Generate volumetric data: (time, rows, cols, planes)
    video_shape = (num_samples, rows, columns, num_planes)

    dtype = "uint16"
    video = generate_dummy_video(size=video_shape, dtype=dtype)

    imaging_plane = mock_ImagingPlane(nwbfile=nwbfile)

    # NWB format: (time, width, height, depth)
    # So transpose from (time, rows, cols, planes) to (time, cols, rows, planes)
    image_series = TwoPhotonSeries(
        name="TwoPhotonSeries",
        data=video.transpose([0, 2, 1, 3]),  # roiextractors -> NWB transpose
        imaging_plane=imaging_plane,
        starting_time=starting_time,
        rate=sampling_frequency,
        unit="normalized amplitude",
    )

    nwbfile.add_acquisition(image_series)

    with NWBHDF5IO(file_path, "w") as io:
        io.write(nwbfile)

    return {
        "file_path": file_path,
        "video": video,
        "frame_shape": (rows, columns),
        "num_samples": num_samples,
        "num_planes": num_planes,
        "starting_time": starting_time,
        "sampling_frequency": sampling_frequency,
    }


class TestNwbVolumetricImagingExtractor:
    """Tests for volumetric (3D) NWB imaging data."""

    def test_get_series_volumetric(self, nwb_volumetric_file):
        nwb_imaging_extractor = NwbImagingExtractor(file_path=nwb_volumetric_file["file_path"])
        video = nwb_volumetric_file["video"]

        # Test full series
        series = nwb_imaging_extractor.get_series()
        expected_series = video
        assert series.shape == expected_series.shape
        np.testing.assert_array_almost_equal(series, expected_series)

    def test_get_native_timestamps(self, nwb_volumetric_file):
        nwb_imaging_extractor = NwbImagingExtractor(file_path=nwb_volumetric_file["file_path"])
        num_samples = nwb_volumetric_file["num_samples"]
        starting_time = nwb_volumetric_file["starting_time"]

        # Test full timestamps
        timestamps = nwb_imaging_extractor.get_native_timestamps()
        assert timestamps is not None
        assert len(timestamps) == num_samples
        assert isinstance(timestamps, np.ndarray)

        # Test that the first timestamp corresponds to the starting_time
        assert timestamps[0] == starting_time
