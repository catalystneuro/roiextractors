import numpy as np
import pytest
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO
from pynwb.ophys import TwoPhotonSeries
from pynwb.testing.mock.file import mock_NWBFile
from pynwb.testing.mock.ophys import (
    mock_Fluorescence,
    mock_ImageSegmentation,
    mock_ImagingPlane,
    mock_PlaneSegmentation,
    mock_RoiResponseSeries,
)

from roiextractors import (
    NwbImagingExtractor,
    NwbSegmentationExtractor,
    PoissonNoiseImagingExtractor,
)

BACKEND_IO_CLASSES = {"hdf5": NWBHDF5IO, "zarr": NWBZarrIO}


def write_nwbfile(nwbfile, folder_path, stem, backend):
    """Write `nwbfile` with the requested backend and return the path it was written to."""
    suffix = ".nwb" if backend == "hdf5" else ".nwb.zarr"
    file_path = folder_path / f"{stem}{suffix}"
    with BACKEND_IO_CLASSES[backend](str(file_path), "w") as io:
        io.write(nwbfile)
    return file_path


@pytest.fixture(scope="module", params=["hdf5", "zarr"])
def nwb_planar_file(request, tmp_path_factory):
    """Create a planar (2D) NWB file for testing."""
    tmp_path = tmp_path_factory.mktemp("nwb_planar")

    sampling_frequency = 30.0
    num_samples = 30
    rows = 50
    columns = 25

    nwbfile = mock_NWBFile()

    # Generate planar data: (time, rows, cols)
    dtype = "uint16"
    video = PoissonNoiseImagingExtractor(
        num_samples=num_samples, num_rows=rows, num_columns=columns, sampling_frequency=sampling_frequency
    ).get_series()
    video = video.astype(dtype)

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

    file_path = write_nwbfile(nwbfile, tmp_path, "test_nwb_planar_imaging_extractor", request.param)

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


@pytest.fixture(scope="module", params=["hdf5", "zarr"])
def nwb_volumetric_file(request, tmp_path_factory):
    """Create a volumetric (3D) NWB file for testing."""
    tmp_path = tmp_path_factory.mktemp("nwb_volumetric")

    sampling_frequency = 30.0
    num_samples = 30
    rows = 50
    columns = 25
    num_planes = 10
    starting_time = 10.0  # Non-zero starting time

    nwbfile = mock_NWBFile()

    # Generate volumetric data: (time, rows, cols, planes)
    dtype = "uint16"
    video = PoissonNoiseImagingExtractor(
        num_samples=num_samples,
        num_rows=rows,
        num_columns=columns,
        num_planes=num_planes,
        sampling_frequency=sampling_frequency,
    ).get_series()
    video = video.astype(dtype)

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

    file_path = write_nwbfile(nwbfile, tmp_path, "test_nwb_volumetric_imaging_extractor", request.param)

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


@pytest.fixture(scope="module", params=["hdf5", "zarr"])
def nwb_segmentation_file(request, tmp_path_factory):
    """Create a segmentation NWB file for testing."""
    tmp_path = tmp_path_factory.mktemp("nwb_segmentation")

    sampling_frequency = 30.0
    num_samples = 30
    num_rois = 4

    nwbfile = mock_NWBFile()
    imaging_plane = mock_ImagingPlane(nwbfile=nwbfile)
    plane_segmentation = mock_PlaneSegmentation(
        name="PlaneSegmentation",
        imaging_plane=imaging_plane,
        n_rois=num_rois,
    )
    mock_ImageSegmentation(
        name="ImageSegmentation",
        plane_segmentations=[plane_segmentation],
        nwbfile=nwbfile,
    )

    traces = np.arange(num_samples * num_rois, dtype="float64").reshape(num_samples, num_rois)
    roi_response_series = mock_RoiResponseSeries(
        name="RoiResponseSeries",
        data=traces,
        plane_segmentation=plane_segmentation,
        rate=sampling_frequency,
    )
    mock_Fluorescence(
        name="Fluorescence",
        roi_response_series=[roi_response_series],
        nwbfile=nwbfile,
    )

    # NWB stores image masks as (num_rois, columns, rows), roiextractors as (rows, columns, num_rois)
    image_masks = np.asarray(plane_segmentation["image_mask"].data).transpose([2, 1, 0])

    file_path = write_nwbfile(nwbfile, tmp_path, "test_nwb_segmentation_extractor", request.param)

    return {
        "file_path": file_path,
        "traces": traces,
        "image_masks": image_masks,
        "num_samples": num_samples,
        "num_rois": num_rois,
        "sampling_frequency": sampling_frequency,
    }


class TestNwbSegmentationExtractor:
    """Tests for NWB segmentation data."""

    def test_roi_ids_and_num_samples(self, nwb_segmentation_file):
        extractor = NwbSegmentationExtractor(file_path=nwb_segmentation_file["file_path"])

        assert extractor.get_num_rois() == nwb_segmentation_file["num_rois"]
        assert extractor.get_roi_ids() == list(range(nwb_segmentation_file["num_rois"]))
        assert extractor.get_num_samples() == nwb_segmentation_file["num_samples"]
        assert extractor.get_sampling_frequency() == nwb_segmentation_file["sampling_frequency"]

    def test_get_traces(self, nwb_segmentation_file):
        extractor = NwbSegmentationExtractor(file_path=nwb_segmentation_file["file_path"])

        traces = extractor.get_traces_dict()["raw"]
        np.testing.assert_array_almost_equal(np.asarray(traces), nwb_segmentation_file["traces"])

    def test_get_roi_image_masks(self, nwb_segmentation_file):
        extractor = NwbSegmentationExtractor(file_path=nwb_segmentation_file["file_path"])
        image_masks = nwb_segmentation_file["image_masks"]

        assert extractor.get_frame_shape() == image_masks.shape[:2]
        np.testing.assert_array_almost_equal(np.asarray(extractor.get_roi_image_masks()), image_masks)

    def test_get_native_timestamps(self, nwb_segmentation_file):
        extractor = NwbSegmentationExtractor(file_path=nwb_segmentation_file["file_path"])
        num_samples = nwb_segmentation_file["num_samples"]
        sampling_frequency = nwb_segmentation_file["sampling_frequency"]

        timestamps = extractor.get_native_timestamps()
        expected_timestamps = np.arange(num_samples) / sampling_frequency
        assert timestamps is not None
        np.testing.assert_array_almost_equal(timestamps, expected_timestamps)

        sliced_timestamps = extractor.get_native_timestamps(start_sample=5, end_sample=10)
        np.testing.assert_array_almost_equal(sliced_timestamps, expected_timestamps[5:10])
