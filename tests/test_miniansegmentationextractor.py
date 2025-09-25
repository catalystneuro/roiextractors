import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from numpy.testing import assert_array_equal

from roiextractors import MinianSegmentationExtractor
from tests.setup_paths import OPHYS_DATA_PATH


@pytest.fixture(scope="module")
def folder_path():
    return OPHYS_DATA_PATH / "segmentation_datasets" / "minian" / "segmented_data_3units_100frames"


@pytest.fixture(scope="module")
def extractor(folder_path):
    return MinianSegmentationExtractor(folder_path=folder_path)


@pytest.fixture(scope="module")
def denoised_traces(folder_path):
    dataset = zarr.open(str(folder_path) + "/C.zarr")
    return np.transpose(np.array(dataset["C"]))


@pytest.fixture(scope="module")
def deconvolved_traces(folder_path):
    dataset = zarr.open(str(folder_path) + "/S.zarr")
    return np.transpose(np.array(dataset["S"]))


@pytest.fixture(scope="module")
def baseline_traces(folder_path):
    dataset = zarr.open(str(folder_path) + "/b0.zarr")
    return np.transpose(np.array(dataset["b0"]))


@pytest.fixture(scope="module")
def neuropil_trace(folder_path):
    dataset = zarr.open(str(folder_path) + "/f.zarr")
    return np.expand_dims(np.array(dataset["f"]), axis=1)


@pytest.fixture(scope="module")
def image_masks(folder_path):
    dataset = zarr.open(str(folder_path) + "/A.zarr")
    return np.transpose(np.array(dataset["A"]), (1, 2, 0))


@pytest.fixture(scope="module")
def background_image_mask(folder_path):
    dataset = zarr.open(str(folder_path) + "/b.zarr")
    return np.expand_dims(np.array(dataset["b"]), axis=2)


@pytest.fixture(scope="module")
def maximum_projection_image(folder_path):
    return np.array(zarr.open(str(folder_path) + "/max_proj.zarr/max_proj"))


@pytest.fixture(scope="module")
def expected_properties():
    return dict(
        num_samples=100,
        frame_shape=(608, 608),
        num_rois=3,
        first_timestamp=[0.329],
        roi_ids=[0, 1, 2],
        subject_id="Ca_EEG3",
        session_id="Ca_EEG3-4",
    )


def test_incomplete_extractor_load_temporal_components(folder_path, tmp_path):
    """Check extractor can be initialized when not all traces are available."""
    # temporary directory for testing assertion when some of the files are missing
    folders_to_copy = [
        "A.zarr",
        "b.zarr",
        "f.zarr",
        "max_proj.zarr",
        ".zgroup",
        "timeStamps.csv",
    ]

    for folder in folders_to_copy:
        src = Path(folder_path) / folder
        dst = tmp_path / folder
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy(src, dst)
    with pytest.raises(
        ValueError,
        match=r"Spatial components \(A\.zarr\) are available but no temporal components \(C\.zarr, S\.zarr, b0\.zarr\) are associated\. "
        r"This means ROI masks exist but without any corresponding fluorescence traces\.",
    ):
        MinianSegmentationExtractor(folder_path=tmp_path)


def test_incomplete_extractor_load_spatial_component(folder_path, tmp_path):
    """Check extractor can be initialized when not all spatial components are available."""
    # temporary directory for testing assertion when some of the files are missing
    folders_to_copy = [
        "S.zarr",
        "C.zarr",
        "b0.zarr",
        "b.zarr",
        "f.zarr",
        "max_proj.zarr",
        ".zgroup",
        "timeStamps.csv",
    ]

    for folder in folders_to_copy:
        src = Path(folder_path) / folder
        dst = tmp_path / folder
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy(src, dst)

    with pytest.raises(ValueError, match="No image masks found in A.zarr dataset."):
        MinianSegmentationExtractor(folder_path=tmp_path)


def test_frame_shape(extractor, expected_properties):
    assert extractor.get_frame_shape() == expected_properties["frame_shape"]


def test_num_samples(extractor, expected_properties):
    assert extractor.get_num_samples() == expected_properties["num_samples"]


def test_sample_indices_to_time(extractor, expected_properties):
    assert extractor.sample_indices_to_time(sample_indices=[0]) == expected_properties["first_timestamp"]


def test_num_rois(extractor, expected_properties):
    assert extractor.get_num_rois() == expected_properties["num_rois"]


def test_extractor_denoised_traces(extractor, denoised_traces):
    assert_array_equal(extractor.get_traces(name="denoised"), denoised_traces)


def test_extractor_neuropil_trace(extractor, neuropil_trace):
    assert_array_equal(extractor.get_traces(name="neuropil"), neuropil_trace)


def test_extractor_image_masks(extractor, image_masks):
    """Test that the image masks are correctly extracted."""
    assert_array_equal(extractor.get_roi_image_masks(), image_masks)


def test_extractor_background_image_masks(extractor, background_image_mask):
    """Test that the image masks are correctly extracted."""
    assert_array_equal(extractor.get_background_image_masks(), background_image_mask)


def test_maximum_projection_image(extractor, maximum_projection_image):
    """Test that the mean image is correctly loaded from the extractor."""
    images_dict = extractor.get_images_dict()
    assert_array_equal(images_dict["maximum_projection"], maximum_projection_image)


def test_get_native_timestamps(extractor, expected_properties):
    """Test that timestamps are correctly read from CSV file."""
    # Get timestamps using the extractor
    timestamps = extractor.get_native_timestamps()

    # First timestamp should match the expected value
    assert timestamps[0] == expected_properties["first_timestamp"]

    # Length should match number of frames
    assert len(timestamps) == expected_properties["num_samples"]


def test_get_roi_ids(extractor, expected_properties):
    """Test that ROI IDs are correctly retrieved."""
    roi_ids = extractor.get_roi_ids()

    # Test the number of ROIs
    assert len(roi_ids) == expected_properties["num_rois"]

    # Test that ROI IDs match expected values
    assert roi_ids == expected_properties["roi_ids"]


def test_get_subject_id(extractor, expected_properties):
    """Test that subject ID is correctly retrieved."""
    subject_id = extractor._get_subject_id()
    assert subject_id == expected_properties["subject_id"]


def test_get_session_id(extractor, expected_properties):
    """Test that session ID is correctly retrieved."""
    session_id = extractor._get_session_id()
    assert session_id == expected_properties["session_id"]


def test_slicing_preserves_has_time_vector():
    """Test that slicing preserves the has_time_vector property."""
    folder_path = OPHYS_DATA_PATH / "segmentation_datasets" / "minian" / "segmented_data_3units_100frames"
    extractor = MinianSegmentationExtractor(folder_path=folder_path)
    assert extractor.has_time_vector()
    sub_extractor = extractor.slice_samples(start_sample=0, end_sample=10)
    assert sub_extractor.has_time_vector()
