from roiextractors.tools.testing import (
    generate_mock_video,
    generate_mock_imaging_extractor,
    generate_mock_segmentation_extractor,
    assert_imaging_equal,
    imaging_equal,
    assert_segmentation_equal,
    segmentation_equal,
)
import pytest
import numpy as np
from numpy.testing import assert_array_equal


@pytest.mark.parametrize("size", [(1, 2, 3), (3, 2, 4), (5, 3, 2)])
def test_generate_mock_video_size(size):
    video = generate_mock_video(size=size)
    assert video.shape == size


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, "uint8", "float32"])
def test_generate_mock_video_dtype(dtype):
    video = generate_mock_video(size=(3, 2, 4), dtype=dtype)
    assert video.dtype == np.dtype(dtype)


def test_generate_mock_video_seed():
    size = (1, 2, 3)
    video1 = generate_mock_video(size=size, seed=0)
    video2 = generate_mock_video(size=size, seed=0)
    video3 = generate_mock_video(size=size, seed=1)
    assert_array_equal(video1, video2)
    assert not np.array_equal(video1, video3)


@pytest.mark.parametrize("num_frames, num_rows, num_columns", [(1, 2, 3), (3, 2, 4), (5, 3, 2)])
def test_generate_mock_imaging_extractor_shape(num_frames, num_rows, num_columns):
    imaging_extractor = generate_mock_imaging_extractor(
        num_frames=num_frames, num_rows=num_rows, num_columns=num_columns
    )
    video = imaging_extractor.get_video()
    assert video.shape == (num_frames, num_rows, num_columns)


@pytest.mark.parametrize("sampling_frequency", [10.0, 20.0, 30.0])
def test_generate_mock_imaging_extractor_sampling_frequency(sampling_frequency):
    imaging_extractor = generate_mock_imaging_extractor(sampling_frequency=sampling_frequency)
    assert imaging_extractor.get_sampling_frequency() == sampling_frequency


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, "uint8", "float32"])
def test_generate_mock_imaging_extractor_dtype(dtype):
    imaging_extractor = generate_mock_imaging_extractor(dtype=dtype)
    assert imaging_extractor.get_dtype() == np.dtype(dtype)


def test_generate_mock_imaging_extractor_seed():
    imaging_extractor1 = generate_mock_imaging_extractor(seed=0)
    imaging_extractor2 = generate_mock_imaging_extractor(seed=0)
    imaging_extractor3 = generate_mock_imaging_extractor(seed=1)
    assert_imaging_equal(imaging_extractor1, imaging_extractor2)
    assert not imaging_equal(imaging_extractor1, imaging_extractor3)


@pytest.mark.parametrize(
    "num_rois, num_frames, num_rows, num_columns, num_background_components",
    [(1, 2, 3, 4, 5), (3, 2, 4, 5, 6), (5, 3, 2, 1, 0)],
)
def test_generate_mock_segmentation_extractor_shape(
    num_rois, num_frames, num_rows, num_columns, num_background_components
):
    segmentation_extractor = generate_mock_segmentation_extractor(
        num_rois=num_rois,
        num_frames=num_frames,
        num_rows=num_rows,
        num_columns=num_columns,
        num_background_components=num_background_components,
    )
    assert segmentation_extractor.get_num_rois() == num_rois
    assert segmentation_extractor.get_num_frames() == num_frames
    assert segmentation_extractor.get_image_size() == (num_rows, num_columns)
    assert segmentation_extractor.get_num_background_components() == num_background_components


@pytest.mark.parametrize("sampling_frequency", [10.0, 20.0, 30.0])
def test_generate_mock_segmentation_extractor_sampling_frequency(sampling_frequency):
    segmentation_extractor = generate_mock_segmentation_extractor(sampling_frequency=sampling_frequency)
    assert segmentation_extractor.get_sampling_frequency() == sampling_frequency


@pytest.mark.parametrize(
    "summary_image_names, roi_response_names, background_response_names",
    [
        ([], ["denoised"], []),
        (["mean"], ["raw", "dff"], ["background"]),
        (["correlation"], ["deconvolved"], ["background"]),
    ],
)
def test_generate_mock_segmentation_extractor_names(summary_image_names, roi_response_names, background_response_names):
    segmentation_extractor = generate_mock_segmentation_extractor(
        summary_image_names=summary_image_names,
        roi_response_names=roi_response_names,
        background_response_names=background_response_names,
    )
    assert list(segmentation_extractor.get_summary_images().keys()) == summary_image_names
    assert list(segmentation_extractor.get_roi_response_traces().keys()) == roi_response_names
    assert list(segmentation_extractor.get_background_response_traces().keys()) == background_response_names


@pytest.mark.parametrize("rejected_roi_ids", [[], [0, 1], [1, 2, 3]])
def test_generate_mock_segmentation_extractor_rejected_list(rejected_roi_ids):
    segmentation_extractor = generate_mock_segmentation_extractor(rejected_roi_ids=rejected_roi_ids)
    assert segmentation_extractor.get_rejected_roi_ids() == rejected_roi_ids


def test_generate_mock_segmentation_extractor_seed():
    segmentation_extractor1 = generate_mock_segmentation_extractor(seed=0)
    segmentation_extractor2 = generate_mock_segmentation_extractor(seed=0)
    segmentation_extractor3 = generate_mock_segmentation_extractor(seed=1)
    assert_segmentation_equal(segmentation_extractor1, segmentation_extractor2)
    assert not segmentation_equal(segmentation_extractor1, segmentation_extractor3)
