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


# def assert_imaging_equal(imaging_extractor1: ImagingExtractor, imaging_extractor2: ImagingExtractor):
#     """Assert that two ImagingExtractor objects are equal by comparing their attributes and data.

#     Parameters
#     ----------
#     imaging_extractor1 : ImagingExtractor
#         The first ImagingExtractor object to compare.
#     imaging_extractor2 : ImagingExtractor
#         The second ImagingExtractor object to compare.

#     Raises
#     ------
#     AssertionError
#         If any of the following attributes or data do not match between the two ImagingExtractor objects:
#         - Image size
#         - Number of frames
#         - Sampling frequency
#         - Data type (dtype)
#         - Video data
#         - Time points (_times)
#     """
#     assert (
#         imaging_extractor1.get_image_size() == imaging_extractor2.get_image_size()
#     ), "ImagingExtractors are not equal: image_sizes do not match."
#     assert (
#         imaging_extractor1.get_num_frames() == imaging_extractor2.get_num_frames()
#     ), "ImagingExtractors are not equal: num_frames do not match."
#     assert np.isclose(
#         imaging_extractor1.get_sampling_frequency(), imaging_extractor2.get_sampling_frequency()
#     ), "ImagingExtractors are not equal: sampling_frequencies do not match."
#     assert (
#         imaging_extractor1.get_dtype() == imaging_extractor2.get_dtype()
#     ), "ImagingExtractors are not equal: dtypes do not match."
#     assert_array_equal(
#         imaging_extractor1.get_video(),
#         imaging_extractor2.get_video(),
#         err_msg="ImagingExtractors are not equal: videos do not match.",
#     )
#     assert_array_equal(
#         imaging_extractor1._times,
#         imaging_extractor2._times,
#         err_msg="ImagingExtractors are not equal: _times do not match.",
#     )


def test_assert_imaging_equal_image_size():
    imaging_extractor1 = generate_mock_imaging_extractor(num_rows=1)
    imaging_extractor2 = generate_mock_imaging_extractor(num_rows=1)
    imaging_extractor3 = generate_mock_imaging_extractor(num_rows=2)
    assert_imaging_equal(imaging_extractor1, imaging_extractor2)
    with pytest.raises(AssertionError):
        assert_imaging_equal(imaging_extractor1, imaging_extractor3)


def test_assert_imaging_equal_num_frames():
    imaging_extractor1 = generate_mock_imaging_extractor(num_frames=1)
    imaging_extractor2 = generate_mock_imaging_extractor(num_frames=1)
    imaging_extractor3 = generate_mock_imaging_extractor(num_frames=2)
    assert_imaging_equal(imaging_extractor1, imaging_extractor2)
    with pytest.raises(AssertionError):
        assert_imaging_equal(imaging_extractor1, imaging_extractor3)


def test_assert_imaging_equal_sampling_frequency():
    imaging_extractor1 = generate_mock_imaging_extractor(sampling_frequency=30.0)
    imaging_extractor2 = generate_mock_imaging_extractor(sampling_frequency=30.0)
    imaging_extractor3 = generate_mock_imaging_extractor(sampling_frequency=20.0)
    assert_imaging_equal(imaging_extractor1, imaging_extractor2)
    with pytest.raises(AssertionError):
        assert_imaging_equal(imaging_extractor1, imaging_extractor3)


def test_assert_imaging_equal_dtype():
    imaging_extractor1 = generate_mock_imaging_extractor(dtype="uint16")
    imaging_extractor2 = generate_mock_imaging_extractor(dtype="uint16")
    imaging_extractor3 = generate_mock_imaging_extractor(dtype="float32")
    assert_imaging_equal(imaging_extractor1, imaging_extractor2)
    with pytest.raises(AssertionError):
        assert_imaging_equal(imaging_extractor1, imaging_extractor3)


def test_assert_imaging_equal_video():
    imaging_extractor1 = generate_mock_imaging_extractor(seed=0)
    imaging_extractor2 = generate_mock_imaging_extractor(seed=0)
    imaging_extractor3 = generate_mock_imaging_extractor(seed=1)
    assert_imaging_equal(imaging_extractor1, imaging_extractor2)
    with pytest.raises(AssertionError):
        assert_imaging_equal(imaging_extractor1, imaging_extractor3)


def test_assert_imaging_equal_times():
    imaging_extractor1 = generate_mock_imaging_extractor()
    imaging_extractor2 = generate_mock_imaging_extractor()
    imaging_extractor3 = generate_mock_imaging_extractor()
    imaging_extractor1._times = np.array([0, 1, 2])
    imaging_extractor2._times = np.array([0, 1, 2])
    imaging_extractor3._times = np.array([0, 1, 3])
    assert_imaging_equal(imaging_extractor1, imaging_extractor2)
    with pytest.raises(AssertionError):
        assert_imaging_equal(imaging_extractor1, imaging_extractor3)


def test_imaging_equal():
    imaging_extractor1 = generate_mock_imaging_extractor(seed=0)
    imaging_extractor2 = generate_mock_imaging_extractor(seed=0)
    imaging_extractor3 = generate_mock_imaging_extractor(seed=1)
    assert imaging_equal(imaging_extractor1, imaging_extractor2)
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


def test_assert_segmentation_equal_image_size():
    segmentation_extractor1 = generate_mock_segmentation_extractor(num_rows=1)
    segmentation_extractor2 = generate_mock_segmentation_extractor(num_rows=1)
    segmentation_extractor3 = generate_mock_segmentation_extractor(num_rows=2)
    assert_segmentation_equal(segmentation_extractor1, segmentation_extractor2)
    with pytest.raises(AssertionError):
        assert_segmentation_equal(segmentation_extractor1, segmentation_extractor3)


def test_assert_segmentation_equal_num_frames():
    segmentation_extractor1 = generate_mock_segmentation_extractor(num_frames=1)
    segmentation_extractor2 = generate_mock_segmentation_extractor(num_frames=1)
    segmentation_extractor3 = generate_mock_segmentation_extractor(num_frames=2)
    assert_segmentation_equal(segmentation_extractor1, segmentation_extractor2)
    with pytest.raises(AssertionError):
        assert_segmentation_equal(segmentation_extractor1, segmentation_extractor3)


def test_assert_segmentation_equal_sampling_frequency():
    segmentation_extractor1 = generate_mock_segmentation_extractor(sampling_frequency=30.0)
    segmentation_extractor2 = generate_mock_segmentation_extractor(sampling_frequency=30.0)
    segmentation_extractor3 = generate_mock_segmentation_extractor(sampling_frequency=20.0)
    assert_segmentation_equal(segmentation_extractor1, segmentation_extractor2)
    with pytest.raises(AssertionError):
        assert_segmentation_equal(segmentation_extractor1, segmentation_extractor3)


def test_assert_segmentation_equal_times():
    segmentation_extractor1 = generate_mock_segmentation_extractor()
    segmentation_extractor2 = generate_mock_segmentation_extractor()
    segmentation_extractor3 = generate_mock_segmentation_extractor()
    segmentation_extractor1._times = np.array([0, 1, 2])
    segmentation_extractor2._times = np.array([0, 1, 2])
    segmentation_extractor3._times = np.array([0, 1, 3])
    assert_segmentation_equal(segmentation_extractor1, segmentation_extractor2)
    with pytest.raises(AssertionError):
        assert_segmentation_equal(segmentation_extractor1, segmentation_extractor3)


def test_assert_segmentation_equal_roi_ids():
    segmentation_extractor1 = generate_mock_segmentation_extractor(num_rois=3)
    segmentation_extractor2 = generate_mock_segmentation_extractor(num_rois=3)
    segmentation_extractor3 = generate_mock_segmentation_extractor(num_rois=4)
    assert_segmentation_equal(segmentation_extractor1, segmentation_extractor2)
    with pytest.raises(AssertionError):
        assert_segmentation_equal(segmentation_extractor1, segmentation_extractor3)


def test_assert_segmentation_equal_accepted_rejected_roi_ids():
    segmentation_extractor1 = generate_mock_segmentation_extractor(rejected_roi_ids=[1])
    segmentation_extractor2 = generate_mock_segmentation_extractor(rejected_roi_ids=[1])
    segmentation_extractor3 = generate_mock_segmentation_extractor(rejected_roi_ids=[2])
    assert_segmentation_equal(segmentation_extractor1, segmentation_extractor2)
    with pytest.raises(AssertionError):
        assert_segmentation_equal(segmentation_extractor1, segmentation_extractor3)


def test_assert_segmentation_equal_roi_locations():
    roi_locations1 = np.array([[0, 1], [1, 2]])
    roi_locations2 = np.array([[0, 1], [1, 2]])
    roi_locations3 = np.array([[0, 1], [1, 3]])
    segmentation_extractor1 = generate_mock_segmentation_extractor(num_rois=2, roi_locations=roi_locations1)
    segmentation_extractor2 = generate_mock_segmentation_extractor(num_rois=2, roi_locations=roi_locations2)
    segmentation_extractor3 = generate_mock_segmentation_extractor(num_rois=2, roi_locations=roi_locations3)
    assert_segmentation_equal(segmentation_extractor1, segmentation_extractor2)
    with pytest.raises(AssertionError):
        assert_segmentation_equal(segmentation_extractor1, segmentation_extractor3)


def test_assert_segmentation_equal_roi_image_pixel_masks():
    image_masks1 = np.array([[[0, 1], [1, 2]], [[0, 1], [1, 2]]])
    image_masks2 = np.array([[[0, 1], [1, 2]], [[0, 1], [1, 2]]])
    image_masks3 = np.array([[[0, 1], [1, 2]], [[0, 1], [1, 3]]])
    segmentation_extractor1 = generate_mock_segmentation_extractor(
        image_masks=image_masks1, num_rois=2, num_rows=2, num_columns=2
    )
    segmentation_extractor2 = generate_mock_segmentation_extractor(
        image_masks=image_masks2, num_rois=2, num_rows=2, num_columns=2
    )
    segmentation_extractor3 = generate_mock_segmentation_extractor(
        image_masks=image_masks3, num_rois=2, num_rows=2, num_columns=2
    )
    assert_segmentation_equal(segmentation_extractor1, segmentation_extractor2)
    with pytest.raises(AssertionError):
        assert_segmentation_equal(segmentation_extractor1, segmentation_extractor3)


def test_assert_segmentation_equal_roi_response_traces():
    trace1 = np.array([[0, 1], [1, 2]])
    trace2 = np.array([[0, 1], [1, 2]])
    trace3 = np.array([[0, 1], [1, 3]])
    response_traces1 = {"raw": trace1}
    response_traces2 = {"raw": trace2}
    response_traces3 = {"raw": trace3}
    response_traces4 = {"raw": trace1, "dff": trace1}
    segmentation_extractor1 = generate_mock_segmentation_extractor(
        roi_response_traces=response_traces1, num_rois=2, num_frames=2
    )
    segmentation_extractor2 = generate_mock_segmentation_extractor(
        roi_response_traces=response_traces2, num_rois=2, num_frames=2
    )
    segmentation_extractor3 = generate_mock_segmentation_extractor(
        roi_response_traces=response_traces3, num_rois=2, num_frames=2
    )
    segmentation_extractor4 = generate_mock_segmentation_extractor(
        roi_response_traces=response_traces4, num_rois=2, num_frames=2
    )
    assert_segmentation_equal(segmentation_extractor1, segmentation_extractor2)
    with pytest.raises(AssertionError):
        assert_segmentation_equal(segmentation_extractor1, segmentation_extractor3)
    with pytest.raises(AssertionError):
        assert_segmentation_equal(segmentation_extractor1, segmentation_extractor4)


def test_assert_segmentation_equal_background_ids():
    segmentation_extractor1 = generate_mock_segmentation_extractor(num_background_components=2)
    segmentation_extractor2 = generate_mock_segmentation_extractor(num_background_components=2)
    segmentation_extractor3 = generate_mock_segmentation_extractor(num_background_components=3)
    assert_segmentation_equal(segmentation_extractor1, segmentation_extractor2)
    with pytest.raises(AssertionError):
        assert_segmentation_equal(segmentation_extractor1, segmentation_extractor3)


def test_assert_segmentation_equal_background_image_masks():
    image_masks1 = np.array([[[0, 1], [1, 2]], [[0, 1], [1, 2]]])
    image_masks2 = np.array([[[0, 1], [1, 2]], [[0, 1], [1, 2]]])
    image_masks3 = np.array([[[0, 1], [1, 2]], [[0, 1], [1, 3]]])
    segmentation_extractor1 = generate_mock_segmentation_extractor(
        background_image_masks=image_masks1, num_background_components=2, num_rows=2, num_columns=2
    )
    segmentation_extractor2 = generate_mock_segmentation_extractor(
        background_image_masks=image_masks2, num_background_components=2, num_rows=2, num_columns=2
    )
    segmentation_extractor3 = generate_mock_segmentation_extractor(
        background_image_masks=image_masks3, num_background_components=2, num_rows=2, num_columns=2
    )
    assert_segmentation_equal(segmentation_extractor1, segmentation_extractor2)
    with pytest.raises(AssertionError):
        assert_segmentation_equal(segmentation_extractor1, segmentation_extractor3)


def test_assert_segmentation_equal_background_response_traces():
    trace1 = np.array([[0, 1], [1, 2]])
    trace2 = np.array([[0, 1], [1, 2]])
    trace3 = np.array([[0, 1], [1, 3]])
    response_traces1 = {"background": trace1}
    response_traces2 = {"background": trace2}
    response_traces3 = {"background": trace3}
    response_traces4 = {"background": trace1, "dff": trace1}
    segmentation_extractor1 = generate_mock_segmentation_extractor(
        background_response_traces=response_traces1, num_background_components=2, num_frames=2
    )
    segmentation_extractor2 = generate_mock_segmentation_extractor(
        background_response_traces=response_traces2, num_background_components=2, num_frames=2
    )
    segmentation_extractor3 = generate_mock_segmentation_extractor(
        background_response_traces=response_traces3, num_background_components=2, num_frames=2
    )
    segmentation_extractor4 = generate_mock_segmentation_extractor(
        background_response_traces=response_traces4, num_background_components=2, num_frames=2
    )
    assert_segmentation_equal(segmentation_extractor1, segmentation_extractor2)
    with pytest.raises(AssertionError):
        assert_segmentation_equal(segmentation_extractor1, segmentation_extractor3)
    with pytest.raises(AssertionError):
        assert_segmentation_equal(segmentation_extractor1, segmentation_extractor4)


def test_segmentation_equal():
    segmentation_extractor1 = generate_mock_segmentation_extractor(seed=0)
    segmentation_extractor2 = generate_mock_segmentation_extractor(seed=0)
    segmentation_extractor3 = generate_mock_segmentation_extractor(seed=1)
    assert segmentation_equal(segmentation_extractor1, segmentation_extractor2)
    assert not segmentation_equal(segmentation_extractor1, segmentation_extractor3)
