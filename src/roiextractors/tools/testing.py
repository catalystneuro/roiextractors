"""Testing utilities for the roiextractors package."""

from collections.abc import Iterable
from typing import Tuple, Optional, List

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..segmentationextractor import SegmentationExtractor
from ..imagingextractor import ImagingExtractor

from roiextractors import NumpyImagingExtractor, NumpySegmentationExtractor

from roiextractors.tools.typing import DtypeType, ArrayType


def generate_mock_video(size: Tuple[int], dtype: DtypeType = "uint16", seed: int = 0):
    """Generate a mock video of a given size and dtype.

    Parameters
    ----------
    size : Tuple[int]
        Size of the video to generate.
    dtype : DtypeType, optional
        Dtype of the video to generate, by default "uint16".
    seed : int, default 0
        seed for the random number generator, by default 0.

    Returns
    -------
    video : np.ndarray
        A mock video of the given size and dtype.
    """
    dtype = np.dtype(dtype)
    number_of_bytes = dtype.itemsize

    rng = np.random.default_rng(seed)

    low = 0 if "u" in dtype.name else 2 ** (number_of_bytes - 1) - 2**number_of_bytes
    high = 2**number_of_bytes - 1 if "u" in dtype.name else 2**number_of_bytes - 2 ** (number_of_bytes - 1) - 1
    video = (
        rng.random(size=size, dtype=dtype)
        if "float" in dtype.name
        else rng.integers(low=low, high=high, size=size, dtype=dtype)
    )

    return video


def generate_mock_imaging_extractor(
    num_frames: int = 30,
    num_rows: int = 10,
    num_columns: int = 10,
    sampling_frequency: float = 30.0,
    dtype: DtypeType = "uint16",
    seed: int = 0,
):
    """Generate a mock imaging extractor for testing.

    The imaging extractor is built by feeding random data into the `NumpyImagingExtractor`.

    Parameters
    ----------
    num_frames : int, optional
        number of frames in the video, by default 30.
    num_rows : int, optional
        number of rows in the video, by default 10.
    num_columns : int, optional
        number of columns in the video, by default 10.
    sampling_frequency : float, optional
        sampling frequency of the video, by default 30.
    dtype : DtypeType, optional
        dtype of the video, by default "uint16".
    seed : int, default 0
        seed for the random number generator, by default 0.

    Returns
    -------
    NumpyImagingExtractor
        An imaging extractor with random data fed into `NumpyImagingExtractor`.
    """
    size = (num_frames, num_rows, num_columns)
    video = generate_mock_video(size=size, dtype=dtype, seed=seed)
    imaging_extractor = NumpyImagingExtractor(timeseries=video, sampling_frequency=sampling_frequency)

    return imaging_extractor


def assert_imaging_equal(imaging_extractor1: ImagingExtractor, imaging_extractor2: ImagingExtractor):
    """Assert that two ImagingExtractor objects are equal by comparing their attributes and data.

    Parameters
    ----------
    imaging_extractor1 : ImagingExtractor
        The first ImagingExtractor object to compare.
    imaging_extractor2 : ImagingExtractor
        The second ImagingExtractor object to compare.

    Raises
    ------
    AssertionError
        If any of the following attributes or data do not match between the two ImagingExtractor objects:
        - Image size
        - Number of frames
        - Sampling frequency
        - Data type (dtype)
        - Video data
        - Time points (_times)
    """
    assert (
        imaging_extractor1.get_image_size() == imaging_extractor2.get_image_size()
    ), "ImagingExtractors are not equal: image_sizes do not match."
    assert (
        imaging_extractor1.get_num_frames() == imaging_extractor2.get_num_frames()
    ), "ImagingExtractors are not equal: num_frames do not match."
    assert np.isclose(
        imaging_extractor1.get_sampling_frequency(), imaging_extractor2.get_sampling_frequency()
    ), "ImagingExtractors are not equal: sampling_frequencies do not match."
    assert (
        imaging_extractor1.get_dtype() == imaging_extractor2.get_dtype()
    ), "ImagingExtractors are not equal: dtypes do not match."
    assert_array_equal(
        imaging_extractor1.get_video(),
        imaging_extractor2.get_video(),
        err_msg="ImagingExtractors are not equal: videos do not match.",
    )
    assert_array_equal(
        imaging_extractor1._times,
        imaging_extractor2._times,
        err_msg="ImagingExtractors are not equal: _times do not match.",
    )


def imaging_equal(imaging_extractor1: ImagingExtractor, imaging_extractor2: ImagingExtractor) -> bool:
    """Return True if two ImagingExtractors are equal, False otherwise.

    Parameters
    ----------
    imaging_extractor1 : ImagingExtractor
        The first ImagingExtractor object to compare.
    imaging_extractor2 : ImagingExtractor
        The second ImagingExtractor object to compare.

    Returns
    -------
    bool
        True if all of the following fields match between the two ImagingExtractor objects:
        - Image size
        - Number of frames
        - Sampling frequency
        - Data type (dtype)
        - Video data
        - Time points (_times)
    """
    try:
        assert_imaging_equal(imaging_extractor1, imaging_extractor2)
        return True
    except AssertionError:
        return False


def generate_mock_segmentation_extractor(
    num_rois: int = 10,
    num_frames: int = 30,
    num_rows: int = 25,
    num_columns: int = 25,
    num_background_components: int = 2,
    sampling_frequency: float = 30.0,
    summary_image_names: List[str] = ["mean", "correlation"],
    roi_response_names: List[str] = ["raw", "dff", "deconvolved", "denoised"],
    background_response_names: List[str] = ["background"],
    rejected_roi_ids: Optional[list] = None,
    roi_locations: Optional[ArrayType] = None,
    image_masks: Optional[ArrayType] = None,
    roi_response_traces: Optional[dict] = None,
    background_image_masks: Optional[ArrayType] = None,
    background_response_traces: Optional[dict] = None,
    summary_images: Optional[dict] = None,
    seed: int = 0,
) -> NumpySegmentationExtractor:
    """Generate a mock segmentation extractor for testing.

    The segmentation extractor is built by feeding random data into the
    `NumpySegmentationExtractor`.

    Parameters
    ----------
    num_rois : int, optional
        number of regions of interest, by default 10.
    num_frames : int, optional
       Number of frames in the recording, by default 30.
    num_rows : int, optional
        number of rows in the hypothetical video from which the data was extracted, by default 25.
    num_columns : int, optional
        number of columns in the hypothetical video from which the data was extracted, by default 25.
    num_background_components : int, optional
        number of background components, by default 2.
    sampling_frequency : float, optional
        sampling frequency of the hypothetical video from which the data was extracted, by default 30.0.
    summary_image_names : List[str], optional
        names of summary images, by default ["mean", "correlation"].
    roi_response_names : List[str], optional
        names of roi response traces, by default ["raw", "dff", "deconvolved", "denoised"].
    background_response_names : List[str], optional
        names of background response traces, by default ["background"].
    rejected_roi_ids: Optional[list], optional
        A list of rejected rois, None by default.
    roi_locations : Optional[ArrayType], optional
        A 2D array of shape (2, num_rois) containing the locations of the rois, None by default.
    seed : int, default 0
        seed for the random number generator, by default 0.

    Returns
    -------
    NumpySegmentationExtractor
        A segmentation extractor with random data fed into `NumpySegmentationExtractor`

    Notes
    -----
    Note that this dummy example is meant to be a mock object with the right shape, structure and objects but does not
    contain meaningful content. That is, the image masks matrices are not plausible image mask for a roi, the raw signal
    is not a meaningful biological signal and is not related appropriately to the deconvolved signal , etc.
    """
    rng = np.random.default_rng(seed)

    # Create dummy image masks
    if image_masks is None:
        image_masks = rng.random((num_rows, num_columns, num_rois))
    else:
        assert image_masks.shape == (
            num_rows,
            num_columns,
            num_rois,
        ), f"image_masks should have shape (num_rows, num_columns, num_rois) but got {image_masks.shape}."
    if background_image_masks is None:
        background_image_masks = rng.random((num_rows, num_columns, num_background_components))
    else:
        assert background_image_masks.shape == (
            num_rows,
            num_columns,
            num_background_components,
        ), f"background_image_masks should have shape (num_rows, num_columns, num_background_components) but got {background_image_masks.shape}."

    # Create signals
    if roi_response_traces is None:
        roi_response_traces = {name: rng.random((num_frames, num_rois)) for name in roi_response_names}
    else:
        for name, trace in roi_response_traces.items():
            assert trace.shape == (
                num_frames,
                num_rois,
            ), f"roi_response_traces[{name}] should have shape (num_frames, num_rois) but got {trace.shape}."
    if background_response_traces is None:
        background_response_traces = {
            name: rng.random((num_frames, num_background_components)) for name in background_response_names
        }
    else:
        for name, trace in background_response_traces.items():
            assert trace.shape == (
                num_frames,
                num_background_components,
            ), f"background_response_traces[{name}] should have shape (num_frames, num_background_components) but got {trace.shape}."

    # Summary images
    if summary_images is None:
        summary_images = {name: rng.random((num_rows, num_columns)) for name in summary_image_names}
    else:
        for name, image in summary_images.items():
            assert image.shape == (
                num_rows,
                num_columns,
            ), f"summary_images[{name}] should have shape (num_rows, num_columns) but got {image.shape}."

    # Rois
    roi_ids = [id for id in range(num_rois)]
    if roi_locations is None:
        roi_locations_rows = rng.integers(low=0, high=num_rows, size=num_rois)
        roi_locations_columns = rng.integers(low=0, high=num_columns, size=num_rois)
        roi_locations = np.vstack((roi_locations_rows, roi_locations_columns))
    else:
        assert roi_locations.shape == (
            2,
            num_rois,
        ), f"roi_locations should have shape (2, num_rois) but got {roi_locations.shape}."
    background_ids = [i for i in range(num_background_components)]

    rejected_roi_ids = rejected_roi_ids if rejected_roi_ids else None

    accepted_roi_ids = roi_ids
    if rejected_roi_ids is not None:
        accepted_roi_ids = list(set(accepted_roi_ids).difference(rejected_roi_ids))

    dummy_segmentation_extractor = NumpySegmentationExtractor(
        image_masks=image_masks,
        roi_response_traces=roi_response_traces,
        sampling_frequency=sampling_frequency,
        roi_ids=roi_ids,
        accepted_roi_ids=accepted_roi_ids,
        rejected_roi_ids=rejected_roi_ids,
        roi_locations=roi_locations,
        summary_images=summary_images,
        background_image_masks=background_image_masks,
        background_response_traces=background_response_traces,
        background_ids=background_ids,
    )

    return dummy_segmentation_extractor


def assert_segmentation_equal(
    segmentation_extractor1: SegmentationExtractor, segmentation_extractor2: SegmentationExtractor
):
    """Assert that two segmentation extractors have equal fields.

    Parameters
    ----------
    segmentation_extractor1 : SegmentationExtractor
        First segmentation extractor to compare.
    segmentation_extractor2 : SegmentationExtractor
        Second segmentation extractor to compare.

    Raises
    ------
    AssertionError
        If any of the following attributes or data do not match between the two SegmentationExtractor objects:
        - image_size
        - num_frames
        - sampling_frequency
        - _times
        - roi_ids
        - num_rois
        - accepted_roi_ids
        - rejected_roi_ids
        - roi_locations
        - roi_image_masks
        - roi_pixel_masks
        - roi_response_traces
        - background_ids
        - num_background_components
        - background_image_masks
        - background_response_traces
        - summary_images
    """
    assert (
        segmentation_extractor1.get_image_size() == segmentation_extractor2.get_image_size()
    ), "SegmentationExtractors are not equal: image_sizes do not match."
    assert (
        segmentation_extractor1.get_num_frames() == segmentation_extractor2.get_num_frames()
    ), "SegmentationExtractors are not equal: num_frames do not match."
    assert (
        segmentation_extractor1.get_sampling_frequency() == segmentation_extractor2.get_sampling_frequency()
    ), "SegmentationExtractors are not equal: sampling_frequencies do not match."
    assert_array_equal(
        segmentation_extractor1._times,
        segmentation_extractor2._times,
        err_msg="SegmentationExtractors are not equal: _times do not match.",
    )
    assert_array_equal(
        segmentation_extractor1.get_roi_ids(),
        segmentation_extractor2.get_roi_ids(),
        err_msg="SegmentationExtractors are not equal: roi_ids do not match.",
    )
    assert (
        segmentation_extractor1.get_num_rois() == segmentation_extractor2.get_num_rois()
    ), "SegmentationExtractors are not equal: num_rois do not match."
    assert_array_equal(
        segmentation_extractor1.get_accepted_roi_ids(),
        segmentation_extractor2.get_accepted_roi_ids(),
        err_msg="SegmentationExtractors are not equal: accepted_roi_ids do not match.",
    )
    assert_array_equal(
        segmentation_extractor1.get_rejected_roi_ids(),
        segmentation_extractor2.get_rejected_roi_ids(),
        err_msg="SegmentationExtractors are not equal: rejected_roi_ids do not match.",
    )
    assert_array_equal(
        segmentation_extractor1.get_roi_locations(),
        segmentation_extractor2.get_roi_locations(),
        err_msg="SegmentationExtractors are not equal: roi_locations do not match.",
    )
    assert_array_equal(
        segmentation_extractor1.get_roi_image_masks(),
        segmentation_extractor2.get_roi_image_masks(),
        err_msg="SegmentationExtractors are not equal: roi_image_masks do not match.",
    )
    for pixel_mask1, pixel_mask2 in zip(
        segmentation_extractor1.get_roi_pixel_masks(), segmentation_extractor2.get_roi_pixel_masks()
    ):
        assert_array_equal(
            pixel_mask1,
            pixel_mask2,
            err_msg="SegmentationExtractors are not equal: roi_pixel_masks do not match.",
        )
    roi_response_traces1 = segmentation_extractor1.get_roi_response_traces()
    roi_response_traces2 = segmentation_extractor2.get_roi_response_traces()
    for name, trace1 in roi_response_traces1.items():
        assert (
            name in roi_response_traces2
        ), f"SegmentationExtractors are not equal: SegmentationExtractor1 has roi_response_trace {name} but SegmentationExtractor2 does not."
        trace2 = roi_response_traces2[name]
        assert_array_equal(
            trace1,
            trace2,
            err_msg=f"SegmentationExtractors are not equal: roi_response_trace {name} does not match.",
        )
    for name2 in roi_response_traces2:
        assert (
            name2 in roi_response_traces1
        ), f"SegmentationExtractors are not equal: SegmentationExtractor2 has roi_response_trace {name2} but SegmentationExtractor1 does not."
    assert_array_equal(
        segmentation_extractor1.get_background_ids(),
        segmentation_extractor2.get_background_ids(),
        err_msg="SegmentationExtractors are not equal: background_ids do not match.",
    )
    assert (
        segmentation_extractor1.get_num_background_components()
        == segmentation_extractor2.get_num_background_components()
    ), "SegmentationExtractors are not equal: num_background_components do not match."
    assert_array_equal(
        segmentation_extractor1.get_background_image_masks(),
        segmentation_extractor2.get_background_image_masks(),
        err_msg="SegmentationExtractors are not equal: background_image_masks do not match.",
    )
    background_response_traces1 = segmentation_extractor1.get_background_response_traces()
    background_response_traces2 = segmentation_extractor2.get_background_response_traces()
    for name, trace1 in background_response_traces1.items():
        assert (
            name in background_response_traces2
        ), f"SegmentationExtractors are not equal: SegmentationExtractor1 has background_response_trace {name} but SegmentationExtractor2 does not."
        trace2 = background_response_traces2[name]
        assert_array_equal(
            trace1,
            trace2,
            err_msg=f"SegmentationExtractors are not equal: background_response_trace {name} does not match.",
        )
    for name2 in background_response_traces2:
        assert (
            name2 in background_response_traces1
        ), f"SegmentationExtractors are not equal: SegmentationExtractor2 has background_response_trace {name2} but SegmentationExtractor1 does not."
    summary_images1 = segmentation_extractor1.get_summary_images()
    summary_images2 = segmentation_extractor2.get_summary_images()
    for name, image1 in summary_images1.items():
        assert (
            name in summary_images2
        ), f"SegmentationExtractors are not equal: SegmentationExtractor1 has summary_image {name} but SegmentationExtractor2 does not."
        image2 = summary_images2[name]
        assert_array_equal(
            image1,
            image2,
            err_msg=f"SegmentationExtractors are not equal: summary_image {name} does not match.",
        )
    for name2 in summary_images2:
        assert (
            name2 in summary_images1
        ), f"SegmentationExtractors are not equal: SegmentationExtractor2 has summary_image {name2} but SegmentationExtractor1 does not."


def segmentation_equal(
    segmentation_extractor1: SegmentationExtractor, segmentation_extractor2: SegmentationExtractor
) -> bool:
    """Return True if two SegmentationExtractors have equal fields, False otherwise.

    Parameters
    ----------
    segmentation_extractor1 : SegmentationExtractor
        First segmentation extractor to compare.
    segmentation_extractor2 : SegmentationExtractor
        Second segmentation extractor to compare.

    Returns
    -------
    bool
        True if all of the following fields match between the two SegmentationExtractor objects:
        - image_size
        - num_frames
        - sampling_frequency
        - _times
        - roi_ids
        - num_rois
        - accepted_roi_ids
        - rejected_roi_ids
        - roi_locations
        - roi_image_masks
        - roi_pixel_masks
        - roi_response_traces
        - background_ids
        - num_background_components
        - background_image_masks
        - background_response_traces
        - summary_images
    """
    try:
        assert_segmentation_equal(segmentation_extractor1, segmentation_extractor2)
        return True
    except AssertionError:
        return False
