"""Testing utilities for the roiextractors package."""

from collections.abc import Iterable
from typing import Tuple, Optional, List

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..segmentationextractor import SegmentationExtractor
from ..imagingextractor import ImagingExtractor

from roiextractors import NumpyImagingExtractor, NumpySegmentationExtractor

from roiextractors.tools.typing import DtypeType, NoneType, FloatType, IntType


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
    image_masks = rng.random((num_rows, num_columns, num_rois))
    background_image_masks = rng.random((num_rows, num_columns, num_background_components))

    # Create signals
    roi_response_traces = {name: rng.random((num_frames, num_rois)) for name in roi_response_names}
    background_response_traces = {
        name: rng.random((num_frames, num_background_components)) for name in background_response_names
    }

    # Summary images
    summary_images = {name: rng.random((num_rows, num_columns)) for name in summary_image_names}

    # Rois
    roi_ids = [id for id in range(num_rois)]
    roi_locations_rows = rng.integers(low=0, high=num_rows, size=num_rois)
    roi_locations_columns = rng.integers(low=0, high=num_columns, size=num_rois)
    roi_locations = np.vstack((roi_locations_rows, roi_locations_columns))
    background_ids = [i for i in range(num_background_components)]

    rejected_roi_ids = rejected_roi_ids if rejected_roi_ids else None

    accepeted_list = roi_ids
    if rejected_roi_ids is not None:
        accepeted_list = list(set(accepeted_list).difference(rejected_roi_ids))

    dummy_segmentation_extractor = NumpySegmentationExtractor(
        image_masks=image_masks,
        roi_response_traces=roi_response_traces,
        sampling_frequency=sampling_frequency,
        roi_ids=roi_ids,
        accepted_roi_ids=accepeted_list,
        rejected_roi_ids=rejected_roi_ids,
        roi_locations=roi_locations,
        summary_images=summary_images,
        background_image_masks=background_image_masks,
        background_response_traces=background_response_traces,
        background_ids=background_ids,
    )

    return dummy_segmentation_extractor


def check_segmentations_equal(
    segmentation_extractor1: SegmentationExtractor, segmentation_extractor2: SegmentationExtractor
):
    """Check that two segmentation extractors have equal fields."""
    check_segmentation_return_types(segmentation_extractor1)
    check_segmentation_return_types(segmentation_extractor2)
    # assert equality:
    assert segmentation_extractor1.get_num_rois() == segmentation_extractor2.get_num_rois()
    assert segmentation_extractor1.get_num_frames() == segmentation_extractor2.get_num_frames()
    assert segmentation_extractor1.get_num_channels() == segmentation_extractor2.get_num_channels()
    assert np.isclose(
        segmentation_extractor1.get_sampling_frequency(), segmentation_extractor2.get_sampling_frequency()
    )
    assert_array_equal(segmentation_extractor1.get_channel_names(), segmentation_extractor2.get_channel_names())
    assert_array_equal(segmentation_extractor1.get_image_size(), segmentation_extractor2.get_image_size())
    assert_array_equal(
        segmentation_extractor1.get_roi_image_masks(roi_ids=segmentation_extractor1.get_roi_ids()[:1]),
        segmentation_extractor2.get_roi_image_masks(roi_ids=segmentation_extractor2.get_roi_ids()[:1]),
    )
    assert set(
        segmentation_extractor1.get_roi_pixel_masks(roi_ids=segmentation_extractor1.get_roi_ids()[:1])[0].flatten()
    ) == set(
        segmentation_extractor2.get_roi_pixel_masks(roi_ids=segmentation_extractor1.get_roi_ids()[:1])[0].flatten()
    )

    check_segmentations_images(segmentation_extractor1, segmentation_extractor2)

    assert_array_equal(segmentation_extractor1.get_accepted_list(), segmentation_extractor2.get_accepted_list())
    assert_array_equal(segmentation_extractor1.get_rejected_list(), segmentation_extractor2.get_rejected_list())
    assert_array_equal(segmentation_extractor1.get_roi_locations(), segmentation_extractor2.get_roi_locations())
    assert_array_equal(segmentation_extractor1.get_roi_ids(), segmentation_extractor2.get_roi_ids())
    assert_array_equal(segmentation_extractor1.get_traces(), segmentation_extractor2.get_traces())

    assert_array_equal(
        segmentation_extractor1.frame_to_time(np.arange(segmentation_extractor1.get_num_frames())),
        segmentation_extractor2.frame_to_time(np.arange(segmentation_extractor2.get_num_frames())),
    )


def check_imaging_equal(imaging_extractor1: ImagingExtractor, imaging_extractor2: ImagingExtractor):
    """Check that two imaging extractors have equal fields."""
    assert imaging_extractor1.get_image_size() == imaging_extractor2.get_image_size()
    assert imaging_extractor1.get_num_frames() == imaging_extractor2.get_num_frames()
    assert np.close(imaging_extractor1.get_sampling_frequency(), imaging_extractor2.get_sampling_frequency())
    assert imaging_extractor1.get_dtype() == imaging_extractor2.get_dtype()
    assert_array_equal(imaging_extractor1.get_video(), imaging_extractor2.get_video())
    assert_array_almost_equal(
        imaging_extractor1.frame_to_time(np.arange(imaging_extractor1.get_num_frames())),
        imaging_extractor2.frame_to_time(np.arange(imaging_extractor2.get_num_frames())),
    )
