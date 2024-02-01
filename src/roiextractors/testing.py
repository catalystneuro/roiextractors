"""Testing utilities for the roiextractors package."""

from collections.abc import Iterable
from typing import Tuple, Optional

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from .segmentationextractor import SegmentationExtractor
from .imagingextractor import ImagingExtractor

from roiextractors import NumpyImagingExtractor, NumpySegmentationExtractor

from roiextractors.extraction_tools import DtypeType

NoneType = type(None)
floattype = (float, np.floating)
inttype = (int, np.integer)


def generate_dummy_video(size: Tuple[int], dtype: DtypeType = "uint16"):
    """Generate a dummy video of a given size and dtype.

    Parameters
    ----------
    size : Tuple[int]
        Size of the video to generate.
    dtype : DtypeType, optional
        Dtype of the video to generate, by default "uint16".

    Returns
    -------
    video : np.ndarray
        A dummy video of the given size and dtype.
    """
    dtype = np.dtype(dtype)
    number_of_bytes = dtype.itemsize

    low = 0 if "u" in dtype.name else 2 ** (number_of_bytes - 1) - 2**number_of_bytes
    high = 2**number_of_bytes - 1 if "u" in dtype.name else 2**number_of_bytes - 2 ** (number_of_bytes - 1) - 1
    video = (
        np.random.random(size=size)
        if "float" in dtype.name
        else np.random.randint(low=low, high=high, size=size, dtype=dtype)
    )

    return video


def generate_dummy_imaging_extractor(
    num_frames: int = 30,
    num_rows: int = 10,
    num_columns: int = 10,
    num_channels: int = 1,
    sampling_frequency: float = 30,
    dtype: DtypeType = "uint16",
    channel_names: Optional[list] = None,
):
    """Generate a dummy imaging extractor for testing.

    The imaging extractor is built by feeding random data into the `NumpyImagingExtractor`.

    Parameters
    ----------
    num_frames : int, optional
        number of frames in the video, by default 30.
    num_rows : int, optional
        number of rows in the video, by default 10.
    num_columns : int, optional
        number of columns in the video, by default 10.
    num_channels : int, optional
        number of channels in the video, by default 1.
    sampling_frequency : float, optional
        sampling frequency of the video, by default 30.
    dtype : DtypeType, optional
        dtype of the video, by default "uint16".

    Returns
    -------
    ImagingExtractor
        An imaging extractor with random data fed into `NumpyImagingExtractor`.
    """
    if channel_names is None:
        channel_names = [f"channel_num_{num}" for num in range(num_channels)]

    size = (num_frames, num_rows, num_columns, num_channels)
    video = generate_dummy_video(size=size, dtype=dtype)

    imaging_extractor = NumpyImagingExtractor(
        timeseries=video, sampling_frequency=sampling_frequency, channel_names=channel_names
    )

    return imaging_extractor


def generate_dummy_segmentation_extractor(
    num_rois: int = 10,
    num_frames: int = 30,
    num_rows: int = 25,
    num_columns: int = 25,
    sampling_frequency: float = 30.0,
    has_summary_images: bool = True,
    has_raw_signal: bool = True,
    has_dff_signal: bool = True,
    has_deconvolved_signal: bool = True,
    has_neuropil_signal: bool = True,
    rejected_list: Optional[list] = None,
) -> SegmentationExtractor:
    """Generate a dummy segmentation extractor for testing.

    The segmentation extractor is built by feeding random data into the
    `NumpySegmentationExtractor`.

    Parameters
    ----------
    num_rois : int, optional
        number of regions of interest, by default 10.
    num_frames : int, optional
        _description_, by default 30
    num_rows : number of frames used in the hypotethical video from which the data was extracted, optional
        number of rows in the hypotethical video from which the data was extracted, by default 25.
    num_columns : int, optional
        numbe rof columns in the hypotethical video from which the data was extracted, by default 25.
    sampling_frequency : float, optional
        sampling frequency of the hypotethical video form which the data was extracted, by default 30.0.
    has_summary_images : bool, optional
        whether the dummy segmentation extractor has summary images or not (mean and correlation)
    has_raw_signal : bool, optional
        whether a raw fluoresence signal is desired in the object, by default True.
    has_dff_signal : bool, optional
        whether a relative (df/f) fluoresence signal is desired in the object, by default True.
    has_deconvolved_signal : bool, optional
        whether a deconvolved signal is desired in the object, by default True.
    has_neuropil_signal : bool, optional
        whether a neuropil signal is desiredi n the object, by default True.
    rejected_list: list, optional
        A list of rejected rois, None by default.

    Returns
    -------
    SegmentationExtractor
        A segmentation extractor with random data fed into `NumpySegmentationExtractor`

    Notes
    -----
    Note that this dummy example is meant to be a mock object with the right shape, structure and objects but does not
    contain meaningful content. That is, the image masks matrices are not plausible image mask for a roi, the raw signal
    is not a meaningful biological signal and is not related appropriately to the deconvolved signal , etc.
    """
    # Create dummy image masks
    image_masks = np.random.rand(num_rows, num_columns, num_rois)
    movie_dims = (num_rows, num_columns)

    # Create signals
    raw = np.random.rand(num_frames, num_rois) if has_raw_signal else None
    dff = np.random.rand(num_frames, num_rois) if has_dff_signal else None
    deconvolved = np.random.rand(num_frames, num_rois) if has_deconvolved_signal else None
    neuropil = np.random.rand(num_frames, num_rois) if has_neuropil_signal else None

    # Summary images
    mean_image = np.random.rand(num_rows, num_columns) if has_summary_images else None
    correlation_image = np.random.rand(num_rows, num_columns) if has_summary_images else None

    # Rois
    roi_ids = [id for id in range(num_rois)]
    roi_locations_rows = np.random.randint(low=0, high=num_rows, size=num_rois)
    roi_locations_columns = np.random.randint(low=0, high=num_columns, size=num_rois)
    roi_locations = np.vstack((roi_locations_rows, roi_locations_columns))

    rejected_list = rejected_list if rejected_list else None

    accepeted_list = roi_ids
    if rejected_list is not None:
        accepeted_list = list(set(accepeted_list).difference(rejected_list))

    dummy_segmentation_extractor = NumpySegmentationExtractor(
        sampling_frequency=sampling_frequency,
        image_masks=image_masks,
        raw=raw,
        dff=dff,
        deconvolved=deconvolved,
        neuropil=neuropil,
        mean_image=mean_image,
        correlation_image=correlation_image,
        roi_ids=roi_ids,
        roi_locations=roi_locations,
        accepted_lst=accepeted_list,
        rejected_list=rejected_list,
        movie_dims=movie_dims,
        channel_names=["channel_num_0"],
    )

    return dummy_segmentation_extractor


def _assert_iterable_shape(iterable, shape):
    """Assert that the iterable has the given shape. If the iterable is a numpy array, the shape is checked directly."""
    ar = iterable if isinstance(iterable, np.ndarray) else np.array(iterable)
    for ar_shape, given_shape in zip(ar.shape, shape):
        if isinstance(given_shape, int):
            assert ar_shape == given_shape, f"Expected {given_shape}, received {ar_shape}!"


def _assert_iterable_shape_max(iterable, shape_max):
    """Assert that the iterable has a shape less than or equal to the given maximum shape."""
    ar = iterable if isinstance(iterable, np.ndarray) else np.array(iterable)
    for ar_shape, given_shape in zip(ar.shape, shape_max):
        if isinstance(given_shape, int):
            assert ar_shape <= given_shape


def _assert_iterable_element_dtypes(iterable, dtypes):
    """Assert that the iterable has elements of the given dtypes."""
    if isinstance(iterable, Iterable) and not isinstance(iterable, str):
        for iter in iterable:
            _assert_iterable_element_dtypes(iter, dtypes)
    else:
        assert isinstance(iterable, dtypes), f"array is none of the types {dtypes}"


def _assert_iterable_complete(iterable, dtypes=None, element_dtypes=None, shape=None, shape_max=None):
    """Assert that the iterable is complete, i.e. it is not None and has the given dtypes, element_dtypes, shape and shape_max."""
    assert isinstance(iterable, dtypes), f"iterable {type(iterable)} is none of the types {dtypes}"
    if not isinstance(iterable, NoneType):
        if shape is not None:
            _assert_iterable_shape(iterable, shape=shape)
        if shape_max is not None:
            _assert_iterable_shape_max(iterable, shape_max=shape_max)
        if element_dtypes is not None:
            _assert_iterable_element_dtypes(iterable, element_dtypes)


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


def check_segmentations_images(
    segmentation_extractor1: SegmentationExtractor,
    segmentation_extractor2: SegmentationExtractor,
):
    """Check that the segmentation images are equal for the given segmentation extractors."""
    images_in_extractor1 = segmentation_extractor1.get_images_dict()
    images_in_extractor2 = segmentation_extractor2.get_images_dict()

    assert len(images_in_extractor1) == len(images_in_extractor2)

    image_names_are_equal = all(image_name in images_in_extractor1.keys() for image_name in images_in_extractor2.keys())
    assert image_names_are_equal, "The names of segmentation images in the segmentation extractors are not the same."

    for image_name in images_in_extractor1.keys():
        assert_array_equal(
            images_in_extractor1[image_name],
            images_in_extractor2[image_name],
        ), f"The segmentation images for {image_name} are not equal."


def check_segmentation_return_types(seg: SegmentationExtractor):
    """Check that the return types of the segmentation extractor are correct."""
    assert isinstance(seg.get_num_rois(), int)
    assert isinstance(seg.get_num_frames(), int)
    assert isinstance(seg.get_num_channels(), int)
    assert isinstance(seg.get_sampling_frequency(), (NoneType, floattype))
    _assert_iterable_complete(
        seg.get_channel_names(),
        dtypes=list,
        element_dtypes=str,
        shape_max=(seg.get_num_channels(),),
    )
    _assert_iterable_complete(seg.get_image_size(), dtypes=Iterable, element_dtypes=inttype, shape=(2,))
    _assert_iterable_complete(
        seg.get_roi_image_masks(roi_ids=seg.get_roi_ids()[:1]),
        dtypes=(np.ndarray,),
        element_dtypes=floattype,
        shape=(*seg.get_image_size(), 1),
    )
    _assert_iterable_complete(
        seg.get_roi_ids(),
        dtypes=(list,),
        shape=(seg.get_num_rois(),),
        element_dtypes=inttype,
    )
    assert isinstance(seg.get_roi_pixel_masks(roi_ids=seg.get_roi_ids()[:2]), list)
    _assert_iterable_complete(
        seg.get_roi_pixel_masks(roi_ids=seg.get_roi_ids()[:1])[0],
        dtypes=(np.ndarray,),
        element_dtypes=floattype,
        shape_max=(np.prod(seg.get_image_size()), 3),
    )
    for image_name in seg.get_images_dict():
        _assert_iterable_complete(
            seg.get_image(image_name),
            dtypes=(np.ndarray, NoneType),
            element_dtypes=floattype,
            shape_max=(*seg.get_image_size(),),
        )
    _assert_iterable_complete(
        seg.get_accepted_list(),
        dtypes=(list, NoneType),
        element_dtypes=inttype,
        shape_max=(seg.get_num_rois(),),
    )
    _assert_iterable_complete(
        seg.get_rejected_list(),
        dtypes=(list, NoneType),
        element_dtypes=inttype,
        shape_max=(seg.get_num_rois(),),
    )
    _assert_iterable_complete(
        seg.get_roi_locations(),
        dtypes=(np.ndarray,),
        shape=(2, seg.get_num_rois()),
        element_dtypes=inttype,
    )
    _assert_iterable_complete(
        seg.get_traces(),
        dtypes=(np.ndarray, NoneType),
        element_dtypes=floattype,
        shape=(np.prod(seg.get_num_rois()), None),
    )
    assert isinstance(seg.get_traces_dict(), dict)
    assert isinstance(seg.get_images_dict(), dict)
    assert {"raw", "dff", "neuropil", "deconvolved"} == set(seg.get_traces_dict().keys())


def check_imaging_equal(
    imaging_extractor1: ImagingExtractor, imaging_extractor2: ImagingExtractor, exclude_channel_comparison: bool = False
):
    """Check that two imaging extractors have equal fields."""
    # assert equality:
    assert imaging_extractor1.get_num_frames() == imaging_extractor2.get_num_frames()
    assert imaging_extractor1.get_num_channels() == imaging_extractor2.get_num_channels()
    assert np.isclose(imaging_extractor1.get_sampling_frequency(), imaging_extractor2.get_sampling_frequency())
    assert_array_equal(imaging_extractor1.get_image_size(), imaging_extractor2.get_image_size())

    if not exclude_channel_comparison:
        assert_array_equal(imaging_extractor1.get_channel_names(), imaging_extractor2.get_channel_names())

    assert_array_equal(
        imaging_extractor1.get_video(start_frame=0, end_frame=1),
        imaging_extractor2.get_video(start_frame=0, end_frame=1),
    )
    assert_array_almost_equal(
        imaging_extractor1.frame_to_time(np.arange(imaging_extractor1.get_num_frames())),
        imaging_extractor2.frame_to_time(np.arange(imaging_extractor2.get_num_frames())),
    )


def assert_get_frames_return_shape(imaging_extractor: ImagingExtractor):
    """Check whether an ImagingExtractor get_frames function behaves as expected.

    We aim for the function to behave as numpy slicing and indexing as much as possible.
    """
    image_size = imaging_extractor.get_image_size()

    frame_idxs = 0
    frames_with_scalar = imaging_extractor.get_frames(frame_idxs=frame_idxs, channel=0)
    assert frames_with_scalar.shape == image_size, "get_frames does not work correctly with frame_idxs=0"

    frame_idxs = [0]
    frames_with_single_element_list = imaging_extractor.get_frames(frame_idxs=frame_idxs, channel=0)
    assert_msg = "get_frames does not work correctly with frame_idxs=[0]"
    assert frames_with_single_element_list.shape == (1, image_size[0], image_size[1]), assert_msg

    frame_idxs = [0, 1]
    frames_with_list = imaging_extractor.get_frames(frame_idxs=frame_idxs, channel=0)
    assert_msg = "get_frames does not work correctly with frame_idxs=[0, 1]"
    assert frames_with_list.shape == (2, image_size[0], image_size[1]), assert_msg

    frame_idxs = np.array([0, 1])
    frames_with_array = imaging_extractor.get_frames(frame_idxs=frame_idxs, channel=0)
    assert_msg = "get_frames does not work correctly with frame_idxs=np.arrray([0, 1])"
    assert frames_with_array.shape == (2, image_size[0], image_size[1]), assert_msg

    frame_idxs = [0, 2]
    frames_with_array = imaging_extractor.get_frames(frame_idxs=frame_idxs, channel=0)
    assert_msg = "get_frames does not work correctly with frame_idxs=[0, 2]"
    assert frames_with_array.shape == (2, image_size[0], image_size[1]), assert_msg


def check_imaging_return_types(img_ex: ImagingExtractor):
    """Check that the return types of the imaging extractor are correct."""
    assert isinstance(img_ex.get_num_frames(), inttype)
    assert isinstance(img_ex.get_num_channels(), inttype)
    assert isinstance(img_ex.get_sampling_frequency(), floattype)
    _assert_iterable_complete(
        iterable=img_ex.get_channel_names(),
        dtypes=(list, NoneType),
        element_dtypes=str,
        shape_max=(img_ex.get_num_channels(),),
    )
    _assert_iterable_complete(iterable=img_ex.get_image_size(), dtypes=Iterable, element_dtypes=inttype, shape=(2,))

    # This needs a method for getting frame shape not image size. It only works for n_channel==1
    # two_first_frames = img_ex.get_frames(frame_idxs=[0, 1])
    # _assert_iterable_complete(
    #     iterable=two_first_frames,
    #     dtypes=(np.ndarray,),
    #     element_dtypes=inttype + floattype,
    #     shape=(2, *img_ex.get_image_size()),
    # )
