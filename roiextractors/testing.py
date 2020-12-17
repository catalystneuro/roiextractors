from collections.abc import Iterable

import numpy as np
from numpy.testing import assert_array_equal

from .segmentationextractor import SegmentationExtractor

NoneType = type(None)
floattype = (float, np.floating)
inttype = (int, np.integer)


def _assert_iterable_shape(iterable, shape):
    ar = iterable if isinstance(iterable, np.ndarray) else np.array(iterable)
    for ar_shape, given_shape in zip(ar.shape, shape):
        if isinstance(given_shape, int):
            assert ar_shape == given_shape


def _assert_iterable_shape_max(iterable, shape_max):
    ar = iterable if isinstance(iterable, np.ndarray) else np.array(iterable)
    for ar_shape, given_shape in zip(ar.shape, shape_max):
        if isinstance(given_shape, int):
            assert ar_shape <= given_shape


def _assert_iterable_element_dtypes(iterable, dtypes):
    if isinstance(iterable, Iterable) and not isinstance(iterable, str):
        for iter in iterable:
            _assert_iterable_element_dtypes(iter, dtypes)
    else:
        assert isinstance(iterable, dtypes), f'array is none of the types {dtypes}'


def _assert_iterable_complete(iterable, dtypes=None, element_dtypes=None, shape=None, shape_max=None):
    assert isinstance(iterable, dtypes), f'iterable is none of the types {dtypes}'
    if not isinstance(iterable, NoneType):
        if shape is not None:
            _assert_iterable_shape(iterable, shape=shape)
        if shape_max is not None:
            _assert_iterable_shape_max(iterable, shape_max=shape_max)
        if element_dtypes is not None:
            _assert_iterable_element_dtypes(iterable, element_dtypes)


def check_segmentations_equal(seg1: SegmentationExtractor, seg2: SegmentationExtractor):
    check_segmentation_return_types(seg1)
    check_segmentation_return_types(seg2)
    # assert equality:
    assert seg1.get_num_rois() == seg2.get_num_rois()
    assert seg1.get_num_frames() == seg2.get_num_frames()
    assert seg1.get_num_channels() == seg2.get_num_channels()
    assert seg1.get_sampling_frequency() == seg2.get_sampling_frequency()
    assert_array_equal(seg1.get_channel_names(), seg2.get_channel_names())
    assert_array_equal(seg1.get_image_size(), seg2.get_image_size())
    assert_array_equal(seg1.get_roi_image_masks(), seg2.get_roi_image_masks())
    assert_array_equal(seg1.get_roi_pixel_masks(roi_ids=seg1.get_roi_ids()[:1]),
                       seg2.get_roi_pixel_masks(roi_ids=seg1.get_roi_ids()[:1]))
    assert_array_equal(seg1.get_image(), seg2.get_image())
    assert_array_equal(seg1.get_accepted_list(), seg2.get_accepted_list())
    assert_array_equal(seg1.get_rejected_list(), seg2.get_rejected_list())
    assert_array_equal(seg1.get_roi_locations(), seg2.get_roi_locations())
    assert_array_equal(seg1.get_roi_ids(), seg2.get_roi_ids())
    assert_array_equal(seg1.get_traces(), seg2.get_traces())


def check_segmentation_return_types(seg: SegmentationExtractor):
    """
    Parameters
    ----------
    seg:SegmentationExtractor
    """
    assert isinstance(seg.get_num_rois(), int)
    assert isinstance(seg.get_num_frames(), int)
    assert isinstance(seg.get_num_channels(), int)
    assert isinstance(seg.get_sampling_frequency(), (NoneType, floattype))
    _assert_iterable_complete(seg.get_channel_names(),
                              dtypes=list,
                              element_dtypes=str,
                              shape_max=(seg.get_num_channels(),))
    _assert_iterable_complete(seg.get_image_size(),
                              dtypes=Iterable,
                              element_dtypes=inttype,
                              shape=(2,))
    _assert_iterable_complete(seg.get_roi_image_masks(),
                              dtypes=(np.ndarray,),
                              element_dtypes=floattype,
                              shape=(*seg.get_image_size(), seg.get_num_rois()))
    _assert_iterable_complete(seg.get_roi_ids(),
                              dtypes=(list,),
                              shape=(seg.get_num_rois(),),
                              element_dtypes=inttype)
    assert isinstance(seg.get_roi_pixel_masks(roi_ids=seg.get_roi_ids()[:2]), list)
    _assert_iterable_complete(seg.get_roi_pixel_masks(roi_ids=seg.get_roi_ids()[:1])[0],
                              dtypes=(np.ndarray,),
                              element_dtypes=floattype,
                              shape_max=(np.prod(seg.get_image_size()), 3))
    _assert_iterable_complete(seg.get_image(),
                              dtypes=(np.ndarray, NoneType),
                              element_dtypes=floattype,
                              shape_max=(*seg.get_image_size(),))
    _assert_iterable_complete(seg.get_accepted_list(),
                              dtypes=(list, NoneType),
                              element_dtypes=inttype,
                              shape_max=(seg.get_num_rois(),))
    _assert_iterable_complete(seg.get_rejected_list(),
                              dtypes=(list, NoneType),
                              element_dtypes=inttype,
                              shape_max=(seg.get_num_rois(),))
    _assert_iterable_complete(seg.get_roi_locations(),
                              dtypes=(np.ndarray,),
                              shape=(2, seg.get_num_rois()),
                              element_dtypes=inttype)
    _assert_iterable_complete(seg.get_traces(),
                              dtypes=(np.ndarray, NoneType),
                              element_dtypes=floattype,
                              shape=(np.prod(seg.get_num_rois()), None))
    assert isinstance(seg.get_traces_dict(), dict)
    assert isinstance(seg.get_images_dict(), dict)
    assert {'raw', 'dff', 'neuropil', 'deconvolved'} == set(seg.get_traces_dict().keys())
    assert {'mean', 'correlation'} == set(seg.get_images_dict().keys())
