import numpy as np
import shutil
import os
from pathlib import Path
from numbers import Number
from ..segmentationextractor import SegmentationExtractor
from .tools import assert_iterable_complete
from .tools import floattype, NoneType, inttype

def check_segmentations_equal(seg1, seg2):
    check_segmentation_return_types(seg1)
    check_segmentation_return_types(seg2)
    pass


def check_segmentation_return_types(seg):
    """
    Parameters
    ----------
    seg:SegmentationExtractor
    """
    assert isinstance(seg.get_num_rois(), int)
    assert isinstance(seg.get_num_frames(), int)
    assert isinstance(seg.get_num_channels(), int)
    assert isinstance(seg.get_sampling_frequency(), (NoneType, floattype))
    assert_iterable_complete(seg.get_channel_names(),
                             dtypes=list,
                             element_dtypes=str,
                             shape_max=(seg.get_num_channels(),))
    assert_iterable_complete(seg.get_image_size(),
                             dtypes=list,
                             element_dtypes=inttype,
                             shape=(2,))
    assert_iterable_complete(seg.get_roi_image_masks(),
                             dtypes=(np.ndarray,),
                             element_dtypes=floattype,
                             shape=(*seg.get_image_size(),seg.get_num_rois()))
    assert isinstance(seg.get_roi_pixel_masks(), list)
    assert_iterable_complete(seg.get_roi_pixel_masks()[0],
                             dtypes=(np.ndarray,),
                             element_dtypes=floattype,
                             shape_max=(np.prod(seg.get_image_size()), 3))
    assert_iterable_complete(seg.get_image(),
                             dtypes=(np.ndarray,),
                             element_dtypes=floattype,
                             shape_max=(*seg.get_image_size(),))
    assert_iterable_complete(seg.get_accepted_list(),
                             dtypes=(list,),
                             element_dtypes=inttype,
                             shape_max=(seg.get_num_rois(),))
    assert_iterable_complete(seg.get_rejected_list(),
                             dtypes=(list, NoneType),
                             element_dtypes=inttype,
                             shape_max=(seg.get_num_rois(),))
    assert_iterable_complete(seg.get_roi_locations(),
                             dtypes=(np.ndarray,),
                             shape=(2, seg.get_num_rois()),
                             element_dtypes=inttype)
    assert_iterable_complete(seg.get_roi_ids(),
                             dtypes=(list,),
                             shape=(seg.get_num_rois(),),
                             element_dtypes=inttype)
    assert_iterable_complete(seg.get_traces(),
                             dtypes=(np.ndarray,),
                             element_dtypes=floattype,
                             shape=(np.prod(seg.get_num_rois()), None))
    assert isinstance(seg.get_traces_dict(), dict)
    assert isinstance(seg.get_images_dict(), dict)
    assert {'raw', 'dff', 'neuropil', 'deconvolved'} == set(seg.get_traces_dict().keys())
    assert {'mean', 'correlation'} == set(seg.get_images_dict().keys())


def check_write_segmentation(seg):
    pass
