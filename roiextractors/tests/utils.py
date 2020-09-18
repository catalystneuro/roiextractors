import numpy as np
import shutil
import os
from pathlib import Path
from numbers import Number
from ..segmentationextractor import SegmentationExtractor
from .tools import assert_iterable_element_dtypes, assert_iterable_complete


def check_segmentations_equal(seg1,seg2):
    pass

def check_segmentation_return_types(seg):
    """
    Parameters
    ----------
    seg:SegmentationExtractor
    """
    NoneType=type(None)
    assert isinstance(seg.get_num_rois(), int)
    assert_iterable_complete(seg.get_accepted_list(),
                             dtypes=(list,),
                             element_dtypes=int,
                             shape_max=(1, seg.get_num_rois()))
    assert_iterable_complete(seg.get_rejected_list(),
                             dtypes=(list,NoneType),
                             element_dtypes=int,
                             shape_max=(1,seg.get_num_rois()))
    assert_iterable_complete(seg.get_roi_locations(),
                             shape=(2,None),
                             dtypes=Number,
                             shape_max=(2,seg.get_num_rois()))

    pass

def check_write_segmentation(seg):
    pass