import numpy as np
from numpy.testing import assert_array_equal

from roiextractors import InscopixSegmentationExtractor

from .setup_paths import OPHYS_DATA_PATH


def test_inscopix_segmentation_extractor():
    file_path = OPHYS_DATA_PATH / "segmentation_data" / "inscopix" / "cellset.isxd"
    extractor = InscopixSegmentationExtractor(file_path=file_path)

    assert extractor.get_num_rois() == 4
    assert extractor.get_roi_ids() == ["C0", "C1", "C2", "C3"]
    assert extractor.get_accepted_list() == ["C0", "C1", "C2"]
    assert extractor.get_rejected_list() == ["C3"]
    assert extractor.get_image_size() == (398, 366)
    assert extractor.get_num_frames() == 5444
    img = extractor.get_roi_image_masks(["C1"])
    assert img.shape == (366, 398)
    np.testing.assert_allclose(extractor.get_sampling_frequency(), 9.998700168978033)
    assert extractor.get_traces().shape == (4, 5444)
    assert extractor.get_traces(start_frame=10, end_frame=20).shape == (4, 10)
    assert extractor.get_traces(start_frame=10, end_frame=20, roi_ids=["C1"]).shape == (1, 10)


def test_inscopix_segmentation_extractor_part1():
    file_path = OPHYS_DATA_PATH / "segmentation_data" / "inscopix" / "cellset_series_part1.isxd"
    extractor = InscopixSegmentationExtractor(file_path=file_path)

    assert extractor.get_num_rois() == 6
    assert extractor.get_roi_ids() == ["C0", "C1", "C2", "C3", "C4", "C5"]
    assert extractor.get_accepted_list() == []
    assert extractor.get_rejected_list() == []
    assert extractor.get_image_size() == (21, 21)
    img = extractor.get_roi_image_masks(["C1"])
    assert img.shape == (21, 21)
    assert extractor.get_sampling_frequency() == 10.0
    assert extractor.get_traces().shape == (6, 100)
    assert extractor.get_traces(start_frame=10, end_frame=20).shape == (6, 10)
    assert extractor.get_traces(start_frame=10, end_frame=20, roi_ids=["C1"]).shape == (1, 10)
    assert extractor.get_num_frames() == 100


def test_inscopix_segmentation_extractor_empty():
    file_path = OPHYS_DATA_PATH / "segmentation_data" / "inscopix" / "empty_cellset.isxd"
    extractor = InscopixSegmentationExtractor(file_path=file_path)

    assert extractor.get_num_rois() == 0
    assert extractor.get_roi_ids() == []
    assert extractor.get_accepted_list() == []
    assert extractor.get_rejected_list() == []
    assert extractor.get_image_size() == (5, 4)
    assert extractor.get_sampling_frequency() == 40.0
    assert extractor.get_num_frames() == 7
