import numpy as np
import pytest
import platform
from numpy.testing import assert_array_equal
from pathlib import Path
from roiextractors import InscopixSegmentationExtractor

from .setup_paths import OPHYS_DATA_PATH

# Skip all tests in this file on macOS
pytestmark = pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "arm64",
    reason="Inscopix is not natively supported on macOS ARM",
)

print(f"OPHYS_DATA_PATH: {OPHYS_DATA_PATH}")


def test_inscopix_segmentation_extractor():
    """
    Test with a standard Inscopix segmentation dataset.

    Files: segmentation_datasets/inscopix/cellset.isxd
    Metadata:
    - Source: CellSet produced by CNMF-E (Inscopix isxcore v1.8.0)
    - 4 segmented cells (`C0` to `C3`)
    - 5444 samples (frames)
    - Frame rate: ~9.9987 Hz
    - Volumetric: True (3 active planes: 400 µm, 700 µm, 1000 µm)
    - Frame shape: (366, 398)
    - Pixel size: 6 µm x 6 µm
    - Signal units: dF over noise (analog)
    - Cell status:
        - C0: Rejected
        - C1: Rejected
        - C2: Unknown
        - C3: Accepted
    - Animal ID: FV4581 (male, species: CaMKIICre)
    - Experiment: Retrieval day
    - Channel: Green, Exposure: 33 ms, Gain: 6
    - LED Power: 1.3, Focus: 1000 µm
    - Acquisition date: 2021-04-01

    This test verifies that the extractor correctly:
    1. Loads the dataset and extracts metadata (ROI count, image size, etc.)
    2. Retrieves ROI masks and traces with correct dimensions
    3. Handles frame slicing and ROI-specific trace extraction
    """
    file_path = OPHYS_DATA_PATH / "segmentation_datasets" / "inscopix" / "cellset.isxd"
    extractor = InscopixSegmentationExtractor(file_path=str(file_path))

    assert extractor.get_num_rois() == 4
    assert extractor.get_roi_ids() == ["C0", "C1", "C2", "C3"]
    assert extractor.get_accepted_list() == ["C0", "C1", "C2"]
    assert extractor.get_rejected_list() == ["C3"]
    assert extractor.get_frame_shape() == (398, 366)
    assert extractor.get_num_samples() == 5444
    img = extractor.get_roi_image_masks(["C1"])
    assert img.shape == (366, 398)
    np.testing.assert_allclose(extractor.get_sampling_frequency(), 9.998700168978033)
    assert extractor.get_traces().shape == (4, 5444)
    assert extractor.get_traces(start_frame=10, end_frame=20).shape == (4, 10)
    assert extractor.get_traces(start_frame=10, end_frame=20, roi_ids=["C1"]).shape == (1, 10)


def test_inscopix_segmentation_extractor_part1():
    """
    Test with a smaller Inscopix segmentation dataset

    Files: segmentation_datasets/inscopix/cellset_series_part1.isxd
    Metadata:
    - Source: CellSet produced by CNMF-E (Inscopix isxcore v1.8.0)
    - 6 segmented cells (`C0` to `C5`)
    - 100 samples (frames)
    - Frame rate: 10.0 Hz
    - Volumetric: False
    - Frame shape: (21, 21)
    - Pixel size: 6 µm x 6 µm
    - Footprint offset (top-left): (216, 156) in original movie
    - Signal units: dF over noise (analog)
    - Motion corrected: Yes (pre_mc = True, mc_padding = False)
    - Activity present in all cells (`CellActivity = True`)
    - Additional fields: cellMetrics, CellColors, Matches, PairScores

    This test verifies that the extractor correctly:
    1. Loads the dataset and extracts metadata (ROI count, image size, etc.)
    2. Retrieves ROI masks and traces with correct dimensions
    3. Handles frame slicing and ROI-specific trace extraction
    """
    file_path = OPHYS_DATA_PATH / "segmentation_datasets" / "inscopix" / "cellset_series_part1.isxd"
    extractor = InscopixSegmentationExtractor(file_path=str(file_path))

    assert extractor.get_num_rois() == 6
    assert extractor.get_roi_ids() == ["C0", "C1", "C2", "C3", "C4", "C5"]
    assert extractor.get_accepted_list() == []
    assert extractor.get_rejected_list() == []
    assert extractor.get_frame_shape() == (21, 21)
    img = extractor.get_roi_image_masks(["C1"])
    assert img.shape == (21, 21)
    assert extractor.get_sampling_frequency() == 10.0
    assert extractor.get_traces().shape == (6, 100)
    assert extractor.get_traces(start_frame=10, end_frame=20).shape == (6, 10)
    assert extractor.get_traces(start_frame=10, end_frame=20, roi_ids=["C1"]).shape == (1, 10)
    assert extractor.get_num_samples() == 100


def test_inscopix_segmentation_extractor_empty():
    """
    Test with an empty Inscopix segmentation dataset.

    Files: segmentation_datasets/inscopix/empty_cellset.isxd
    Metadata:
    - Source: CellSet (Inscopix isxcore v1.8.0)
    - 0 segmented cells
    - 7 samples (frames)
    - Frame rate: 40.0 Hz
    - Volumetric: False
    - Frame shape: (4, 5)
    - Pixel size: 3 µm x 3 µm


    This test verifies that the extractor correctly:
    1. Handles datasets with no ROIs
    2. Extracts metadata (image size, frame count, etc.)
    3. Ensures no ROI masks or traces are returned
    """
    file_path = OPHYS_DATA_PATH / "segmentation_datasets" / "inscopix" / "empty_cellset.isxd"
    extractor = InscopixSegmentationExtractor(file_path=str(file_path))

    assert extractor.get_num_rois() == 0
    assert extractor.get_roi_ids() == []
    assert extractor.get_accepted_list() == []
    assert extractor.get_rejected_list() == []
    assert extractor.get_frame_shape() == (5, 4)
    assert extractor.get_sampling_frequency() == 40.0
    assert extractor.get_num_samples() == 7
