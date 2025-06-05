import numpy as np
import pytest
from datetime import datetime
import platform
from numpy.testing import assert_array_equal
from pathlib import Path
from roiextractors import InscopixSegmentationExtractor

from .setup_paths import OPHYS_DATA_PATH

# Warn about macOS ARM64 environment
pytestmark = pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "arm64",
    reason="The isx package is currently not natively supported on macOS with Apple Silicon. "
    "Installation instructions can be found at: "
    "https://github.com/inscopix/pyisx?tab=readme-ov-file#install",
)


def test_inscopix_segmentation_extractor():
    """
    Test with a standard Inscopix segmentation dataset.

    Files: segmentation_datasets/inscopix/cellset.isxd
    Metadata:
    - Source: CellSet produced by CNMF-E (Inscopix isxcore v1.8.0)
    - 4 segmented cells (`C0` to `C3`)
    - 5444 samples (frames)
    - Frame rate: ~10.0 Hz (100.013 ms sampling period)
    - Volumetric: True (3 active planes: 400 µm, 700 µm, 1000 µm)
    - Frame shape: (366, 398)
    - Pixel size: 6 µm x 6 µm
    - Signal units: dF over noise (analog)
    - Cell status: Mixed (accepted/rejected status varies)
    - Animal ID: FV4581 (male, species: CaMKIICre)
    - Experiment: Retrieval day
    - Channel: Green, Exposure: 33 ms, Gain: 6
    - LED Power: 1.3 mW/mm², Focus: 1000 µm
    - Acquisition date: 2021-04-01 12:03:53.290011

    This test verifies that the extractor correctly:
    1. Loads the dataset and extracts basic properties (ROI count, image size, etc.)
    2. Retrieves ROI masks and traces with correct dimensions
    3. Handles frame slicing and ROI-specific trace extraction
    4. Extracts comprehensive metadata via specialized getter methods:
       - get_device_info(): Hardware and acquisition settings
       - get_subject_info(): Animal/specimen information
       - get_session_info(): Session and timing information
       - get_analysis_info(): Processing method information
       - get_probe_info(): Probe specifications (if applicable)
       - get_session_start_time(): Datetime object for session start
    """
    file_path = OPHYS_DATA_PATH / "segmentation_datasets" / "inscopix" / "cellset.isxd"
    extractor = InscopixSegmentationExtractor(file_path=str(file_path))

    # Test basic properties
    assert extractor.get_num_rois() == 4
    assert extractor.get_roi_ids() == [0, 1, 2, 3]
    assert extractor.get_original_roi_ids() == ["C0", "C1", "C2", "C3"]

    # Test status lists
    accepted_list = extractor.get_accepted_list()
    rejected_list = extractor.get_rejected_list()
    assert isinstance(accepted_list, list)
    assert isinstance(rejected_list, list)
    assert len(accepted_list) + len(rejected_list) <= 4

    # Test image properties
    assert extractor.get_image_size() == (398, 366)
    assert extractor.get_num_frames() == 5444

    # Test image masks (using integer ID instead of string)
    img = extractor.get_roi_image_masks([1])
    assert img.shape == (366, 398)

    # Test pixel masks
    pixel_masks = extractor.get_roi_pixel_masks([1])
    assert len(pixel_masks) == 1
    assert pixel_masks[0].shape[1] == 3

    # Test sampling frequency
    np.testing.assert_allclose(extractor.get_sampling_frequency(), 9.9987)

    # Test trace extraction
    assert extractor.get_traces().shape == (4, 5444)
    assert extractor.get_traces(start_frame=10, end_frame=20).shape == (4, 10)
    assert extractor.get_traces(start_frame=10, end_frame=20, roi_ids=[1]).shape == (1, 10)

    # Test session information
    session_info = extractor.get_session_info()
    assert session_info["session_name"] == "FV4581_Ret"
    assert session_info["experimenter_name"] == "Bei-Xuan"

    # Test session start time
    assert extractor.get_session_start_time() == datetime(2021, 4, 1, 12, 3, 53, 290011)

    # Test device information
    device_info = extractor.get_device_info()
    assert device_info["device_name"] == "NVista3"
    assert device_info["device_serial_number"] == "11132301"
    assert device_info["acquisition_software_version"] == "1.5.2"
    # Hardware/optical settings
    assert device_info["microscope_focus"] == 1000
    assert device_info["microscope_gain"] == 6
    assert device_info["channel"] == "green"
    assert device_info["efocus"] == 400
    assert device_info["exposure_time_ms"] == 33
    assert device_info["led_power_1_mw_per_mm2"] == 1.3
    assert device_info["led_power_2_mw_per_mm2"] == 0.2

    # Test subject information
    subject_info = extractor.get_subject_info()
    assert subject_info["animal_id"] == "FV4581"
    assert subject_info["species"] == "CaMKIICre"
    assert subject_info["sex"] == "m"
    assert subject_info["description"] == "Retrieval day"

    # Test analysis information
    analysis_info = extractor.get_analysis_info()
    assert analysis_info["cell_identification_method"] == "cnmfe"
    assert analysis_info["trace_units"] == "dF over noise"

    # Test probe information (returns empty dict for this dataset)
    probe_info = extractor.get_probe_info()
    assert isinstance(probe_info, dict)  # Most values are 0/"none" for this dataset


def test_inscopix_segmentation_extractor_part1():
    """
    Test with a smaller Inscopix segmentation dataset.

    Files: segmentation_datasets/inscopix/cellset_series_part1.isxd
    Metadata:
    - Source: CellSet produced by CNMF-E (Inscopix isxcore v1.8.0)
    - 6 segmented cells (`C0` to `C5`)
    - 100 samples (frames)
    - Frame rate: 10.0 Hz (100.00 ms sampling period)
    - Volumetric: False
    - Frame shape: (21, 21)
    - Pixel size: 6 µm x 6 µm
    - Footprint offset (top-left): (216, 156) in original movie
    - Signal units: dF over noise (analog)
    - Motion corrected: Yes (pre_mc = True, mc_padding = False)
    - Activity present in all cells (`CellActivity = True`)
    - Additional fields: cellMetrics, CellColors, Matches, PairScores
    - Start time: 1970-01-01 00:00:00 (epoch time)

    This test verifies that the extractor correctly:
    1. Loads the dataset and extracts basic properties (ROI count, image size, etc.)
    2. Retrieves ROI masks and traces with correct dimensions
    3. Handles frame slicing and ROI-specific trace extraction
    4. Gracefully handles limited metadata availability in smaller datasets
    """
    file_path = OPHYS_DATA_PATH / "segmentation_datasets" / "inscopix" / "cellset_series_part1.isxd"
    extractor = InscopixSegmentationExtractor(file_path=str(file_path))

    # Test basic properties
    assert extractor.get_num_rois() == 6
    assert extractor.get_roi_ids() == [0, 1, 2, 3, 4, 5]
    assert extractor.get_original_roi_ids() == ["C0", "C1", "C2", "C3", "C4", "C5"]

    # Test status lists (limited metadata may result in empty lists)
    accepted_list = extractor.get_accepted_list()
    rejected_list = extractor.get_rejected_list()
    assert isinstance(accepted_list, list)
    assert isinstance(rejected_list, list)

    # Test image properties
    assert extractor.get_image_size() == (21, 21)

    # Test image masks
    img = extractor.get_roi_image_masks([1])
    assert img.shape == (21, 21)

    # Test pixel masks
    pixel_masks = extractor.get_roi_pixel_masks([1])
    assert len(pixel_masks) == 1
    assert pixel_masks[0].shape[1] == 3

    # Test sampling frequency and frames
    assert extractor.get_sampling_frequency() == 10.0
    assert extractor.get_num_frames() == 100

    # Test session start time
    assert extractor.get_session_start_time() == datetime(1970, 1, 1, 0, 0, 0)

    # Test trace extraction
    assert extractor.get_traces().shape == (6, 100)
    assert extractor.get_traces(start_frame=10, end_frame=20).shape == (6, 10)
    assert extractor.get_traces(start_frame=10, end_frame=20, roi_ids=[1]).shape == (1, 10)


def test_inscopix_segmentation_extractor_empty():
    """
    Test with an empty Inscopix segmentation dataset.

    Files: segmentation_datasets/inscopix/empty_cellset.isxd
    Metadata:
    - Source: CellSet (Inscopix isxcore v1.8.0)
    - 0 segmented cells
    - 7 samples (frames)
    - Frame rate: 40.0 Hz (25.00 ms sampling period)
    - Volumetric: False
    - Frame shape: (4, 5)
    - Pixel size: 3 µm x 3 µm
    - Start time: 1970-01-01 00:00:00 (epoch time)

    This test verifies that the extractor correctly:
    1. Handles datasets with no ROIs gracefully
    2. Extracts available metadata (image size, frame count, etc.)
    3. Returns empty lists/dicts appropriately for missing data
    4. Doesn't crash when accessing metadata methods on empty datasets
    """
    file_path = OPHYS_DATA_PATH / "segmentation_datasets" / "inscopix" / "empty_cellset.isxd"
    extractor = InscopixSegmentationExtractor(file_path=str(file_path))

    # Test session start time
    assert extractor.get_session_start_time() == datetime(1970, 1, 1, 0, 0, 0)

    # Test basic properties
    assert extractor.get_num_rois() == 0
    assert extractor.get_roi_ids() == []
    assert extractor.get_original_roi_ids() == []

    # Test status lists
    assert extractor.get_accepted_list() == []
    assert extractor.get_rejected_list() == []

    # Test image properties
    assert extractor.get_image_size() == (5, 4)

    # Test sampling frequency and frames
    assert extractor.get_sampling_frequency() == 40.0
    assert extractor.get_num_frames() == 7
