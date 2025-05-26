import numpy as np
import pytest
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

print(f"OPHYS_DATA_PATH: {OPHYS_DATA_PATH}")


def test_inscopix_segmentation_extractor():
    """
    Test with a standard Inscopix segmentation dataset.

    Files: segmentation_datasets/inscopix/cellset.isxd
    Metadata:
    - Source: CellSet produced by CNMF-E (Inscopix isxcore v1.8.0)
    - 4 segmented cells (`C0` to `C3`)
    - 5444 samples (frames)
    - Frame rate: 10.0 Hz (100.00 ms sampling period)
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
    - Acquisition date: 2021-04-01 12:03:53.290011
    - Duration: 544.40 seconds

    This test verifies that the extractor correctly:
    1. Loads the dataset and extracts metadata (ROI count, image size, etc.)
    2. Retrieves ROI masks and traces with correct dimensions
    3. Handles frame slicing and ROI-specific trace extraction
    4. Extracts comprehensive metadata including device, subject, and session info
    """
    file_path = OPHYS_DATA_PATH / "segmentation_datasets" / "inscopix" / "cellset.isxd"
    extractor = InscopixSegmentationExtractor(file_path=str(file_path))

    # Test basic properties
    assert extractor.get_num_rois() == 4
    assert extractor.get_roi_ids() == [0, 1, 2, 3]
    assert extractor.get_original_roi_ids() == ["C0", "C1", "C2", "C3"]

    # Test status lists
    assert extractor.get_accepted_list() == [0, 1, 2]
    assert extractor.get_rejected_list() == [3]

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

    # Test comprehensive metadata extraction
    
    # Test session information
    session_info = extractor.get_session_info()
    assert session_info['num_samples'] == 5444
    np.testing.assert_allclose(session_info['duration_seconds'], 544.40, rtol=1e-2)
    assert session_info['session_name'] == 'FV4581_Ret'
    assert session_info['experimenter_name'] == 'Bei-Xuan'
    
    # Test session start time
    start_time = extractor.get_session_start_time()
    assert start_time is not None
    assert start_time.year == 2021
    assert start_time.month == 4
    
    # Test device information
    device_info = extractor.get_device_info()
    assert device_info['device_name'] == 'NVista3'
    assert device_info['device_serial_number'] == '11132301'
    assert device_info['acquisition_software_version'] == '1.5.2'
    
    # Test imaging parameters
    imaging_info = extractor.get_imaging_info()
    assert imaging_info['microscope_focus'] == 1000
    assert imaging_info['microscope_gain'] == 6
    assert imaging_info['channel'] == 'green'
    assert imaging_info['efocus'] == 400
    
    # Test subject information
    subject_info = extractor.get_subject_info()
    assert subject_info['animal_id'] == 'FV4581'
    assert subject_info['species_strain'] == 'CaMKIICre'
    assert subject_info['sex'] == 'm'
    assert subject_info['weight'] == 0
    
    # Test analysis information
    analysis_info = extractor.get_analysis_info()
    assert analysis_info['cell_identification_method'] == 'cnmfe'
    assert analysis_info['trace_units'] == 'dF over noise'
    
    # Test probe information 
    probe_info = extractor.get_probe_info()
    # most values are 0/"none" for this dataset returns empty dict 
    assert isinstance(probe_info, dict)


def test_inscopix_segmentation_extractor_part1():
    """
    Test with a smaller Inscopix segmentation dataset

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
    - Duration: 10.00 seconds

    This test verifies that the extractor correctly:
    1. Loads the dataset and extracts metadata (ROI count, image size, etc.)
    2. Retrieves ROI masks and traces with correct dimensions
    3. Handles frame slicing and ROI-specific trace extraction
    """
    file_path = OPHYS_DATA_PATH / "segmentation_datasets" / "inscopix" / "cellset_series_part1.isxd"
    extractor = InscopixSegmentationExtractor(file_path=str(file_path))

    # Test basic properties
    assert extractor.get_num_rois() == 6
    assert extractor.get_roi_ids() == [0, 1, 2, 3, 4, 5]
    assert extractor.get_original_roi_ids() == ["C0", "C1", "C2", "C3", "C4", "C5"]

    # Test status lists (based on CellStatuses: [1, 1, 1, 1, 1, 1] in the metadata,
    # all cells have the same status, which doesn't map directly to accepted/rejected)
    assert extractor.get_accepted_list() == []
    assert extractor.get_rejected_list() == []

    # Test image properties
    assert extractor.get_image_size() == (21, 21)

    # Test image masks
    img = extractor.get_roi_image_masks([1])
    assert img.shape == (21, 21)

    # Test pixel masks
    pixel_masks = extractor.get_roi_pixel_masks([1])
    assert len(pixel_masks) == 1
    assert pixel_masks[0].shape[1] == 3  # Each row should have (x, y, weight)

    # Test sampling frequency and frames
    assert extractor.get_sampling_frequency() == 10.0
    assert extractor.get_num_frames() == 100

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
    - Duration: 0.17 seconds

    This test verifies that the extractor correctly:
    1. Handles datasets with no ROIs
    2. Extracts metadata (image size, frame count, etc.)
    3. Ensures no ROI masks or traces are returned
    """
    file_path = OPHYS_DATA_PATH / "segmentation_datasets" / "inscopix" / "empty_cellset.isxd"
    extractor = InscopixSegmentationExtractor(file_path=str(file_path))

    # Test basic properties
    assert extractor.get_num_rois() == 0
    assert extractor.get_roi_ids() == []
    assert extractor.get_original_roi_ids() == []

    # Test status lists
    assert extractor.get_accepted_list() == []
    assert extractor.get_rejected_list() == []

    # Test image properties - corrected dimensions based on actual metadata
    assert extractor.get_image_size() == (5, 4)

    # Test sampling frequency and frames - corrected to match actual 25ms period (40 Hz)
    assert extractor.get_sampling_frequency() == 40.0
    assert extractor.get_num_frames() == 7