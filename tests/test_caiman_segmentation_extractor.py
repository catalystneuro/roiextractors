import numpy as np

from roiextractors import CaimanSegmentationExtractor
from tests.setup_paths import OPHYS_DATA_PATH


def test_caiman_segmentation_extractor():
    """
    Test with a CaImAn segmentation dataset from multi-plane miniscope recording.

    File: multi_plane_with_imaging_data/mini_1000_caiman_stubbed_10units_5frames.hdf5
    Metadata:
    - Source: CaImAn CNMF-E analysis output (Fleischmann Lab, Brown University)
    - 10 segmented ROIs (stubbed from original dataset)
    - 5 samples (frames, stubbed from original)
    - Frame rate: 20.0 Hz
    - Frame shape: (231, 242) pixels
    - Imaging plane: 1000 µm depth
    - Dataset type: 1-photon miniscope imaging
    - Processing: Motion corrected and segmented with CaImAn CNMF-E
    - Contains spatial footprints (A), temporal traces (C, F_dff, S, YrA),
    background components (b, f), quality metrics, and component indices

    """
    file_path = (
        OPHYS_DATA_PATH
        / "segmentation_datasets"
        / "caiman"
        / "multi_plane_with_imaging_data"
        / "mini_1000_caiman_stubbed_10units_5frames.hdf5"
    )
    extractor = CaimanSegmentationExtractor(file_path=str(file_path))

    # Test basic properties
    assert extractor.get_num_rois() == 10
    assert extractor.get_frame_shape() == (231, 242)
    assert extractor.get_num_samples() == 5
    assert extractor.get_sampling_frequency() == 20.0

    # Test ROI IDs
    roi_ids = extractor.get_roi_ids()
    assert len(roi_ids) == 10

    # Test accepted/rejected lists
    accepted_list = extractor.get_accepted_list()
    rejected_list = extractor.get_rejected_list()
    assert len(accepted_list) + len(rejected_list) <= 10

    # Test ROI image masks
    single_mask = extractor.get_roi_image_masks([roi_ids[0]])
    assert single_mask.shape == (231, 242, 1)

    all_masks = extractor.get_roi_image_masks()
    assert all_masks.shape[0:2] == (231, 242)

    # Test ROI pixel masks
    pixel_masks = extractor.get_roi_pixel_masks([roi_ids[0]])
    assert len(pixel_masks) == 1
    assert pixel_masks[0].shape[1] == 3  # [y, x, weight]

    # Test traces
    num_rois = extractor.get_num_rois()
    num_samples = extractor.get_num_samples()
    all_traces = extractor.get_traces()
    assert all_traces.shape == (num_samples, num_rois)

    # Test trace slicing
    start_frame, end_frame = 1, 4
    sliced_traces = extractor.get_traces(start_frame=start_frame, end_frame=end_frame)
    assert sliced_traces.shape == (end_frame - start_frame, num_rois)

    # Test ROI-specific traces
    roi_traces = extractor.get_traces(roi_ids=[roi_ids[0]])
    assert roi_traces.shape == (num_samples, 1)

    # Test different trace types
    traces_dict = extractor.get_traces_dict()
    available_traces = {name: traces for name, traces in traces_dict.items() if traces is not None}
    for traces in available_traces.values():
        assert traces.shape[0] == num_samples  # Time dimension should match

    # Test images
    images_dict = extractor.get_images_dict()
    available_images = {name: image for name, image in images_dict.items() if image is not None}
    frame_shape = extractor.get_frame_shape()
    for image in available_images.values():
        assert image.shape == frame_shape


def test_caiman_segmentation_extractor_analysis():
    """
    Test with a full CaImAn analysis dataset from 1-photon miniscope recording.

    File: caiman_analysis.hdf5
    Metadata:
    - Source: CaImAn CNMF analysis output (v1.8.5, greedy ROI initialization)
    - 72 segmented ROIs from 1-photon endoscopic imaging
    - 1000 samples (frames)
    - Frame rate: 30.0 Hz
    - Frame shape: (128, 128) pixels
    - Dataset type: 1-photon miniscope/endoscopic imaging
    - Processing: Motion corrected and segmented with CaImAn CNMF
    - Contains full CaImAn output: spatial footprints (A), temporal traces (C, F_dff, S, YrA),
      background components (b, f), quality metrics, and processing parameters
    - High correlation threshold (0.85) and peak-to-noise ratio (20) typical for 1-photon data

    """
    file_path = OPHYS_DATA_PATH / "segmentation_datasets" / "caiman" / "caiman_analysis.hdf5"
    extractor = CaimanSegmentationExtractor(file_path=str(file_path))

    # Test basic properties
    assert extractor.get_num_rois() == 72
    assert extractor.get_frame_shape() == (128, 128)
    assert extractor.get_num_samples() == 1000
    assert extractor.get_sampling_frequency() == 30.0

    # Test ROI IDs
    roi_ids = extractor.get_roi_ids()
    assert len(roi_ids) == 72

    # Test accepted/rejected lists
    accepted_list = extractor.get_accepted_list()
    rejected_list = extractor.get_rejected_list()
    assert len(accepted_list) + len(rejected_list) <= 72

    # Test ROI image masks
    single_mask = extractor.get_roi_image_masks([roi_ids[0]])
    assert single_mask.shape == (128, 128, 1)

    all_masks = extractor.get_roi_image_masks()
    assert all_masks.shape[0:2] == (128, 128)

    # Test ROI pixel masks
    pixel_masks = extractor.get_roi_pixel_masks([roi_ids[0]])
    assert len(pixel_masks) == 1
    assert pixel_masks[0].shape[1] == 3  # [y, x, weight]

    # Test traces
    num_rois = extractor.get_num_rois()
    num_samples = extractor.get_num_samples()
    all_traces = extractor.get_traces()
    assert all_traces.shape == (num_samples, num_rois)

    # Test trace slicing
    start_frame, end_frame = 100, 200
    sliced_traces = extractor.get_traces(start_frame=start_frame, end_frame=end_frame)
    assert sliced_traces.shape == (end_frame - start_frame, num_rois)

    # Test ROI-specific traces
    roi_traces = extractor.get_traces(roi_ids=[roi_ids[0]])
    assert roi_traces.shape == (num_samples, 1)

    # Test different trace types
    traces_dict = extractor.get_traces_dict()
    available_traces = {name: traces for name, traces in traces_dict.items() if traces is not None}
    for traces in available_traces.values():
        assert traces.shape[0] == num_samples  # Time dimension should match

    # Test images
    images_dict = extractor.get_images_dict()
    available_images = {name: image for name, image in images_dict.items() if image is not None}
    frame_shape = extractor.get_frame_shape()
    for image in available_images.values():
        assert image.shape == frame_shape
