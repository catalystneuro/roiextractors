import numpy as np
from numpy.testing import assert_array_almost_equal

from roiextractors import CaimanSegmentationExtractor
from tests.setup_paths import OPHYS_DATA_PATH


def test_caiman_segmentation_extractor_1000():
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

    # Test quality metrics
    snr_values = extractor._get_snr_values()
    expected_snr = np.array(
        [
            2.91673987,
            1.05541609,
            1.59315755,
            4.66348506,
            2.97824475,
            2.98275578,
            2.9776668,
            2.94389109,
            1.19369399,
            1.76922279,
        ]
    )
    assert_array_almost_equal(snr_values, expected_snr, decimal=6)
    r_values = extractor._get_spatial_correlation_values()
    expected_r_values = np.array(
        [
            0.35878084,
            0.182326,
            0.34179541,
            0.45524581,
            0.13731668,
            0.58202635,
            0.26673053,
            0.29431159,
            0.3361319,
            0.34527933,
        ]
    )
    assert_array_almost_equal(r_values, expected_r_values, decimal=6)
    cnn_preds = extractor._get_cnn_predictions()
    assert cnn_preds is None

    quality_metrics = extractor.get_quality_metrics()
    expected_metrics = ["snr", "r_values"]
    for metric_name in expected_metrics:
        assert metric_name in quality_metrics
        assert len(quality_metrics[metric_name]) == 10
    assert "cnn_predictions" not in quality_metrics


def test_caiman_segmentation_extractor_450():
    """
    Test with a CaImAn segmentation dataset from multi-plane miniscope recording.
    File: multi_plane_with_imaging_data/mini_450_caiman_stubbed_10units_5frames.hdf5
    Metadata:
    - Source: CaImAn CNMF-E analysis output (Fleischmann Lab, Brown University)
    - 10 segmented ROIs (stubbed from original dataset)
    - 5 samples (frames, stubbed from original)
    - Frame rate: 20.0 Hz
    - Frame shape: (231, 242) pixels
    - Imaging plane: 450 µm depth
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
        / "mini_450_caiman_stubbed_10units_5frames.hdf5"
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
    assert accepted_list == [2, 9]
    assert rejected_list == [0, 1, 3, 4, 5, 6, 7, 8]
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

        # Test quality metrics
    snr_values = extractor._get_snr_values()
    expected_snr = np.array(
        [
            0.82003892,
            1.46365421,
            2.37001396,
            1.26491304,
            0.88042214,
            0.57378107,
            1.91813623,
            1.51477906,
            1.34824757,
            4.61553793,
        ]
    )
    assert_array_almost_equal(snr_values, expected_snr, decimal=6)

    r_values = extractor._get_spatial_correlation_values()
    expected_r_values = np.array(
        [
            -0.09622008,
            0.75080916,
            0.46274736,
            0.14843417,
            -0.01086164,
            0.32011244,
            -0.06491879,
            -0.12745297,
            -0.10657519,
            -0.37634573,
        ]
    )
    assert_array_almost_equal(r_values, expected_r_values, decimal=6)

    cnn_preds = extractor._get_cnn_predictions()
    assert cnn_preds is None

    quality_metrics = extractor.get_quality_metrics()
    expected_metrics = ["snr", "r_values"]
    for metric_name in expected_metrics:
        assert metric_name in quality_metrics
        assert len(quality_metrics[metric_name]) == 10
    assert "cnn_predictions" not in quality_metrics


def test_caiman_segmentation_extractor_750():
    """
    Test with a CaImAn segmentation dataset from multi-plane miniscope recording.
    File: multi_plane_with_imaging_data/mini_750_caiman_stubbed_10units_5frames.hdf5
    Metadata:
    - Source: CaImAn CNMF-E analysis output (Fleischmann Lab, Brown University)
    - 10 segmented ROIs (stubbed from original dataset)
    - 5 samples (frames, stubbed from original)
    - Frame rate: 20.0 Hz
    - Frame shape: (231, 242) pixels
    - Imaging plane: 750 µm depth
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
        / "mini_750_caiman_stubbed_10units_5frames.hdf5"
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

    # Test quality metrics
    snr_values = extractor._get_snr_values()
    expected_snr = np.array(
        [
            3.83573586,
            0.99122825,
            1.42381208,
            2.2360468,
            5.05530997,
            2.82601472,
            3.39339375,
            1.27190148,
            4.47956692,
            2.43551304,
        ]
    )
    assert_array_almost_equal(snr_values, expected_snr, decimal=6)

    r_values = extractor._get_spatial_correlation_values()
    expected_r_values = np.array(
        [
            0.1108782,
            -0.2044603,
            -0.30995306,
            0.13629865,
            0.39358738,
            0.33849331,
            -0.31045486,
            -0.64353179,
            0.31970047,
            0.58292105,
        ]
    )
    assert_array_almost_equal(r_values, expected_r_values, decimal=6)

    cnn_preds = extractor._get_cnn_predictions()
    assert cnn_preds is None

    quality_metrics = extractor.get_quality_metrics()
    expected_metrics = ["snr", "r_values"]
    for metric_name in expected_metrics:
        assert metric_name in quality_metrics
        assert len(quality_metrics[metric_name]) == 10
    assert "cnn_predictions" not in quality_metrics


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

    # Test quality metrics
    snr_values = extractor._get_snr_values()
    r_values = extractor._get_spatial_correlation_values()
    cnn_preds = extractor._get_cnn_predictions()
    quality_metrics = extractor.get_quality_metrics()
    assert snr_values is None
    assert r_values is None
    assert cnn_preds is None
    for metric_name in ["snr", "r_values", "cnn_preds"]:
        assert metric_name not in quality_metrics
