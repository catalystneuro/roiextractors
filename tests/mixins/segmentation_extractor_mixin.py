import pytest
import numpy as np


class SegmentationExtractorMixin:
    def test_get_image_size(self, segmentation_extractor, expected_image_masks):
        image_size = segmentation_extractor.get_image_size()
        assert image_size == (expected_image_masks.shape[0], expected_image_masks.shape[1])

    def test_get_num_frames(self, segmentation_extractor, expected_roi_response_traces):
        num_frames = segmentation_extractor.get_num_frames()
        first_expected_roi_response_trace = list(expected_roi_response_traces.values())[0]
        assert num_frames == first_expected_roi_response_trace.shape[0]

    def test_get_sampling_frequency(self, segmentation_extractor, expected_sampling_frequency):
        sampling_frequency = segmentation_extractor.get_sampling_frequency()
        assert sampling_frequency == expected_sampling_frequency

    def test_get_roi_ids(self, segmentation_extractor, expected_roi_ids):
        roi_ids = segmentation_extractor.get_roi_ids()
        np.testing.assert_array_equal(roi_ids, expected_roi_ids)

    def test_get_num_rois(self, segmentation_extractor, expected_roi_ids):
        num_rois = segmentation_extractor.get_num_rois()
        assert num_rois == len(expected_roi_ids)

    def test_get_accepted_roi_ids(self, segmentation_extractor, expected_accepted_list):
        accepted_list = segmentation_extractor.get_accepted_roi_ids()
        np.testing.assert_array_equal(accepted_list, expected_accepted_list)

    def test_get_rejected_roi_ids(self, segmentation_extractor, expected_rejected_list):
        rejected_list = segmentation_extractor.get_rejected_roi_ids()
        np.testing.assert_array_equal(rejected_list, expected_rejected_list)

    def test_get_roi_locations(self, segmentation_extractor, expected_roi_locations):
        roi_locations = segmentation_extractor.get_roi_locations()
        np.testing.assert_array_equal(roi_locations, expected_roi_locations)

    @pytest.mark.parametrize("roi_indices", (None, [], [0], [0, 1]))
    def test_get_roi_image_masks(self, segmentation_extractor, expected_image_masks, expected_roi_ids, roi_indices):
        if roi_indices is None:
            image_masks = segmentation_extractor.get_roi_image_masks()
            np.testing.assert_array_equal(image_masks, expected_image_masks)
        else:
            roi_ids = [expected_roi_ids[i] for i in roi_indices]
            image_masks = segmentation_extractor.get_roi_image_masks(roi_ids=roi_ids)
            np.testing.assert_array_equal(image_masks, expected_image_masks[:, :, roi_indices])

    def test_roi_pixel_masks(self, segmentation_extractor, expected_image_masks, expected_roi_ids):
        pixel_masks = segmentation_extractor.get_roi_pixel_masks()
        for i, pixel_mask in enumerate(pixel_masks):
            expected_image_mask = expected_image_masks[:, :, i]
            expected_locs = np.where(expected_image_mask > 0)
            expected_values = expected_image_mask[expected_image_mask > 0]
            np.testing.assert_array_equal(pixel_mask[:, 0], expected_locs[0])
            np.testing.assert_array_equal(pixel_mask[:, 1], expected_locs[1])
            np.testing.assert_array_equal(pixel_mask[:, 2], expected_values)

    def test_get_roi_response_traces(self, segmentation_extractor, expected_roi_response_traces):
        roi_response_traces = segmentation_extractor.get_roi_response_traces()
        for name, expected_trace in expected_roi_response_traces.items():
            np.testing.assert_array_equal(roi_response_traces[name], expected_trace)

    def test_get_background_ids(self, segmentation_extractor, expected_background_ids):
        background_ids = segmentation_extractor.get_background_ids()
        np.testing.assert_array_equal(background_ids, expected_background_ids)

    def test_get_num_background_components(self, segmentation_extractor, expected_background_ids):
        num_background_components = segmentation_extractor.get_num_background_components()
        assert num_background_components == len(expected_background_ids)

    def test_get_background_image_masks(self, segmentation_extractor, expected_background_image_masks):
        background_image_masks = segmentation_extractor.get_background_image_masks()
        np.testing.assert_array_equal(background_image_masks, expected_background_image_masks)

    def test_get_background_pixel_masks(self, segmentation_extractor, expected_background_image_masks):
        pixel_masks = segmentation_extractor.get_background_pixel_masks()
        for i, pixel_mask in enumerate(pixel_masks):
            expected_image_mask = expected_background_image_masks[:, :, i]
            expected_locs = np.where(expected_image_mask > 0)
            expected_values = expected_image_mask[expected_image_mask > 0]
            np.testing.assert_array_equal(pixel_mask[:, 0], expected_locs[0])
            np.testing.assert_array_equal(pixel_mask[:, 1], expected_locs[1])
            np.testing.assert_array_equal(pixel_mask[:, 2], expected_values)

    def test_get_background_response_traces(self, segmentation_extractor, expected_background_response_traces):
        background_response_traces = segmentation_extractor.get_background_response_traces()
        for name, expected_trace in expected_background_response_traces.items():
            np.testing.assert_array_equal(background_response_traces[name], expected_trace)

    def test_get_summary_images(self, segmentation_extractor, expected_mean_image, expected_correlation_image):
        name_to_image = segmentation_extractor.get_summary_images()
        mean_image = name_to_image["mean"]
        correlation_image = name_to_image["correlation"]
        np.testing.assert_array_equal(mean_image, expected_mean_image)
        np.testing.assert_array_equal(correlation_image, expected_correlation_image)
