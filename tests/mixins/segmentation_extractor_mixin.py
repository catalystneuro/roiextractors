import pytest
import numpy as np


class SegmentationExtractorMixin:
    def test_get_image_size(self, segmentation_extractor, expected_video):
        image_size = segmentation_extractor.get_image_size()
        assert image_size == (expected_video.shape[1], expected_video.shape[2])

    def test_get_num_frames(self, segmentation_extractor, expected_video):
        num_frames = segmentation_extractor.get_num_frames()
        assert num_frames == expected_video.shape[0]

    def test_get_sampling_frequency(self, segmentation_extractor, expected_sampling_frequency):
        sampling_frequency = segmentation_extractor.get_sampling_frequency()
        assert sampling_frequency == expected_sampling_frequency

    def test_get_accepted_roi_ids(self, segmentation_extractor, expected_accepted_list):
        accepted_list = segmentation_extractor.get_accepted_roi_ids()
        np.testing.assert_array_equal(accepted_list, expected_accepted_list)

    def test_get_rejected_roi_ids(self, segmentation_extractor, expected_rejected_list):
        rejected_list = segmentation_extractor.get_rejected_roi_ids()
        np.testing.assert_array_equal(rejected_list, expected_rejected_list)

    def test_get_image_size(self, segmentation_extractor, expected_image_masks):
        image_size = segmentation_extractor.get_image_size()
        assert image_size == expected_image_masks.shape[:2]

    def test_get_num_frames(self, segmentation_extractor, expected_roi_response_traces):
        num_frames = segmentation_extractor.get_num_frames()
        assert num_frames == list(expected_roi_response_traces.values())[0].shape[0]

    @pytest.mark.parametrize("roi_indices", (None, [], [0], [0, 1]))
    def test_get_roi_image_masks(self, segmentation_extractor, expected_image_masks, expected_roi_ids, roi_indices):
        if roi_indices is None:
            image_masks = segmentation_extractor.get_roi_image_masks()
            np.testing.assert_array_equal(image_masks, expected_image_masks)
        else:
            roi_ids = [expected_roi_ids[i] for i in roi_indices]
            image_masks = segmentation_extractor.get_roi_image_masks(roi_ids=roi_ids)
            np.testing.assert_array_equal(image_masks, expected_image_masks[:, :, roi_indices])

    def test_get_roi_response_traces(self, segmentation_extractor, expected_roi_response_traces):
        roi_response_traces = segmentation_extractor.get_roi_response_traces()
        for name, expected_trace in expected_roi_response_traces.items():
            np.testing.assert_array_equal(roi_response_traces[name], expected_trace)

    def test_get_summary_images(self, segmentation_extractor, expected_mean_image, expected_correlation_image):
        name_to_image = segmentation_extractor.get_summary_images()
        mean_image = name_to_image["mean"]
        correlation_image = name_to_image["correlation"]
        np.testing.assert_array_equal(mean_image, expected_mean_image)
        np.testing.assert_array_equal(correlation_image, expected_correlation_image)

    def test_get_num_frames(self, segmentation_extractor, expected_roi_response_traces):
        num_frames = segmentation_extractor.get_num_frames()
        assert num_frames == list(expected_roi_response_traces.values())[0].shape[0]
