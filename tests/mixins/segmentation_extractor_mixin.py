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

    def test_get_roi_indices(self, segmentation_extractor, expected_roi_ids):
        roi_indices = segmentation_extractor.get_roi_indices()
        expected_roi_indices = list(range(len(expected_roi_ids)))
        np.testing.assert_array_equal(roi_indices, expected_roi_indices)

    @pytest.mark.parametrize("expected_roi_indices", ([], [0], [0, 1], [0, 2]))
    def test_get_roi_indices_with_roi_ids(self, segmentation_extractor, expected_roi_ids, expected_roi_indices):
        roi_ids = [expected_roi_ids[i] for i in expected_roi_indices]
        roi_indices = segmentation_extractor.get_roi_indices(roi_ids=roi_ids)
        np.testing.assert_array_equal(roi_indices, expected_roi_indices)

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

    @pytest.mark.parametrize("roi_indices", ([], [0], [0, 1], [0, 2]))
    def test_get_roi_locations_with_roi_ids(
        self, segmentation_extractor, expected_roi_locations, expected_roi_ids, roi_indices
    ):
        roi_ids = [expected_roi_ids[i] for i in roi_indices]
        roi_locations = segmentation_extractor.get_roi_locations(roi_ids=roi_ids)
        np.testing.assert_array_equal(roi_locations, expected_roi_locations[:, roi_indices])

    def test_get_roi_image_masks(self, segmentation_extractor, expected_image_masks):
        image_masks = segmentation_extractor.get_roi_image_masks()
        np.testing.assert_array_equal(image_masks, expected_image_masks)

    @pytest.mark.parametrize("roi_indices", ([], [0], [0, 1], [0, 2]))
    def test_get_roi_image_masks_with_roi_ids(
        self, segmentation_extractor, expected_image_masks, expected_roi_ids, roi_indices
    ):
        roi_ids = [expected_roi_ids[i] for i in roi_indices]
        image_masks = segmentation_extractor.get_roi_image_masks(roi_ids=roi_ids)
        np.testing.assert_array_equal(image_masks, expected_image_masks[:, :, roi_indices])

    def test_get_roi_pixel_masks(self, segmentation_extractor, expected_image_masks):
        pixel_masks = segmentation_extractor.get_roi_pixel_masks()
        assert len(pixel_masks) == expected_image_masks.shape[2]
        for i, pixel_mask in enumerate(pixel_masks):
            expected_image_mask = expected_image_masks[:, :, i]
            expected_locs = np.where(expected_image_mask > 0)
            expected_values = expected_image_mask[expected_image_mask > 0]
            np.testing.assert_array_equal(pixel_mask[:, 0], expected_locs[0])
            np.testing.assert_array_equal(pixel_mask[:, 1], expected_locs[1])
            np.testing.assert_array_equal(pixel_mask[:, 2], expected_values)

    @pytest.mark.parametrize("roi_indices", ([], [0], [0, 1], [0, 2]))
    def test_get_roi_pixel_masks_with_roi_ids(
        self, segmentation_extractor, expected_image_masks, expected_roi_ids, roi_indices
    ):
        expected_image_masks_indexed = expected_image_masks[:, :, roi_indices]
        roi_ids = [expected_roi_ids[i] for i in roi_indices]
        pixel_masks = segmentation_extractor.get_roi_pixel_masks(roi_ids=roi_ids)
        assert len(pixel_masks) == len(roi_indices)
        for i, pixel_mask in enumerate(pixel_masks):
            expected_image_mask = expected_image_masks_indexed[:, :, i]
            expected_locs = np.where(expected_image_mask > 0)
            expected_values = expected_image_mask[expected_image_mask > 0]
            np.testing.assert_array_equal(pixel_mask[:, 0], expected_locs[0])
            np.testing.assert_array_equal(pixel_mask[:, 1], expected_locs[1])
            np.testing.assert_array_equal(pixel_mask[:, 2], expected_values)

    def test_get_roi_response_traces(self, segmentation_extractor, expected_roi_response_traces):
        roi_response_traces = segmentation_extractor.get_roi_response_traces()
        for name, expected_trace in expected_roi_response_traces.items():
            np.testing.assert_array_equal(roi_response_traces[name], expected_trace)

    @pytest.mark.parametrize("roi_indices", ([], [0], [0, 1], [0, 2]))
    def test_get_roi_response_traces_with_roi_ids(
        self, segmentation_extractor, expected_roi_response_traces, expected_roi_ids, roi_indices
    ):
        roi_ids = [expected_roi_ids[i] for i in roi_indices]
        roi_response_traces = segmentation_extractor.get_roi_response_traces(roi_ids=roi_ids)
        for name, trace in roi_response_traces.items():
            expected_trace = expected_roi_response_traces[name][:, roi_indices]
            np.testing.assert_array_equal(trace, expected_trace)

    @pytest.mark.parametrize("start_frame, end_frame", [(0, 1), (1, 3)])
    def test_get_roi_response_traces_with_frames(
        self, segmentation_extractor, expected_roi_response_traces, start_frame, end_frame
    ):
        roi_response_traces = segmentation_extractor.get_roi_response_traces(
            start_frame=start_frame, end_frame=end_frame
        )
        for name, trace in roi_response_traces.items():
            expected_trace = expected_roi_response_traces[name][start_frame:end_frame, :]
            np.testing.assert_array_equal(trace, expected_trace)

    @pytest.mark.parametrize("names", ([], ["raw"], ["dff"], ["raw", "dff"]))
    def test_get_roi_response_traces_with_names(self, segmentation_extractor, expected_roi_response_traces, names):
        roi_response_traces = segmentation_extractor.get_roi_response_traces(names=names)
        assert list(roi_response_traces.keys()) == names
        for name, trace in roi_response_traces.items():
            expected_trace = expected_roi_response_traces[name]
            np.testing.assert_array_equal(trace, expected_trace)

    def test_get_background_ids(self, segmentation_extractor, expected_background_ids):
        background_ids = segmentation_extractor.get_background_ids()
        np.testing.assert_array_equal(background_ids, expected_background_ids)

    def test_get_num_background_components(self, segmentation_extractor, expected_background_ids):
        num_background_components = segmentation_extractor.get_num_background_components()
        assert num_background_components == len(expected_background_ids)

    def test_get_background_image_masks(self, segmentation_extractor, expected_background_image_masks):
        background_image_masks = segmentation_extractor.get_background_image_masks()
        np.testing.assert_array_equal(background_image_masks, expected_background_image_masks)

    @pytest.mark.parametrize("background_indices", ([], [0], [0, 1], [0, 2]))
    def test_get_background_image_masks_with_background_ids(
        self, segmentation_extractor, expected_background_image_masks, expected_background_ids, background_indices
    ):
        expected_background_image_masks_indexed = expected_background_image_masks[:, :, background_indices]
        background_ids = [expected_background_ids[i] for i in background_indices]
        background_image_masks = segmentation_extractor.get_background_image_masks(background_ids=background_ids)
        np.testing.assert_array_equal(background_image_masks, expected_background_image_masks_indexed)

    def test_get_background_pixel_masks(self, segmentation_extractor, expected_background_image_masks):
        pixel_masks = segmentation_extractor.get_background_pixel_masks()
        for i, pixel_mask in enumerate(pixel_masks):
            expected_image_mask = expected_background_image_masks[:, :, i]
            expected_locs = np.where(expected_image_mask > 0)
            expected_values = expected_image_mask[expected_image_mask > 0]
            np.testing.assert_array_equal(pixel_mask[:, 0], expected_locs[0])
            np.testing.assert_array_equal(pixel_mask[:, 1], expected_locs[1])
            np.testing.assert_array_equal(pixel_mask[:, 2], expected_values)

    @pytest.mark.parametrize("background_indices", ([], [0], [0, 1], [0, 2]))
    def test_get_background_pixel_masks_with_background_ids(
        self, segmentation_extractor, expected_background_image_masks, expected_background_ids, background_indices
    ):
        expected_background_image_masks_indexed = expected_background_image_masks[:, :, background_indices]
        background_ids = [expected_background_ids[i] for i in background_indices]
        pixel_masks = segmentation_extractor.get_background_pixel_masks(background_ids=background_ids)
        assert len(pixel_masks) == len(background_indices)
        for i, pixel_mask in enumerate(pixel_masks):
            expected_image_mask = expected_background_image_masks_indexed[:, :, i]
            expected_locs = np.where(expected_image_mask > 0)
            expected_values = expected_image_mask[expected_image_mask > 0]
            np.testing.assert_array_equal(pixel_mask[:, 0], expected_locs[0])
            np.testing.assert_array_equal(pixel_mask[:, 1], expected_locs[1])
            np.testing.assert_array_equal(pixel_mask[:, 2], expected_values)

    def test_get_background_response_traces(self, segmentation_extractor, expected_background_response_traces):
        background_response_traces = segmentation_extractor.get_background_response_traces()
        for name, expected_trace in expected_background_response_traces.items():
            np.testing.assert_array_equal(background_response_traces[name], expected_trace)

    @pytest.mark.parametrize("background_indices", ([], [0], [0, 1], [0, 2]))
    def test_get_background_response_traces_with_background_components(
        self,
        segmentation_extractor,
        expected_background_response_traces,
        expected_background_ids,
        background_indices,
    ):
        background_ids = [expected_background_ids[i] for i in background_indices]
        background_response_traces = segmentation_extractor.get_background_response_traces(
            background_ids=background_ids
        )
        for name, trace in background_response_traces.items():
            expected_trace = expected_background_response_traces[name][:, background_indices]
            np.testing.assert_array_equal(trace, expected_trace)

    @pytest.mark.parametrize("start_frame, end_frame", [(0, 1), (1, 3)])
    def test_get_background_response_traces_with_frames(
        self, segmentation_extractor, expected_background_response_traces, start_frame, end_frame
    ):
        background_response_traces = segmentation_extractor.get_background_response_traces(
            start_frame=start_frame, end_frame=end_frame
        )
        for name, trace in background_response_traces.items():
            expected_trace = expected_background_response_traces[name][start_frame:end_frame, :]
            np.testing.assert_array_equal(trace, expected_trace)

    @pytest.mark.parametrize("names", ([], ["background"]))
    def test_get_background_response_traces_with_names(
        self, segmentation_extractor, expected_background_response_traces, names
    ):
        background_response_traces = segmentation_extractor.get_background_response_traces(names=names)
        assert list(background_response_traces.keys()) == names
        for name, trace in background_response_traces.items():
            expected_trace = expected_background_response_traces[name]
            np.testing.assert_array_equal(trace, expected_trace)

    def test_get_summary_images(self, segmentation_extractor, expected_summary_images):
        summary_images = segmentation_extractor.get_summary_images()
        for name, expected_image in expected_summary_images.items():
            np.testing.assert_array_equal(summary_images[name], expected_image)

    @pytest.mark.parametrize("names", ([], ["mean"], ["correlation"], ["mean", "correlation"]))
    def test_get_summary_images_with_names(self, segmentation_extractor, expected_summary_images, names):
        summary_images = segmentation_extractor.get_summary_images(names=names)
        assert list(summary_images.keys()) == names
        for name, image in summary_images.items():
            expected_image = expected_summary_images[name]
            np.testing.assert_array_equal(image, expected_image)
