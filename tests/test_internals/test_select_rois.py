import numpy as np
import pytest
from numpy.testing import assert_array_equal

from roiextractors.testing import generate_dummy_segmentation_extractor


class TestBasicRoiSelection:
    """Tests for basic ROI selection functionality."""

    def test_basic_roi_selection(self):
        """Test basic ROI selection with explicit IDs."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=10, num_samples=30)
        roi_ids = segmentation.get_roi_ids()

        selected = segmentation.select_rois(roi_ids[:3])

        assert selected.get_num_rois() == 3
        assert selected.get_roi_ids() == roi_ids[:3]
        assert selected.get_num_samples() == 30

    def test_roi_selection_preserves_order(self):
        """Test that ROI selection preserves the order of IDs provided."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=10)
        roi_ids = segmentation.get_roi_ids()

        # Select in reverse order
        reversed_ids = list(reversed(roi_ids[:3]))
        selected = segmentation.select_rois(reversed_ids)

        assert selected.get_roi_ids() == reversed_ids

    def test_single_roi_selection(self):
        """Test selecting a single ROI."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=10)
        roi_ids = segmentation.get_roi_ids()

        selected = segmentation.select_rois([roi_ids[2]])

        assert selected.get_num_rois() == 1
        assert selected.get_roi_ids() == [roi_ids[2]]


class TestRoiSelectionTraces:
    """Tests for trace data with ROI selection."""

    def test_get_traces_returns_selected_rois_only(self):
        """Test that get_traces returns only selected ROI data."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=10)
        roi_ids = segmentation.get_roi_ids()

        selected = segmentation.select_rois(roi_ids[:2])
        traces = selected.get_traces()

        assert traces.shape == (10, 2)  # 10 samples, 2 ROIs

        # Verify data matches parent's filtered data
        expected = segmentation.get_traces(roi_ids=roi_ids[:2])
        assert_array_equal(traces, expected)

    def test_get_traces_with_specific_roi_ids(self):
        """Test get_traces with roi_ids parameter subset."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=10)
        roi_ids = segmentation.get_roi_ids()

        selected = segmentation.select_rois(roi_ids[:3])
        traces = selected.get_traces(roi_ids=[roi_ids[1]])

        assert traces.shape == (10, 1)

    def test_get_traces_invalid_roi_id_raises(self):
        """Test that requesting non-selected ROI raises error."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=10)
        roi_ids = segmentation.get_roi_ids()

        selected = segmentation.select_rois(roi_ids[:2])

        with pytest.raises(ValueError, match="not found"):
            selected.get_traces(roi_ids=[roi_ids[3]])

    def test_get_traces_with_frame_range(self):
        """Test get_traces with start_frame and end_frame parameters."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=20)
        roi_ids = segmentation.get_roi_ids()

        selected = segmentation.select_rois(roi_ids[:2])
        traces = selected.get_traces(start_frame=5, end_frame=15)

        assert traces.shape == (10, 2)  # 10 frames, 2 ROIs


class TestRoiSelectionMasks:
    """Tests for mask data with ROI selection."""

    def test_get_roi_image_masks(self):
        """Test that image masks are filtered correctly."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_rows=10, num_columns=10)
        roi_ids = segmentation.get_roi_ids()

        selected = segmentation.select_rois(roi_ids[:2])
        masks = selected.get_roi_image_masks()

        assert masks.shape == (10, 10, 2)  # height x width x 2 ROIs

    def test_get_roi_pixel_masks(self):
        """Test that pixel masks are filtered correctly."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_rows=10, num_columns=10)
        roi_ids = segmentation.get_roi_ids()

        selected = segmentation.select_rois(roi_ids[:2])
        pixel_masks = selected.get_roi_pixel_masks()

        assert len(pixel_masks) == 2


class TestRoiSelectionValidation:
    """Tests for validation and edge cases."""

    def test_empty_roi_ids_raises(self):
        """Test that empty roi_ids raises ValueError."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=10)

        with pytest.raises(ValueError, match="cannot be empty"):
            segmentation.select_rois([])

    def test_invalid_roi_id_raises(self):
        """Test that invalid ROI IDs raise ValueError."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=10)

        with pytest.raises(ValueError, match="not found in extractor"):
            segmentation.select_rois(["invalid_id"])

    def test_partial_invalid_ids_raises(self):
        """Test that partially invalid IDs raise ValueError."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=10)
        roi_ids = segmentation.get_roi_ids()

        with pytest.raises(ValueError, match="not found in extractor"):
            segmentation.select_rois([roi_ids[0], "invalid_id"])


class TestRoiSelectionComposition:
    """Tests for composing ROI selection with other operations."""

    def test_select_rois_then_slice_samples(self):
        """Test ROI selection followed by temporal slicing."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=100)
        roi_ids = segmentation.get_roi_ids()

        subset = segmentation.select_rois(roi_ids[:3]).slice_samples(10, 30)

        assert subset.get_num_rois() == 3
        assert subset.get_num_samples() == 20

        traces = subset.get_traces()
        assert traces.shape == (20, 3)

    def test_slice_samples_then_select_rois(self):
        """Test temporal slicing followed by ROI selection."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=100)
        roi_ids = segmentation.get_roi_ids()

        subset = segmentation.slice_samples(10, 30).select_rois(roi_ids[:3])

        assert subset.get_num_rois() == 3
        assert subset.get_num_samples() == 20

        traces = subset.get_traces()
        assert traces.shape == (20, 3)

    def test_composition_order_gives_same_result(self):
        """Test that different composition orders give same result."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=100)
        roi_ids = segmentation.get_roi_ids()

        subset1 = segmentation.select_rois(roi_ids[:3]).slice_samples(10, 30)
        subset2 = segmentation.slice_samples(10, 30).select_rois(roi_ids[:3])

        assert_array_equal(subset1.get_traces(), subset2.get_traces())
        assert_array_equal(subset1.get_roi_image_masks(), subset2.get_roi_image_masks())

    def test_double_roi_selection(self):
        """Test chaining multiple ROI selections."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=10, num_samples=30)
        roi_ids = segmentation.get_roi_ids()

        # First selection: 5 ROIs
        first = segmentation.select_rois(roi_ids[:5])
        # Second selection: 2 of those 5
        second = first.select_rois(roi_ids[:2])

        assert second.get_num_rois() == 2
        assert second.get_roi_ids() == roi_ids[:2]


class TestRoiSelectionTimestamps:
    """Tests for timestamp handling with ROI selection."""

    def test_timestamps_preserved(self):
        """Test that timestamps are preserved after ROI selection."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=10)
        times = np.arange(10) + 5.5
        segmentation.set_times(times)

        selected = segmentation.select_rois(segmentation.get_roi_ids()[:2])

        assert selected.has_time_vector()
        assert_array_equal(selected.get_timestamps(), times)

    def test_sampling_frequency_preserved(self):
        """Test that sampling frequency is preserved."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=10, sampling_frequency=20.0)

        selected = segmentation.select_rois(segmentation.get_roi_ids()[:2])

        assert selected.get_sampling_frequency() == 20.0


class TestRoiSelectionSpatialProperties:
    """Tests for spatial properties with ROI selection."""

    def test_frame_shape_preserved(self):
        """Test that frame shape is preserved."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_rows=50, num_columns=60)

        selected = segmentation.select_rois(segmentation.get_roi_ids()[:2])

        assert tuple(selected.get_frame_shape()) == (50, 60)

    def test_summary_images_preserved(self):
        """Test that summary images are preserved."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, has_summary_images=True)

        selected = segmentation.select_rois(segmentation.get_roi_ids()[:2])

        images = selected.get_images_dict()
        assert len(images) > 0

    def test_num_planes_preserved(self):
        """Test that number of planes is preserved."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5)

        selected = segmentation.select_rois(segmentation.get_roi_ids()[:2])

        assert selected.get_num_planes() == segmentation.get_num_planes()


class TestRoiSelectionTraceTypes:
    """Tests for different trace types with ROI selection."""

    def test_get_dff_traces(self):
        """Test getting dff traces from selected ROIs."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=10, has_dff_signal=True)
        roi_ids = segmentation.get_roi_ids()

        selected = segmentation.select_rois(roi_ids[:2])
        dff_traces = selected.get_traces(name="dff")

        if dff_traces is not None:
            assert dff_traces.shape == (10, 2)

    def test_get_traces_dict_preserved(self):
        """Test that traces_dict is accessible from selected extractor."""
        segmentation = generate_dummy_segmentation_extractor(num_rois=5, num_samples=10)

        selected = segmentation.select_rois(segmentation.get_roi_ids()[:2])
        traces_dict = selected.get_traces_dict()

        assert isinstance(traces_dict, dict)
