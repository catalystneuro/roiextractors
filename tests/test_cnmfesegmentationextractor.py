"""Tests for CnmfeSegmentationExtractor."""

import numpy as np
import pytest

from roiextractors.extractors.schnitzerextractor import CnmfeSegmentationExtractor

from .setup_paths import OPHYS_DATA_PATH


class TestCnmfeSegmentationExtractor:
    """Tests for CNMFE segmentation extractor with test data from GIN."""

    @pytest.fixture(scope="class")
    def extractor(self):
        """Create extractor instance for testing."""
        file_path = (
            OPHYS_DATA_PATH / "segmentation_datasets" / "cnmfe" / "2014_04_01_p203_m19_check01_cnmfeAnalysis.mat"
        )
        return CnmfeSegmentationExtractor(file_path=file_path)

    def test_get_frame_shape(self, extractor):
        """Test that get_frame_shape() returns correct shape."""
        frame_shape = extractor.get_frame_shape()

        # Verify the frame shape is returned correctly
        assert len(frame_shape) == 2, "Frame shape should be 2D (height, width)"
        assert isinstance(frame_shape[0], (int, np.integer)), "Height should be an integer"
        assert isinstance(frame_shape[1], (int, np.integer)), "Width should be an integer"

        # Frame shape should be positive integers
        assert frame_shape[0] > 0, "Height should be positive"
        assert frame_shape[1] > 0, "Width should be positive"

        # Known dimensions for this test file
        assert frame_shape == (250, 250), "Expected frame shape is (250, 250)"

    def test_get_num_rois(self, extractor):
        """Test that get_num_rois() returns a valid number."""
        num_rois = extractor.get_num_rois()
        assert isinstance(num_rois, int), "Number of ROIs should be an integer"
        assert num_rois > 0, "Should have at least one ROI"

    def test_get_roi_ids(self, extractor):
        """Test that get_roi_ids() returns valid IDs."""
        roi_ids = extractor.get_roi_ids()
        assert isinstance(roi_ids, list), "ROI IDs should be a list"
        assert len(roi_ids) == extractor.get_num_rois(), "Number of IDs should match number of ROIs"

    def test_get_traces(self, extractor):
        """Test that get_traces() returns valid trace data."""
        traces = extractor.get_traces()
        assert traces is not None, "Traces should not be None"
        assert isinstance(traces, np.ndarray), "Traces should be a numpy array"
        assert traces.ndim == 2, "Traces should be 2D (num_samples x num_rois)"
        assert traces.shape[1] == extractor.get_num_rois(), "Number of trace columns should match number of ROIs"
