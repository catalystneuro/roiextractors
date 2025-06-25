import numpy as np
import pytest
from roiextractors.testing import generate_dummy_segmentation_extractor


def test_basic_properties():
    """Test basic property functionality."""
    extractor = generate_dummy_segmentation_extractor(num_rois=5, num_frames=10)

    # Test setting and getting properties
    roi_ids = extractor.get_roi_ids()
    quality_scores = np.array([0.8, 0.9, 0.7, 0.95, 0.6])
    extractor.set_property(key="quality", values=quality_scores, ids=roi_ids)

    retrieved_quality = extractor.get_property(key="quality", ids=roi_ids)
    assert np.array_equal(quality_scores, retrieved_quality), "Property values do not match"

    # Test get_property_keys
    assert "quality" in extractor.get_property_keys(), "Property key not found"


def test_multiple_properties_and_types():
    """Test setting multiple properties."""
    extractor = generate_dummy_segmentation_extractor(num_rois=3, num_frames=10)

    # Get ROI IDs
    roi_ids = extractor.get_roi_ids()

    # Set multiple properties of different data types
    quality_scores = np.array([0.8, 0.9, 0.7])
    areas = np.array([100, 150, 120])
    labels = np.array(["roi_1", "roi_2", "roi_3"])
    active = np.array([True, False, True])

    extractor.set_property(key="quality", values=quality_scores, ids=roi_ids)
    extractor.set_property(key="area", values=areas, ids=roi_ids)
    extractor.set_property(key="label", values=labels, ids=roi_ids)
    extractor.set_property(key="active", values=active, ids=roi_ids)

    # Check all properties exist
    keys = extractor.get_property_keys()
    assert "quality" in keys, "Quality property not found"
    assert "area" in keys, "Area property not found"
    assert "label" in keys, "Label property not found"
    assert "active" in keys, "Active property not found"

    # Check values
    assert np.array_equal(extractor.get_property(key="quality", ids=roi_ids), quality_scores)
    assert np.array_equal(extractor.get_property(key="area", ids=roi_ids), areas)
    assert np.array_equal(extractor.get_property(key="label", ids=roi_ids), labels)
    assert np.array_equal(extractor.get_property(key="active", ids=roi_ids), active)


def test_property_validation():
    """Test property validation."""
    extractor = generate_dummy_segmentation_extractor(num_rois=5, num_frames=10)
    roi_ids = extractor.get_roi_ids()

    # Test wrong length values should raise error
    wrong_length_values = np.array([1, 2, 3])  # Only 3 values for 5 ROIs
    with pytest.raises(ValueError, match="must match number of ROIs"):
        extractor.set_property(key="wrong_length", values=wrong_length_values, ids=roi_ids)

    # Test wrong length ids should raise error
    correct_values = np.array([1, 2, 3, 4, 5])
    wrong_length_ids = np.array([0, 1, 2])  # Only 3 IDs for 5 ROIs
    with pytest.raises(ValueError, match="must match number of ROIs"):
        extractor.set_property(key="wrong_length", values=correct_values, ids=wrong_length_ids)

    # Test with IDs that don't match the extractor's ROI IDs
    with pytest.raises(ValueError, match="must match the extractor's ROI ids"):
        extractor.set_property(
            key="quality", values=[0.8, 0.9, 0.7, 0.5, 0.4], ids=[999, 998, 997, 996, 995]
        )  # Invalid IDs

    # Test with partial mismatch
    mixed_ids = [roi_ids[0], roi_ids[1], roi_ids[2], roi_ids[3], 999]  # Four valid, one invalid
    with pytest.raises(ValueError, match="must match the extractor's ROI ids"):
        extractor.set_property(key="quality", values=[0.8, 0.9, 0.7, 0.5, 0.4], ids=mixed_ids)


def test_empty_properties():
    """Test mismatch in property name."""
    extractor = generate_dummy_segmentation_extractor(num_rois=3, num_frames=10)

    roi_ids = extractor.get_roi_ids()
    extractor.set_property(key="quality", values=[0.8, 0.9, 0.7], ids=roi_ids)

    # Try to get a property with similar name that doesn't exist, the error should tell the user what is available
    with pytest.raises(KeyError, match="Property 'quality_score' not found. Available properties: \\['quality'\\]"):
        extractor.get_property(key="quality_score", ids=roi_ids)


def test_that_positional_order_does_not_matter():
    """Test that the property interface is ID-based, not position-based.

    The key concept: When setting properties, what matters is which VALUE goes with which ROI ID,
    not the positional order in the arrays. The property system should correctly map values to
    their corresponding ROI IDs regardless of the order they are provided in.

    This is because positional order is an internal implementation detail and should not affect
    how properties are set or retrieved. The mapping should be based on the IDs provided which is the
    user facing interface.

    For example:
    - If ROI 'A' should have value 10 and ROI 'B' should have value 20
    - It doesn't matter if you provide ids=['A', 'B'] with values=[10, 20]
    - Or ids=['B', 'A'] with values=[20, 10]
    - Both should result in the same final mapping: A->10, B->20

    This test verifies this ID-based interface works correctly.
    """
    extractor = generate_dummy_segmentation_extractor(num_rois=5, num_frames=10)

    # Get the ROI IDs from the extractor
    roi_ids = extractor.get_roi_ids()

    # Shuffle the order of IDs and corresponding values
    shuffled_ids = [roi_ids[2], roi_ids[0], roi_ids[4], roi_ids[1], roi_ids[3]]
    quality_values = np.array([0.7, 0.8, 0.6, 0.9, 0.95])  # Values corresponding to shuffled IDs

    extractor.set_property(key="quality", values=quality_values, ids=shuffled_ids)

    # Get the full property array (should be in extractor's original order)
    full_quality = extractor.get_property(key="quality", ids=roi_ids)

    # Check that values are stored in the correct positions according to extractor's ROI order
    assert full_quality[0] == 0.8, "First ROI quality not set correctly"  # roi_ids[0] -> 0.8
    assert full_quality[1] == 0.9, "Second ROI quality not set correctly"  # roi_ids[1] -> 0.9
    assert full_quality[2] == 0.7, "Third ROI quality not set correctly"  # roi_ids[2] -> 0.7
    assert full_quality[3] == 0.95, "Fourth ROI quality not set correctly"  # roi_ids[3] -> 0.95
    assert full_quality[4] == 0.6, "Fifth ROI quality not set correctly"  # roi_ids[4] -> 0.6

    # Test getting subset of properties
    subset_ids = [roi_ids[1], roi_ids[3]]
    subset_quality = extractor.get_property(key="quality", ids=subset_ids)
    assert len(subset_quality) == 2, "Subset should have 2 values"
    assert subset_quality[0] == 0.9, "First subset value incorrect"  # roi_ids[1] -> 0.9
    assert subset_quality[1] == 0.95, "Second subset value incorrect"  # roi_ids[3] -> 0.95
