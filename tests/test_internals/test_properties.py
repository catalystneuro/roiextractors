"""Test script for properties functionality in SegmentationExtractor."""

import numpy as np
import pytest
from roiextractors.testing import generate_dummy_segmentation_extractor


def test_basic_properties():
    """Test basic property functionality."""
    # Create dummy segmentation extractor
    extractor = generate_dummy_segmentation_extractor(num_rois=5, num_frames=10)

    # Test setting and getting properties
    roi_ids = extractor.get_roi_ids()
    quality_scores = np.array([0.8, 0.9, 0.7, 0.95, 0.6])
    extractor.set_property(key="quality", values=quality_scores, ids=roi_ids)

    retrieved_quality = extractor.get_property(key="quality", ids=roi_ids)
    assert np.array_equal(quality_scores, retrieved_quality), "Property values do not match"

    # Test get_property_keys
    assert "quality" in extractor.get_property_keys(), "Property key not found"

    print("✓ Basic property functionality test passed")


def test_multiple_properties():
    """Test setting multiple properties."""
    extractor = generate_dummy_segmentation_extractor(num_rois=3, num_frames=10)

    # Get ROI IDs
    roi_ids = extractor.get_roi_ids()

    # Set multiple properties
    quality_scores = np.array([0.8, 0.9, 0.7])
    areas = np.array([100, 150, 120])
    labels = np.array(["roi_1", "roi_2", "roi_3"])

    extractor.set_property(key="quality", values=quality_scores, ids=roi_ids)
    extractor.set_property(key="area", values=areas, ids=roi_ids)
    extractor.set_property(key="label", values=labels, ids=roi_ids)

    # Check all properties exist
    keys = extractor.get_property_keys()
    assert "quality" in keys, "Quality property not found"
    assert "area" in keys, "Area property not found"
    assert "label" in keys, "Label property not found"

    # Check values
    assert np.array_equal(extractor.get_property(key="quality", ids=roi_ids), quality_scores)
    assert np.array_equal(extractor.get_property(key="area", ids=roi_ids), areas)
    assert np.array_equal(extractor.get_property(key="label", ids=roi_ids), labels)


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

    print("✓ Property validation test passed")


def test_property_types():
    """Test different property data types."""
    extractor = generate_dummy_segmentation_extractor(num_rois=4, num_frames=10)
    roi_ids = extractor.get_roi_ids()

    # Test different data types
    float_property = np.array([1.5, 2.3, 3.7, 4.1])
    int_property = np.array([1, 2, 3, 4])
    bool_property = np.array([True, False, True, False])
    string_property = np.array(["a", "b", "c", "d"])

    extractor.set_property(key="float_prop", values=float_property, ids=roi_ids)
    extractor.set_property(key="int_prop", values=int_property, ids=roi_ids)
    extractor.set_property(key="bool_prop", values=bool_property, ids=roi_ids)
    extractor.set_property(key="string_prop", values=string_property, ids=roi_ids)

    # Check all properties are stored correctly
    assert np.array_equal(extractor.get_property(key="float_prop", ids=roi_ids), float_property)
    assert np.array_equal(extractor.get_property(key="int_prop", ids=roi_ids), int_property)
    assert np.array_equal(extractor.get_property(key="bool_prop", ids=roi_ids), bool_property)
    assert np.array_equal(extractor.get_property(key="string_prop", ids=roi_ids), string_property)

    print("✓ Property types test passed")


def test_empty_properties():
    """Test behavior with no properties set."""
    extractor = generate_dummy_segmentation_extractor(num_rois=3, num_frames=10)

    # Should start with empty properties
    assert len(extractor.get_property_keys()) == 0, "Should start with no properties"

    # Getting non-existent property should raise KeyError
    roi_ids = extractor.get_roi_ids()
    with pytest.raises(KeyError):
        extractor.get_property(key="nonexistent", ids=roi_ids)

    print("✓ Empty properties test passed")


def test_properties_with_different_id_order():
    """Test setting properties with IDs in different order."""
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

    print("✓ Properties with different ID order test passed")


def test_properties_with_invalid_ids():
    """Test validation when setting properties with invalid IDs."""
    extractor = generate_dummy_segmentation_extractor(num_rois=3, num_frames=10)
    roi_ids = extractor.get_roi_ids()

    # Test with IDs that don't match the extractor's ROI IDs
    with pytest.raises(ValueError, match="must match the extractor's ROI ids"):
        extractor.set_property(key="quality", values=[0.8, 0.9, 0.7], ids=[999, 998, 997])  # Invalid IDs

    # Test with partial mismatch
    mixed_ids = [roi_ids[0], roi_ids[1], 999]  # Two valid, one invalid
    with pytest.raises(ValueError, match="must match the extractor's ROI ids"):
        extractor.set_property(key="quality", values=[0.8, 0.9, 0.7], ids=mixed_ids)

    print("✓ Properties with invalid IDs validation test passed")


if __name__ == "__main__":
    test_basic_properties()
    test_multiple_properties()
    test_property_validation()
    test_property_types()
    test_empty_properties()
    test_properties_with_different_id_order()
    test_properties_with_invalid_ids()
    print("\n🎉 All properties tests passed!")
