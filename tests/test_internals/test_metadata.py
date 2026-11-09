"""Tests for the `metadata` of imaging extractors."""

from roiextractors._metadata import Metadata
from roiextractors.testing import generate_dummy_imaging_extractor


def test_views_have_the_metadata_of_the_parent():
    """The views from `slice_samples()`, `slice_field_of_view()` and `select_plane()` have the metadata of the parent."""
    volumetric = generate_dummy_imaging_extractor(num_samples=5, num_rows=4, num_columns=4, num_planes=3)
    # The mock records no metadata, so a field is set to tell the metadata of the parent from an empty one
    volumetric.metadata.device_name = "Microscope"

    sample_slice = volumetric.slice_samples(start_sample=1)
    field_of_view_slice = volumetric.slice_field_of_view(row_end=2, column_end=2)
    plane_extractor = volumetric.select_plane(0)

    assert sample_slice.metadata == volumetric.metadata
    assert field_of_view_slice.metadata == volumetric.metadata
    assert plane_extractor.metadata == volumetric.metadata


def test_to_populated_dict():
    """`to_populated_dict()` returns the fields that are set, with `extra_metadata` unless it is left out."""
    metadata = Metadata(device_name="Microscope", subject_id="mouse_1", extra_metadata={"format_key": "value"})

    populated_metadata = metadata.to_populated_dict()
    populated_metadata_without_extra = metadata.to_populated_dict(include_extra=False)

    expected_populated_metadata_without_extra = {"device_name": "Microscope", "subject_id": "mouse_1"}
    expected_populated_metadata = {
        **expected_populated_metadata_without_extra,
        "extra_metadata": {"format_key": "value"},
    }
    assert populated_metadata == expected_populated_metadata
    assert populated_metadata_without_extra == expected_populated_metadata_without_extra
    assert Metadata().to_populated_dict() == {}
