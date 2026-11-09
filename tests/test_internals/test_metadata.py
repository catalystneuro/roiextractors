"""Tests for the `metadata` of imaging extractors."""

import re
from dataclasses import replace

import pytest

from roiextractors._metadata import Metadata
from roiextractors.testing import generate_dummy_imaging_extractor


def test_reading_metadata_warns_that_it_is_experimental():
    """Every read of `metadata`, on an extractor or on a view, warns that the metadata API is experimental."""
    imaging = generate_dummy_imaging_extractor(num_samples=5, num_rows=4, num_columns=4)
    expected_message = (
        "The metadata API is experimental and has not stabilized: the fields of `metadata` can change or be "
        "removed in any release."
    )

    with pytest.warns(UserWarning, match=re.escape(expected_message)) as warning_records:
        imaging.metadata
        imaging.metadata
        imaging.slice_samples(start_sample=1).metadata

    assert len(warning_records) == 3


def test_views_have_the_metadata_of_their_part():
    """Each view has the metadata of its part of the parent.

    A slice of samples has the metadata of the parent, a crop the field of view of its rows and columns, and a plane
    its own z offset, with the grid spacing and the field of view of a plane.
    """
    volumetric = generate_dummy_imaging_extractor(num_samples=5, num_rows=4, num_columns=8, num_planes=3)
    # The mock records no metadata, so the fields are set here
    volumetric.metadata.microscope_name = "Microscope"
    volumetric.metadata.grid_spacing_um = [5.0, 2.0, 1.0]
    volumetric.metadata.field_of_view_um = [10.0, 8.0, 8.0]
    volumetric.metadata.plane_z_offsets_um = [0.0, 5.0, 10.0]

    sample_slice = volumetric.slice_samples(start_sample=1)
    cropped_extractor = volumetric.slice_field_of_view(row_end=2, column_end=2)
    plane_extractor = volumetric.select_plane(1)

    assert sample_slice.metadata == volumetric.metadata
    # Two of the four rows and two of the eight columns
    expected_cropped_metadata = replace(volumetric.metadata, field_of_view_um=[10.0, 4.0, 2.0])
    assert cropped_extractor.metadata == expected_cropped_metadata
    expected_plane_metadata = replace(
        volumetric.metadata, grid_spacing_um=[2.0, 1.0], field_of_view_um=[8.0, 8.0], plane_z_offsets_um=[5.0]
    )
    assert plane_extractor.metadata == expected_plane_metadata


def test_to_populated_dict():
    """`to_populated_dict()` returns the fields that are set, with `extra_metadata` unless it is left out."""
    metadata = Metadata(microscope_name="Microscope", subject_id="mouse_1", extra_metadata={"format_key": "value"})

    populated_metadata = metadata.to_populated_dict()
    populated_metadata_without_extra = metadata.to_populated_dict(include_extra=False)

    expected_populated_metadata_without_extra = {"microscope_name": "Microscope", "subject_id": "mouse_1"}
    expected_populated_metadata = {
        **expected_populated_metadata_without_extra,
        "extra_metadata": {"format_key": "value"},
    }
    assert populated_metadata == expected_populated_metadata
    assert populated_metadata_without_extra == expected_populated_metadata_without_extra
    assert Metadata().to_populated_dict() == {}
