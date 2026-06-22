"""Unit tests for ScanImage multi-ROI ROI-list parsing and selection.

These tests exercise the pure ROI-list parsing (``_get_roi_list``, ``_get_imaging_scanfields``) and
selector-resolution (``_resolve_roi_index``) logic with synthetic ROI metadata, so they run in the
minimal (no-data) test job.
"""

import pytest

from roiextractors import ScanImageMultiROIImagingExtractor

_resolve = ScanImageMultiROIImagingExtractor._resolve_roi_index
_get_roi_list = ScanImageMultiROIImagingExtractor._get_roi_list
_get_imaging_scanfields = ScanImageMultiROIImagingExtractor._get_imaging_scanfields


def _metadata_with_rois(rois) -> dict:
    """Wrap a ROI list (or single ROI dict) in the nested scanimage_metadata structure."""
    return {"RoiGroups": {"imagingRoiGroup": {"rois": rois}}}


# Synthetic ROI list: note "ROI dup" appears twice (duplicate name), mirroring real recordings.
ROIS = [
    {"name": "ROI A", "roiUuid": "AAAA1111"},
    {"name": "ROI dup", "roiUuid": "BBBB2222"},
    {"name": "ROI dup", "roiUuid": "CCCC3333"},
    {"name": "ROI D", "roiUuid": "DDDD4444"},
]


def test_defaults_to_first_roi():
    """With no selector, the first ROI (index 0) is used."""
    assert _resolve(ROIS) == 0


def test_select_by_index():
    """An explicit in-range index is returned as-is."""
    assert _resolve(ROIS, roi_index=2) == 2


def test_index_out_of_range_raises():
    """An out-of-range index raises with the valid bounds."""
    with pytest.raises(ValueError, match="roi_index"):
        _resolve(ROIS, roi_index=4)


def test_select_by_unique_name():
    """A unique name resolves to its index."""
    assert _resolve(ROIS, roi_name="ROI D") == 3


def test_ambiguous_name_raises_with_indices():
    """A duplicated name is rejected, and the error names the conflicting indices."""
    with pytest.raises(ValueError, match=r"ambiguous.*\[1, 2\]"):
        _resolve(ROIS, roi_name="ROI dup")


def test_unknown_name_raises():
    """A name that matches no ROI raises a not-found error."""
    with pytest.raises(ValueError, match="No ROI named"):
        _resolve(ROIS, roi_name="ROI Z")


def test_select_by_uuid():
    """A uuid resolves to its index."""
    assert _resolve(ROIS, roi_uuid="CCCC3333") == 2


def test_select_by_uuid_is_case_insensitive():
    """UUID matching ignores hex case."""
    assert _resolve(ROIS, roi_uuid="cccc3333") == 2


def test_unknown_uuid_raises():
    """A uuid that matches no ROI raises a not-found error."""
    with pytest.raises(ValueError, match="No ROI with roiUuid"):
        _resolve(ROIS, roi_uuid="9999FFFF")


def test_multiple_selectors_raises():
    """Giving more than one selector is rejected."""
    with pytest.raises(ValueError, match="at most one"):
        _resolve(ROIS, roi_index=0, roi_name="ROI A")


def test_get_roi_list_normalizes_single_roi_dict():
    """A single ROI stored as a bare dict is normalized to a one-element list."""
    metadata = _metadata_with_rois({"name": "ROI A", "roiUuid": "AAAA1111"})
    rois = _get_roi_list(metadata)
    assert isinstance(rois, list)
    assert len(rois) == 1
    assert rois[0]["name"] == "ROI A"


def test_get_roi_list_excludes_disabled_rois():
    """Disabled ROIs are dropped; ROIs missing the 'enable' key are treated as enabled."""
    metadata = _metadata_with_rois(
        [
            {"name": "kept-no-key"},  # missing 'enable' -> enabled
            {"name": "kept-enabled", "enable": True},
            {"name": "dropped", "enable": False},
        ]
    )
    names = [roi["name"] for roi in _get_roi_list(metadata)]
    assert names == ["kept-no-key", "kept-enabled"]


def test_get_imaging_scanfields_returns_one_per_roi():
    """Each enabled ROI contributes its single planar scanfield, in stacking order."""
    metadata = _metadata_with_rois(
        [
            {"name": "ROI A", "scanfields": {"pixelResolutionXY": [512, 100]}},
            {"name": "ROI B", "scanfields": {"pixelResolutionXY": [512, 200]}},
        ]
    )
    scanfields = _get_imaging_scanfields(metadata)
    assert [scanfield["pixelResolutionXY"] for scanfield in scanfields] == [[512, 100], [512, 200]]


def test_get_imaging_scanfields_rejects_per_plane_scanfields():
    """A ROI carrying a list of per-plane scanfields is volumetric and is rejected."""
    metadata = _metadata_with_rois(
        [{"name": "ROI A", "scanfields": [{"pixelResolutionXY": [512, 100]}, {"pixelResolutionXY": [512, 100]}]}]
    )
    with pytest.raises(NotImplementedError, match="[Vv]olumetric"):
        _get_imaging_scanfields(metadata)
