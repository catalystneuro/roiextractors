"""End-to-end tests for ``ScanImageMultiROIImagingExtractor`` against real planar mROI fixtures.

These tests exercise the full construction and cropping path against any ScanImage multi-ROI
(mesoscope) TIFF dropped under ``tests/fixtures/``. They carry no hard-coded ground truth: every
expectation is derived from each file's own ScanImage metadata, and the assertions check structural
invariants and relationships (e.g. cropped data matches the raw page rows, duplicate ROI names are
rejected when used as a selector). Any planar, equal-width multi-ROI file of the same structure is
discovered automatically and run through the same checks.
"""

import functools
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tifffile import TiffReader

from roiextractors import ScanImageMultiROIImagingExtractor
from roiextractors.extractors.tiffimagingextractors.scanimagetiff_utils import (
    read_scanimage_metadata,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@functools.lru_cache(maxsize=None)
def _page_info(path_str: str) -> tuple[int, tuple[int, int], np.dtype]:
    """Return (num_pages, page_shape, dtype) read directly from the file (cached).

    The raw pixel grid is the independent oracle for the cropping tests and no repo helper exposes
    it, so this and ``_raw_stack`` are the only places tifffile is used directly.
    """
    with TiffReader(path_str) as reader:
        return len(reader.pages), reader.pages[0].shape, reader.pages[0].dtype


@functools.lru_cache(maxsize=None)
def _raw_stack(path_str: str) -> np.ndarray:
    """The full uncropped page stack read directly with tifffile, for crop comparisons (cached)."""
    with TiffReader(path_str) as reader:
        return np.stack([page.asarray() for page in reader.pages])


def _is_planar_mroi(path_str: str) -> bool:
    """Whether a file is a planar, equal-width multi-ROI ScanImage acquisition."""
    try:
        metadata = read_scanimage_metadata(path_str)
    except Exception:
        return False
    frame_data = metadata["scan_image_non_varying_frame_metadata"]
    if not frame_data.get("SI.hRoiManager.mroiEnable", 0):
        return False
    # Volumetric data (stack enabled with more than one slice) is out of scope.
    if frame_data.get("SI.hStackManager.enable", False) and frame_data.get("SI.hStackManager.numSlices", 1) > 1:
        return False
    try:
        # `_get_imaging_scanfields` raises NotImplementedError on per-plane (volumetric) scanfields.
        scanfields = ScanImageMultiROIImagingExtractor._get_imaging_scanfields(
            metadata["scan_image_roi_group_metadata"]
        )
    except (KeyError, TypeError, NotImplementedError):
        return False
    widths = {scanfield["pixelResolutionXY"][0] for scanfield in scanfields}
    return len(widths) == 1


@functools.lru_cache(maxsize=None)
def _expected(path_str: str) -> SimpleNamespace:
    """Derive the expected ROI geometry and acquisition layout from a file's metadata."""
    metadata = read_scanimage_metadata(path_str)
    frame_data = metadata["scan_image_non_varying_frame_metadata"]
    parsed = metadata["roiextractors_parsed_metadata"]
    roi_group_metadata = metadata["scan_image_roi_group_metadata"]

    rois = ScanImageMultiROIImagingExtractor._get_roi_list(roi_group_metadata)
    scanfields = ScanImageMultiROIImagingExtractor._get_imaging_scanfields(roi_group_metadata)
    num_pages, (page_height, page_width), dtype = _page_info(path_str)

    return SimpleNamespace(
        num_rois=len(rois),
        names=[roi.get("name") for roi in rois],
        uuids=[roi.get("roiUuid") for roi in rois],
        heights=[scanfield["pixelResolutionXY"][1] for scanfield in scanfields],
        widths=[scanfield["pixelResolutionXY"][0] for scanfield in scanfields],
        page_height=page_height,
        page_width=page_width,
        num_pages=num_pages,
        dtype=dtype,
        num_channels=parsed["num_channels"],
        channel_names=parsed["channel_names"],
        sampling_frequency=parsed["sampling_frequency"],
        line_period=frame_data.get("SI.hRoiManager.linePeriod"),
    )


def _make_extractor(path: Path, **kwargs) -> ScanImageMultiROIImagingExtractor:
    """Construct an extractor, supplying a channel name automatically for multi-channel files."""
    expected = _expected(str(path))
    if expected.num_channels > 1 and "channel_name" not in kwargs:
        kwargs["channel_name"] = expected.channel_names[0]
    return ScanImageMultiROIImagingExtractor(file_path=path, **kwargs)


# --- Fixture discovery and parametrization -----------------------------------------------------

PLANAR_MROI_FILES = [
    path
    for path in sorted(FIXTURES_DIR.glob("*.tif")) + sorted(FIXTURES_DIR.glob("*.tiff"))
    if _is_planar_mroi(str(path))
]

pytestmark = pytest.mark.skipif(not PLANAR_MROI_FILES, reason=f"No planar multi-ROI fixture found in {FIXTURES_DIR}")

FILE_PARAMS = [pytest.param(path, id=path.name) for path in PLANAR_MROI_FILES]
FILE_ROI_PARAMS = [
    pytest.param(path, roi_index, id=f"{path.name}-roi{roi_index}")
    for path in PLANAR_MROI_FILES
    for roi_index in range(_expected(str(path)).num_rois)
]


def _names_by_occurrence(file_path: Path) -> tuple[list, list]:
    """Split a file's named ROIs into (unique names, duplicated names), ignoring unnamed ROIs.

    ROI names are not guaranteed to be unique, so each file may have any mix of unique and
    duplicated names (or none of either). Tests use this to decide, per file, which selection
    behavior to assert rather than assuming a file has duplicates.
    """
    names = _expected(str(file_path)).names
    counts = Counter(name for name in names if name is not None)
    unique = [name for name in dict.fromkeys(names) if name is not None and counts[name] == 1]
    duplicated = [name for name in dict.fromkeys(names) if name is not None and counts[name] > 1]
    return unique, duplicated


# --- Static inspectors -------------------------------------------------------------------------


@pytest.mark.parametrize("file_path", FILE_PARAMS)
def test_static_inspectors_match_metadata(file_path):
    """The static inspectors report the enabled ROIs, in order, as read straight from the file."""
    expected = _expected(str(file_path))
    assert ScanImageMultiROIImagingExtractor.get_num_rois(file_path) == expected.num_rois
    assert ScanImageMultiROIImagingExtractor.get_roi_names(file_path) == expected.names
    assert ScanImageMultiROIImagingExtractor.get_roi_uuids(file_path) == expected.uuids


# --- Basic properties --------------------------------------------------------------------------


@pytest.mark.parametrize("file_path", FILE_PARAMS)
def test_basic_properties(file_path):
    """A planar mROI file yields a planar extractor with one sample per acquisition cycle."""
    expected = _expected(str(file_path))
    extractor = _make_extractor(file_path)
    assert extractor.is_volumetric is False
    assert extractor.get_dtype() == expected.dtype
    assert extractor.get_sampling_frequency() == pytest.approx(expected.sampling_frequency)
    # One sample per cycle: pages divided over the channels stored in the file.
    assert extractor.get_num_samples() == expected.num_pages // expected.num_channels


# --- ROI geometry invariants -------------------------------------------------------------------


@pytest.mark.parametrize("file_path", FILE_PARAMS)
def test_roi_row_spans_partition_the_page(file_path):
    """The ROI row spans start at the top, have equal heights' gaps, and reach the page bottom."""
    expected = _expected(str(file_path))
    extractors = [_make_extractor(file_path, roi_index=index) for index in range(expected.num_rois)]
    starts = [extractor._roi_row_start for extractor in extractors]
    ends = [extractor._roi_row_end for extractor in extractors]

    # Each span is exactly the ROI's metadata height, and the image shape follows the ROI.
    for index, extractor in enumerate(extractors):
        assert ends[index] - starts[index] == expected.heights[index]
        assert extractor.get_image_shape() == (expected.heights[index], expected.widths[index])
        # ROIs are stacked across the full page width.
        assert extractor.get_image_shape()[1] == expected.page_width

    # The first field is at the top and the last reaches the bottom of the page.
    assert starts[0] == 0
    assert ends[-1] == expected.page_height
    # The inter-ROI gaps are all identical and non-negative.
    gaps = {starts[index + 1] - ends[index] for index in range(expected.num_rois - 1)}
    assert len(gaps) <= 1
    assert all(gap >= 0 for gap in gaps)


@pytest.mark.parametrize("file_path", FILE_PARAMS)
def test_gap_inference_methods_agree(file_path):
    """Geometric (default) and timing gap inference agree, so construction does not raise."""
    geometric = _make_extractor(file_path, gap_inference="geometric")._inter_roi_gap
    timing = _make_extractor(file_path, gap_inference="timing")._inter_roi_gap
    assert geometric == timing


# --- Cropping correctness ----------------------------------------------------------------------


@pytest.mark.parametrize("file_path,roi_index", FILE_ROI_PARAMS)
def test_series_crops_exactly_the_claimed_rows(file_path, roi_index):
    """get_series returns precisely the page rows the extractor claims for the ROI."""
    extractor = _make_extractor(file_path, roi_index=roi_index)
    raw_stack = _raw_stack(str(file_path))
    expected = raw_stack[:, extractor._roi_row_start : extractor._roi_row_end, :]
    series = extractor.get_series()
    assert series.shape == (extractor.get_num_samples(), *extractor.get_image_shape())
    assert_array_equal(series, expected)


@pytest.mark.parametrize("file_path", FILE_PARAMS)
def test_adjacent_rois_read_different_data(file_path):
    """Distinct ROIs expose distinct image content (the crop is not stuck on one band)."""
    expected = _expected(str(file_path))
    if expected.num_rois < 2:
        pytest.skip("file has a single ROI")
    first = _make_extractor(file_path, roi_index=0).get_series(start_sample=0, end_sample=1)
    second = _make_extractor(file_path, roi_index=1).get_series(start_sample=0, end_sample=1)
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("file_path", FILE_PARAMS)
def test_sample_subrange_matches_full_series(file_path):
    """A sub-range read equals the matching slice of the full cropped series."""
    extractor = _make_extractor(file_path, roi_index=0)
    full = extractor.get_series()
    if extractor.get_num_samples() < 2:
        pytest.skip("file has a single sample")
    assert_array_equal(extractor.get_series(start_sample=1, end_sample=2), full[1:2])


# --- Selector resolution -----------------------------------------------------------------------


@pytest.mark.parametrize("file_path,roi_index", FILE_ROI_PARAMS)
def test_select_by_uuid_is_case_insensitive(file_path, roi_index):
    """Selecting a ROI by its UUID (any case) resolves to the same ROI as the index."""
    expected = _expected(str(file_path))
    uuid = expected.uuids[roi_index]
    if uuid is None or expected.uuids.count(uuid) > 1:
        pytest.skip("ROI has no unique UUID")
    extractor = _make_extractor(file_path, roi_uuid=uuid.lower())
    assert extractor.roi_index == roi_index


@pytest.mark.parametrize("file_path", FILE_PARAMS)
def test_unique_names_select_their_roi(file_path):
    """Every ROI name that occurs exactly once resolves to that ROI's position."""
    names = _expected(str(file_path)).names
    unique_names, _ = _names_by_occurrence(file_path)
    if not unique_names:
        pytest.skip("file has no uniquely-named ROIs")
    for name in unique_names:
        extractor = _make_extractor(file_path, roi_name=name)
        assert extractor.roi_index == names.index(name)
        assert extractor.roi_name == name


@pytest.mark.parametrize("file_path", FILE_PARAMS)
def test_selectors_resolve_to_the_same_data(file_path):
    """Index, UUID, and (when unique) name selectors for one ROI yield identical pixel data."""
    expected = _expected(str(file_path))
    by_index = _make_extractor(file_path, roi_index=0)
    reference = by_index.get_series()
    if expected.uuids[0] is not None and expected.uuids.count(expected.uuids[0]) == 1:
        assert_array_equal(_make_extractor(file_path, roi_uuid=expected.uuids[0]).get_series(), reference)
    if expected.names[0] is not None and expected.names.count(expected.names[0]) == 1:
        assert_array_equal(_make_extractor(file_path, roi_name=expected.names[0]).get_series(), reference)


# --- Selector error paths ----------------------------------------------------------------------


@pytest.mark.parametrize("file_path", FILE_PARAMS)
def test_duplicate_names_are_rejected_as_selectors(file_path):
    """If (and only if) a file has ROI names shared by several ROIs, selecting by such a name raises.

    ROI names need not be unique. This checks each file for ambiguous names and, when present,
    requires that using one as ``roi_name`` raises rather than silently picking one of the matches.
    Files without duplicate names exercise nothing here and are skipped.
    """
    _, duplicate_names = _names_by_occurrence(file_path)
    if not duplicate_names:
        pytest.skip("file has no duplicate ROI names")
    for name in duplicate_names:
        with pytest.raises(ValueError, match="ambiguous"):
            _make_extractor(file_path, roi_name=name)


@pytest.mark.parametrize("file_path", FILE_PARAMS)
def test_multiple_selectors_are_rejected(file_path):
    """Passing more than one of index/name/uuid is rejected as mutually exclusive."""
    name = _expected(str(file_path)).names[0]
    with pytest.raises(ValueError, match="at most one"):
        _make_extractor(file_path, roi_index=0, roi_name=name)


@pytest.mark.parametrize("file_path", FILE_PARAMS)
def test_out_of_range_index_is_rejected(file_path):
    """A roi_index beyond the available ROIs is rejected."""
    num_rois = _expected(str(file_path)).num_rois
    with pytest.raises(ValueError, match="roi_index"):
        _make_extractor(file_path, roi_index=num_rois)


@pytest.mark.parametrize("file_path", FILE_PARAMS)
def test_unknown_name_and_uuid_are_rejected(file_path):
    """Selectors that match no ROI are rejected."""
    with pytest.raises(ValueError, match="No ROI named"):
        _make_extractor(file_path, roi_name="__no_such_roi__")
    with pytest.raises(ValueError, match="No ROI with roiUuid"):
        _make_extractor(file_path, roi_uuid="__no_such_uuid__")


# --- Per-field timing --------------------------------------------------------------------------


@pytest.mark.parametrize("file_path,roi_index", FILE_ROI_PARAMS)
def test_field_time_offset_tracks_start_row(file_path, roi_index):
    """The within-frame field offset is the per-row dwell time times the field's start row."""
    expected = _expected(str(file_path))
    extractor = _make_extractor(file_path, roi_index=roi_index)
    if expected.line_period is None:
        with pytest.raises(ValueError, match="line_period"):
            extractor.get_field_time_offset()
        return
    assert extractor.get_field_time_offset() == pytest.approx(expected.line_period * extractor.roi_row_start)
    if extractor.roi_row_start == 0:
        assert extractor.get_field_time_offset() == 0.0


@pytest.mark.parametrize("file_path", FILE_PARAMS)
def test_field_time_offset_increases_down_the_page(file_path):
    """Fields scanned lower in the page have strictly larger within-frame offsets."""
    expected = _expected(str(file_path))
    if expected.line_period is None or expected.num_rois < 2:
        pytest.skip("no line period, or a single ROI")
    offsets = [
        _make_extractor(file_path, roi_index=index).get_field_time_offset() for index in range(expected.num_rois)
    ]
    assert offsets == sorted(offsets)
    assert offsets[0] < offsets[-1]


@pytest.mark.parametrize("file_path", FILE_PARAMS)
def test_timestamps_correction_adds_the_field_offset(file_path):
    """correct_field_offset shifts every frame timestamp by the constant per-field offset."""
    expected = _expected(str(file_path))
    if expected.line_period is None:
        pytest.skip("no line period")
    # Use the last ROI so the offset is non-zero.
    extractor = _make_extractor(file_path, roi_index=expected.num_rois - 1)
    base = extractor.get_timestamps()
    assert base.shape == (extractor.get_num_samples(),)
    assert_array_equal(extractor.get_timestamps(correct_field_offset=True), base + extractor.get_field_time_offset())
    assert_array_equal(extractor.get_timestamps(correct_field_offset=False), base)


# --- ROI metadata exposure ---------------------------------------------------------------------


@pytest.mark.parametrize("file_path,roi_index", FILE_ROI_PARAMS)
def test_roi_identity_is_exposed(file_path, roi_index):
    """The selected ROI's name, UUID, and index are exposed and match the metadata order."""
    expected = _expected(str(file_path))
    extractor = _make_extractor(file_path, roi_index=roi_index)
    assert extractor.roi_index == roi_index
    assert extractor.roi_name == expected.names[roi_index]
    assert extractor.roi_uuid == expected.uuids[roi_index]
