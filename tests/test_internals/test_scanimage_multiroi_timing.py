"""Unit tests for ScanImage multi-ROI per-field timing.

These tests exercise the per-field timing accessors (``get_field_time_offset`` and the
``correct_field_offset`` option on ``get_timestamps``) with synthetic state, so they run in the
minimal (no-data) test job and do not depend on any particular recording. There is currently no
planar multi-ROI fixture in the GIN data, so the construction happy path cannot be exercised end to
end; instances are built with ``object.__new__`` and the relevant attributes are set directly.
"""

import numpy as np
import pytest

from roiextractors import ScanImageMultiROIImagingExtractor
from roiextractors.imagingextractor import ImagingExtractor


def _make_extractor(line_period, roi_row_start) -> ScanImageMultiROIImagingExtractor:
    """Build a bare extractor with only the timing attributes set, bypassing __init__."""
    extractor = object.__new__(ScanImageMultiROIImagingExtractor)
    extractor.line_period = line_period
    extractor.roi_row_start = roi_row_start
    return extractor


def test_field_time_offset_is_line_period_times_row_start():
    """The offset is the per-row dwell time times the field's start row in the stacked page."""
    extractor = _make_extractor(line_period=2.0, roi_row_start=100)
    assert extractor.get_field_time_offset() == 200.0


def test_field_time_offset_is_zero_for_top_field():
    """The topmost field starts at row 0, so it is acquired at the frame start (offset 0)."""
    extractor = _make_extractor(line_period=2.0, roi_row_start=0)
    assert extractor.get_field_time_offset() == 0.0


def test_field_time_offset_raises_without_line_period():
    """When line_period is unavailable the offset cannot be computed and raises loudly."""
    extractor = _make_extractor(line_period=None, roi_row_start=100)
    with pytest.raises(ValueError, match="line_period"):
        extractor.get_field_time_offset()


def test_get_timestamps_returns_frame_timestamps_by_default(monkeypatch):
    """By default get_timestamps returns the resolved frame timestamps, unshifted."""
    extractor = _make_extractor(line_period=2.0, roi_row_start=100)
    frame_timestamps = np.arange(5, dtype=float)
    # Stub the inherited resolved timeline so this test needs no file access.
    monkeypatch.setattr(
        ImagingExtractor,
        "get_timestamps",
        lambda self, start_sample=None, end_sample=None: frame_timestamps,
    )
    np.testing.assert_array_equal(extractor.get_timestamps(), frame_timestamps)


def test_get_timestamps_applies_offset_when_requested(monkeypatch):
    """With correct_field_offset=True the field offset is added to the frame timestamps."""
    extractor = _make_extractor(line_period=2.0, roi_row_start=100)
    frame_timestamps = np.arange(5, dtype=float)
    monkeypatch.setattr(
        ImagingExtractor,
        "get_timestamps",
        lambda self, start_sample=None, end_sample=None: frame_timestamps,
    )
    np.testing.assert_array_equal(
        extractor.get_timestamps(correct_field_offset=True),
        frame_timestamps + 200.0,
    )


def test_get_timestamps_raises_when_offset_uncomputable(monkeypatch):
    """Requesting the correction when the offset is uncomputable (no line_period) fails loudly."""
    extractor = _make_extractor(line_period=None, roi_row_start=100)
    frame_timestamps = np.arange(5, dtype=float)
    monkeypatch.setattr(
        ImagingExtractor,
        "get_timestamps",
        lambda self, start_sample=None, end_sample=None: frame_timestamps,
    )
    # The uncorrected path still works; only the explicit correction request raises.
    np.testing.assert_array_equal(extractor.get_timestamps(), frame_timestamps)
    with pytest.raises(ValueError, match="line_period"):
        extractor.get_timestamps(correct_field_offset=True)


class _FakePage:
    """Minimal stand-in for a tifffile page exposing the stacked array via ``asarray()``."""

    def __init__(self, array: np.ndarray):
        self._array = array

    def asarray(self) -> np.ndarray:
        return self._array


class _FakeReader:
    """Minimal stand-in for a tifffile reader exposing its pages as a list."""

    def __init__(self, pages: list):
        self.pages = pages

    def close(self) -> None:
        """No-op close so the extractor's __del__ handle cleanup succeeds."""


def test_get_series_crops_the_rois_rows():
    """get_series slices the configured field-of-view window down to the ROI's rows.

    The multi-ROI extractor sets ``_row_slice``/``_column_slice`` to the ROI window in ``__init__``
    and lets the base class crop each page as it is read; this checks that wiring with a fake page.
    """
    extractor = object.__new__(ScanImageMultiROIImagingExtractor)
    # State that __init__ would set for an ROI spanning page rows [100, 150) over a full-width page.
    extractor._roi_row_start = 100
    extractor._roi_row_end = 150
    extractor._row_slice = slice(100, 150)
    extractor._column_slice = slice(0, 8)
    extractor._num_rows = 50
    extractor._num_columns = 8
    extractor._num_planes = 1
    extractor._num_samples = 1
    extractor._dtype = np.dtype("float64")
    extractor.is_volumetric = False
    # A 0..209 row-indexed page so the cropped rows are self-identifying.
    page_array = np.arange(210)[:, np.newaxis] * np.ones((210, 8))
    extractor._tiff_readers = [_FakeReader([_FakePage(page_array)])]
    extractor._frames_to_ifd_table = np.array([(0, 0)], dtype=[("file_index", int), ("IFD_index", int)])

    series = extractor.get_series()

    assert series.shape == (1, 50, 8)
    # The cropped rows are exactly page rows 100..149.
    np.testing.assert_array_equal(series[0, :, 0], np.arange(100, 150))
