"""Unit tests for ScanImage multi-ROI inter-ROI gap inference.

These tests exercise the pure gap-inference logic with small synthetic values, so they run in the
minimal (no-data) test job and do not depend on any particular recording.
"""

import warnings

import pytest

from roiextractors import ScanImageMultiROIImagingExtractor

# The gap inference lives as static methods on the multi-ROI extractor; alias them for brevity.
_even_ceil = ScanImageMultiROIImagingExtractor._even_ceil
infer_inter_roi_gap = ScanImageMultiROIImagingExtractor._infer_inter_roi_gap


@pytest.mark.parametrize(
    "value, expected",
    [
        (4.0, 4),  # already even -> unchanged
        (5.0, 6),  # odd integer -> next even
        (4.1, 6),  # non-integer -> rounds up to the next even integer
        (5.9, 6),
        (0.1, 2),  # rounds up to the smallest positive even integer
    ],
)
def test_even_ceil(value, expected):
    """`_even_ceil` rounds up to the nearest even integer (resonant scanning needs even line counts)."""
    assert _even_ceil(value) == expected


def test_geometric_inference():
    """Geometric inference spreads the leftover page rows over the inter-ROI boundaries."""
    # 4 ROIs of height 100 (sum 400) on a 430-row page -> 30 leftover over 3 gaps -> 10.
    assert infer_inter_roi_gap(page_height=430, roi_heights=[100] * 4) == 10


def test_geometric_handles_unequal_heights():
    """Geometric inference sums the actual ROI heights, so unequal heights are handled."""
    # heights 100, 200, 300 (sum 600) on a 700-row page -> 100 leftover over 2 gaps -> 50.
    assert infer_inter_roi_gap(page_height=700, roi_heights=[100, 200, 300]) == 50


def test_timing_inference_rounds_up_to_even():
    """Timing inference rounds the fly-to line count up to an even number, not down."""
    # raw fly-to lines = flyto_time / line_period = 7.2 / 1.0 = 7.2; naive int gives 7, even-ceil gives 8.
    gap = infer_inter_roi_gap(
        page_height=208,
        roi_heights=[100, 100],
        line_period=1.0,
        flyto_time=7.2,
        method="timing",
    )
    assert gap == 8


def test_methods_agree_no_warning():
    """When both methods agree, no mismatch warning is emitted."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gap = infer_inter_roi_gap(
            page_height=208,
            roi_heights=[100, 100],
            line_period=1.0,
            flyto_time=7.2,
            method="geometric",
        )
    assert gap == 8


def test_single_roi_has_no_gap():
    """A single ROI has no inter-ROI boundary, so the gap is zero."""
    assert infer_inter_roi_gap(page_height=100, roi_heights=[100]) == 0


def test_mismatch_raises_by_default():
    """When the two methods disagree, a ValueError is raised by default."""
    # geometric: 220 - 200 = 20 leftover over 1 gap -> 20; timing: even-ceil(7.2) -> 8.
    with pytest.raises(ValueError, match="disagree"):
        infer_inter_roi_gap(
            page_height=220,
            roi_heights=[100, 100],
            line_period=1.0,
            flyto_time=7.2,
        )


def test_mismatch_warns_and_respects_method_when_requested():
    """With on_mismatch='warn', a warning is emitted and the requested method's value is returned."""
    with pytest.warns(UserWarning, match="disagree"):
        geometric_gap = infer_inter_roi_gap(
            page_height=220,
            roi_heights=[100, 100],
            line_period=1.0,
            flyto_time=7.2,
            method="geometric",
            on_mismatch="warn",
        )
    assert geometric_gap == 20

    with pytest.warns(UserWarning, match="disagree"):
        timing_gap = infer_inter_roi_gap(
            page_height=220,
            roi_heights=[100, 100],
            line_period=1.0,
            flyto_time=7.2,
            method="timing",
            on_mismatch="warn",
        )
    assert timing_gap == 8


def test_falls_back_to_timing_when_geometric_not_divisible():
    """If the geometric residual is not evenly divisible, fall back to the timing estimate."""
    # 3 ROIs -> 2 gaps; residual 21 is not divisible by 2, so geometric is unavailable.
    gap = infer_inter_roi_gap(
        page_height=321,
        roi_heights=[100, 100, 100],
        line_period=1.0,
        flyto_time=7.2,
        method="geometric",
    )
    assert gap == 8


def test_raises_when_unresolvable():
    """If neither method can be computed, a ValueError is raised."""
    # Non-divisible geometric residual and no timing inputs.
    with pytest.raises(ValueError):
        infer_inter_roi_gap(page_height=321, roi_heights=[100, 100, 100])
