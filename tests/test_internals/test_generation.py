"""Tests for noise generator imaging extractors."""

import sys

import numpy as np
import pytest

from roiextractors import GaussianNoiseImagingExtractor, PoissonNoiseImagingExtractor

# ---------------------------------------------------------------------------
# Shared tests parametrized over both concrete classes
# ---------------------------------------------------------------------------

generator_classes = [PoissonNoiseImagingExtractor, GaussianNoiseImagingExtractor]


@pytest.fixture(params=generator_classes)
def extractor_cls(request):
    return request.param


class TestNoiseImagingExtractors:

    def test_basic_properties(self, extractor_cls):
        num_samples = 500
        num_rows = 50
        num_columns = 60
        sampling_frequency = 25.0
        expected_dtypes = {
            PoissonNoiseImagingExtractor: np.dtype("int64"),
            GaussianNoiseImagingExtractor: np.dtype("float32"),
        }

        ext = extractor_cls(
            num_samples=num_samples,
            num_rows=num_rows,
            num_columns=num_columns,
            sampling_frequency=sampling_frequency,
        )
        assert ext.get_image_shape() == (num_rows, num_columns)
        assert ext.get_num_samples() == num_samples
        assert ext.get_sampling_frequency() == sampling_frequency
        assert ext.get_dtype() == expected_dtypes[extractor_cls]

        data = ext.get_series(0, 10)
        assert data.dtype == expected_dtypes[extractor_cls]

    def test_reproducibility(self, extractor_cls):
        """Two extractors with the same seed must return identical data, just
        like reading the same file twice would."""
        kwargs = dict(num_samples=200, num_rows=20, num_columns=20, seed=42)
        ext1 = extractor_cls(**kwargs)
        ext2 = extractor_cls(**kwargs)
        np.testing.assert_array_equal(ext1.get_series(0, 50), ext2.get_series(0, 50))

    def test_contiguous_reads_equal_single_read(self, extractor_cls):
        """Reading [0, 200) must equal concatenating [0, 100) and [100, 200),
        just like reading contiguous chunks from a real file would."""
        ext = extractor_cls(num_samples=300, num_rows=10, num_columns=10, seed=42)
        full = ext.get_series(0, 200)
        part1 = ext.get_series(0, 100)
        part2 = ext.get_series(100, 200)
        combined = np.concatenate([part1, part2], axis=0)
        np.testing.assert_array_equal(full, combined)

    def test_sub_range_matches_slice(self, extractor_cls):
        """A sub-range must match the corresponding slice of a larger read,
        just like indexing into real array data would."""
        ext = extractor_cls(num_samples=300, num_rows=10, num_columns=10, seed=42)
        full_start = ext.get_series(0, 50)
        partial = ext.get_series(10, 40)
        np.testing.assert_array_equal(full_start[10:40], partial)

    def test_different_seeds_produce_different_data(self, extractor_cls):
        ext1 = extractor_cls(num_samples=100, num_rows=10, num_columns=10, seed=0)
        ext2 = extractor_cls(num_samples=100, num_rows=10, num_columns=10, seed=1)
        assert not np.array_equal(ext1.get_series(0, 50), ext2.get_series(0, 50))

    def test_native_timestamps_returns_none(self, extractor_cls):
        ext = extractor_cls(num_samples=100, num_rows=10, num_columns=10)
        assert ext.get_native_timestamps() is None


# ---------------------------------------------------------------------------
# Poisson-specific tests
# ---------------------------------------------------------------------------


class TestPoissonNoiseImagingExtractor:
    def test_non_negative_integer_values(self):
        """Poisson data represents photon counts: values must be non-negative integers."""
        ext = PoissonNoiseImagingExtractor(num_samples=100, num_rows=20, num_columns=20)
        data = ext.get_series(0, 100)
        assert np.all(data >= 0)
        np.testing.assert_array_equal(data, np.floor(data))

    def test_mean_approximately_baseline(self):
        """Verify that the baseline parameter is correctly propagated to the Poisson
        lambda, so the sample mean is close to the requested baseline."""
        baseline = 100.0
        ext = PoissonNoiseImagingExtractor(num_samples=1000, num_rows=50, num_columns=50, baseline=baseline)
        data = ext.get_series(0, 1000)
        np.testing.assert_allclose(data.mean(), baseline, rtol=0.05)


# ---------------------------------------------------------------------------
# Gaussian-specific tests
# ---------------------------------------------------------------------------


class TestGaussianNoiseImagingExtractor:
    def test_mean_and_std_match_parameters(self):
        """Verify that noise_mean and noise_std are correctly propagated to the
        Gaussian generator, so the sample statistics match the requested values."""
        ext = GaussianNoiseImagingExtractor(
            num_samples=1000, num_rows=50, num_columns=50, noise_mean=5.0, noise_std=2.0
        )
        data = ext.get_series(0, 1000)
        np.testing.assert_allclose(data.mean(), 5.0, atol=0.1)
        np.testing.assert_allclose(data.std(), 2.0, atol=0.1)


# ---------------------------------------------------------------------------
# Volumetric tests
# ---------------------------------------------------------------------------


class TestVolumetricNoiseGenerator:
    @pytest.fixture(params=generator_classes)
    def volumetric_ext(self, request):
        return request.param(num_samples=200, num_rows=30, num_columns=40, num_planes=5)

    def test_basic_properties(self, volumetric_ext):
        num_rows = 30
        num_columns = 40
        num_planes = 5

        assert volumetric_ext.is_volumetric
        assert volumetric_ext.get_image_shape() == (num_rows, num_columns)
        assert volumetric_ext.get_num_planes() == num_planes
        assert volumetric_ext.get_volume_shape() == (num_rows, num_columns, num_planes)
        assert volumetric_ext.get_sample_shape() == (num_rows, num_columns, num_planes)

        data = volumetric_ext.get_series(0, 10)
        assert data.shape == (10, num_rows, num_columns, num_planes)

    def test_non_volumetric_raises_get_num_planes(self):
        """A non-volumetric extractor must raise NotImplementedError when
        get_num_planes is called, matching the base class contract."""
        ext = PoissonNoiseImagingExtractor(num_samples=10, num_rows=10, num_columns=10)
        assert not ext.is_volumetric
        with pytest.raises(NotImplementedError):
            ext.get_num_planes()

    def test_reproducibility(self, volumetric_ext):
        """Repeated calls to get_series with the same range must return
        identical data, just like reading the same file twice would."""
        data1 = volumetric_ext.get_series(0, 50)
        data2 = volumetric_ext.get_series(0, 50)
        np.testing.assert_array_equal(data1, data2)

    def test_contiguous_reads(self, volumetric_ext):
        """Concatenating two contiguous reads must equal a single read over
        the full range, just like reading contiguous chunks from a real file."""
        full = volumetric_ext.get_series(0, 150)
        part1 = volumetric_ext.get_series(0, 100)
        part2 = volumetric_ext.get_series(100, 150)
        np.testing.assert_array_equal(full, np.concatenate([part1, part2], axis=0))


# ---------------------------------------------------------------------------
# Memory tests (pytest-memray)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="memray is not supported on Windows")
class TestMemoryUsage:
    """Memory regression tests using pytest-memray.

    These verify that get_series allocations stay bounded: the result array
    plus a small overhead, but no extra copies or RNG allocations at read time.

    Skipped on Windows because memray only supports Linux and macOS.
    """

    @pytest.mark.limit_memory("420 MB")
    def test_poisson_get_series_memory(self):
        """256x256 int64: tile ~100 MB (200 samples) + result ~250 MB (500 samples) = ~350 MB.
        An accidental extra copy of the result would push to ~600 MB, well over the limit."""
        ext = PoissonNoiseImagingExtractor(num_samples=5000, num_rows=256, num_columns=256)
        ext.get_series(0, 500)

    @pytest.mark.limit_memory("270 MB")
    def test_gaussian_get_series_memory(self):
        """256x256 float32: tile ~100 MB (400 samples) + result ~125 MB (500 samples) = ~225 MB.
        An accidental extra copy of the result would push to ~350 MB, well over the limit."""
        ext = GaussianNoiseImagingExtractor(num_samples=5000, num_rows=256, num_columns=256)
        ext.get_series(0, 500)
