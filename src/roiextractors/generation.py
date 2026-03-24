"""Lazy noise generators for imaging data.

Both generators use a "tile pregenerated" strategy: a single block of random
noise is generated once during construction and then tiled (repeated) across
the time axis in ``get_series``. This makes ``get_series`` a fast array copy
(no per-call RNG work) while keeping memory usage bounded to a
single tile. The tile size is chosen automatically to stay under ~100 MB.

::

    __init__ generates a small tile of T noise samples (kept under ~100 MB):

        sample 0     sample 1           sample T-1
        +-------+    +-------+          +-------+
        | noise |    | noise |   ...    | noise |
        | image |    | image |          | image |
        +-------+    +-------+          +-------+
        |<------------- tile ------------------>|

    get_series views the tile as repeating along the time axis and
    copies out the requested [start, end) range:

        |--- tile ---|--- tile ---|--- tile ---|--- tile ---|
                          ^start        ^end
                          |-- result ---|

The extractors behave like any other ``ImagingExtractor``: calling
``get_series()`` with the same arguments always returns the same data, and
concatenating ``get_series(0, k)`` with ``get_series(k, n)`` is identical
to calling ``get_series(0, n)``.
"""

from __future__ import annotations

import numpy as np

from .imagingextractor import ImagingExtractor


class PoissonNoiseImagingExtractor(ImagingExtractor):
    """Lazy imaging extractor that generates Poisson noise on-the-fly.

    Physically correct for microscopy (photon counting). Returns int64 data
    (the native dtype of ``numpy.random.Generator.poisson``).

    A single tile of noise is generated at construction time and tiled across
    the time axis during ``get_series``, so reads are fast copy operations.

    Parameters
    ----------
    num_samples : int, default: 1000
        Number of samples in the video.
    num_rows : int, default: 100
        Number of rows in each sample.
    num_columns : int, default: 100
        Number of columns in each sample.
    num_planes : int or None, default: None
        Number of depth planes. When not None the extractor is volumetric and
        ``get_series`` returns arrays with shape ``(samples, rows, cols, planes)``.
    sampling_frequency : float, default: 30.0
        Sampling frequency in Hz.
    seed : int or None, default: None
        Random seed for reproducibility. If None, a random seed is generated.
    baseline : float, default: 100.0
        Mean photon count per pixel (lambda parameter of the Poisson distribution).
    """

    extractor_name = "PoissonNoiseImagingExtractor"

    def __init__(
        self,
        *,
        num_samples: int = 1000,
        num_rows: int = 100,
        num_columns: int = 100,
        num_planes: int | None = None,
        sampling_frequency: float = 30.0,
        seed: int | None = None,
        baseline: float = 100.0,
    ):
        super().__init__()

        self._num_rows = num_rows
        self._num_columns = num_columns
        self._num_samples = num_samples
        self._sampling_frequency = float(sampling_frequency)
        self._seed = int(seed) if seed is not None else int(np.random.default_rng().integers(0, 2**31))
        self._baseline = baseline

        if num_planes is not None and num_planes > 1:
            self.is_volumetric = True
            self._num_planes = num_planes

        spatial = (self._num_rows, self._num_columns)
        if self.is_volumetric:
            spatial = (*spatial, self._num_planes)

        # Pregenerate a noise tile of ~100 MB (at least 1 sample, at most num_samples).
        # The tile is tiled across time in get_series, no RNG work at read time.
        bytes_per_sample = np.prod(spatial) * np.dtype(np.int64).itemsize
        tile_samples = min(max(1, (100 * 1024 * 1024) // bytes_per_sample), self._num_samples)
        rng = np.random.default_rng(seed=self._seed)
        self._tile = rng.poisson(lam=self._baseline, size=(tile_samples, *spatial))

    def get_image_shape(self) -> tuple[int, int]:
        return (self._num_rows, self._num_columns)

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_dtype(self) -> np.dtype:
        return np.dtype(np.int64)

    def get_num_planes(self) -> int:
        if not self.is_volumetric:
            raise NotImplementedError(
                "This extractor is not volumetric. "
                "The get_num_planes method is only available for volumetric extractors."
            )
        return self._num_planes

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:  # noqa: ARG002
        return None

    def get_series(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
        """Return Poisson noise for the requested sample range.

        Data is read from a pregenerated tile that is conceptually repeated
        along the time axis. No RNG work at read time.
        """
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self._num_samples

        num_requested = end_sample - start_sample
        tile_len = self._tile.shape[0]
        result = np.empty((num_requested, *self._tile.shape[1:]), dtype=np.int64)

        output_pos = 0
        while output_pos < num_requested:
            tile_offset = (start_sample + output_pos) % tile_len
            available = tile_len - tile_offset
            n = min(available, num_requested - output_pos)
            result[output_pos : output_pos + n] = self._tile[tile_offset : tile_offset + n]
            output_pos += n

        return result


class GaussianNoiseImagingExtractor(ImagingExtractor):
    """Lazy imaging extractor that generates Gaussian noise on-the-fly.

    Returns float32 data. A single tile of noise is generated at construction
    time and tiled across the time axis during ``get_series``, so reads are
    fast memcopy operations.

    Parameters
    ----------
    num_samples : int, default: 1000
        Number of samples in the video.
    num_rows : int, default: 100
        Number of rows in each sample.
    num_columns : int, default: 100
        Number of columns in each sample.
    num_planes : int or None, default: None
        Number of depth planes. When not None the extractor is volumetric and
        ``get_series`` returns arrays with shape ``(samples, rows, cols, planes)``.
    sampling_frequency : float, default: 30.0
        Sampling frequency in Hz.
    seed : int or None, default: None
        Random seed for reproducibility. If None, a random seed is generated.
    noise_mean : float, default: 0.0
        Mean of the Gaussian distribution.
    noise_std : float, default: 1.0
        Standard deviation of the Gaussian distribution.
    """

    extractor_name = "GaussianNoiseImagingExtractor"

    def __init__(
        self,
        *,
        num_samples: int = 1000,
        num_rows: int = 100,
        num_columns: int = 100,
        num_planes: int | None = None,
        sampling_frequency: float = 30.0,
        seed: int | None = None,
        noise_mean: float = 0.0,
        noise_std: float = 1.0,
    ):
        super().__init__()

        self._num_rows = num_rows
        self._num_columns = num_columns
        self._num_samples = num_samples
        self._sampling_frequency = float(sampling_frequency)
        self._seed = int(seed) if seed is not None else int(np.random.default_rng().integers(0, 2**31))
        self._noise_mean = noise_mean
        self._noise_std = noise_std

        if num_planes is not None and num_planes > 1:
            self.is_volumetric = True
            self._num_planes = num_planes

        spatial = (self._num_rows, self._num_columns)
        if self.is_volumetric:
            spatial = (*spatial, self._num_planes)

        # Pregenerate a noise tile of ~100 MB (at least 1 sample, at most num_samples).
        # The tile is tiled across time in get_series, no RNG work at read time.
        bytes_per_sample = np.prod(spatial) * np.dtype(np.float32).itemsize
        tile_samples = min(max(1, (100 * 1024 * 1024) // bytes_per_sample), self._num_samples)
        rng = np.random.default_rng(seed=self._seed)
        self._tile = rng.standard_normal(size=(tile_samples, *spatial), dtype=np.float32)
        if self._noise_std != 1.0:
            self._tile *= self._noise_std
        if self._noise_mean != 0.0:
            self._tile += self._noise_mean

    def get_image_shape(self) -> tuple[int, int]:
        return (self._num_rows, self._num_columns)

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_dtype(self) -> np.dtype:
        return np.dtype(np.float32)

    def get_num_planes(self) -> int:
        if not self.is_volumetric:
            raise NotImplementedError(
                "This extractor is not volumetric. "
                "The get_num_planes method is only available for volumetric extractors."
            )
        return self._num_planes

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:  # noqa: ARG002
        return None

    def get_series(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
        """Return Gaussian noise for the requested sample range.

        Data is read from a pregenerated tile that is conceptually repeated
        along the time axis. No RNG work at read time.
        """
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self._num_samples

        num_requested = end_sample - start_sample
        tile_len = self._tile.shape[0]
        result = np.empty((num_requested, *self._tile.shape[1:]), dtype=np.float32)

        output_pos = 0
        while output_pos < num_requested:
            tile_offset = (start_sample + output_pos) % tile_len
            available = tile_len - tile_offset
            n = min(available, num_requested - output_pos)
            result[output_pos : output_pos + n] = self._tile[tile_offset : tile_offset + n]
            output_pos += n

        return result
