"""Single-plane view of a volumetric ImagingExtractor.

Classes
-------
_SinglePlaneImagingExtractor
    A lazy planar view of a single depth plane of a volumetric ImagingExtractor.
    Use ``ImagingExtractor.select_plane(...)`` to construct one.
"""

import numpy as np
from numpy.typing import ArrayLike

from .imagingextractor import ImagingExtractor


class _SinglePlaneImagingExtractor(ImagingExtractor):
    """Class to get a lazy planar view of a single depth plane from a volumetric extractor.

    Do not use this class directly but use ``.select_plane(...)`` on a volumetric
    ``ImagingExtractor`` object.
    """

    def __init__(self, parent_imaging: ImagingExtractor, plane_index: int):
        """Initialize a planar ImagingExtractor selecting one depth plane of a parent.

        Parameters
        ----------
        parent_imaging : ImagingExtractor
            The volumetric ImagingExtractor object to select a plane from.
        plane_index : int
            The depth plane to select. Must satisfy ``0 <= plane_index < parent_imaging.get_num_planes()``.
        """
        if not parent_imaging.is_volumetric:
            raise ValueError("select_plane is only valid for volumetric extractors.")
        num_planes = parent_imaging.get_num_planes()
        if not 0 <= plane_index < num_planes:
            raise ValueError(f"plane_index ({plane_index}) must satisfy 0 <= plane_index < num_planes ({num_planes}).")

        self._parent_imaging = parent_imaging
        self._plane_index = plane_index

        super().__init__()
        # The parent's _times are per-volume, not per-plane. For acquisitions where each plane
        # is sampled at a different time within the volume (e.g. ScanImage with framesPerSlice > 1,
        # or any sequential z-scan), this discards the per-plane offset. The same caveat applies
        # to get_native_timestamps below. Per-plane timestamps would require an axis-aware time API.
        if getattr(self._parent_imaging, "_times") is not None:
            self._times = self._parent_imaging._times

        self.is_volumetric = False

    def get_series(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
        """Get the planar series for the selected depth plane.

        Parameters
        ----------
        start_sample: int, optional
            Start sample index (inclusive).
        end_sample: int, optional
            End sample index (exclusive).

        Returns
        -------
        series: numpy.ndarray
            The planar series of shape ``(samples, rows, columns)``.
        """
        series = self._parent_imaging.get_series(start_sample=start_sample, end_sample=end_sample)
        return series[..., self._plane_index]

    def get_samples(self, sample_indices: ArrayLike) -> np.ndarray:
        """Get specific planar samples from indices for the selected depth plane.

        Parameters
        ----------
        sample_indices: array-like
            Indices of samples to return.

        Returns
        -------
        samples: numpy.ndarray
            The planar samples of shape ``(len(sample_indices), rows, columns)``.
        """
        samples = self._parent_imaging.get_samples(sample_indices=sample_indices)
        return samples[..., self._plane_index]

    def get_image_shape(self) -> tuple[int, int]:
        """Get the shape of the planar image (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the planar image (num_rows, num_columns).
        """
        return self._parent_imaging.get_image_shape()

    def get_num_samples(self) -> int:
        """Get the number of samples (delegates to parent)."""
        return self._parent_imaging.get_num_samples()

    def get_sampling_frequency(self) -> float:
        """Get the sampling frequency (delegates to parent)."""
        return self._parent_imaging.get_sampling_frequency()

    def get_dtype(self) -> np.dtype:
        """Get the data type of the video (delegates to parent)."""
        return self._parent_imaging.get_dtype()

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        """Get native timestamps from the parent extractor.

        Parameters
        ----------
        start_sample: int, optional
            Start sample index (inclusive).
        end_sample: int, optional
            End sample index (exclusive).

        Returns
        -------
        timestamps: numpy.ndarray or None
            The native timestamps from the parent. Plane selection does not affect the
            time axis, so the returned timestamps are the same the parent would return.
        """
        return self._parent_imaging.get_native_timestamps(start_sample=start_sample, end_sample=end_sample)

    def has_time_vector(self) -> bool:
        """Check if parent has a time vector set.

        Returns
        -------
        has_times: bool
            True if the parent ImagingExtractor has a time vector set.
        """
        return self._parent_imaging.has_time_vector()
