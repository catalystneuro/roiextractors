"""Checks shared by the extractor test files, modeled on the interface test mixins of NeuroConv.

A test file subclasses a mixin once per fixture, sets the extractor class and the keyword arguments of its
constructor, and adds the checks specific to the format as methods of the subclass.
"""

import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from roiextractors import ImagingExtractor


class ImagingExtractorTestMixin:
    """Checks that every imaging extractor must pass.

    The checks read a range that leaves out the first and the last sample, so the fixture needs at least three
    samples. The expected values describe the file, so take them from its README or from another reader, never
    from the extractor.

    Attributes
    ----------
    extractor_cls : type[ImagingExtractor]
        The extractor class, not an instance.
    extractor_kwargs : dict
        The keyword arguments of the constructor.
    num_samples : int
        The expected number of samples.
    sample_shape : tuple[int, int] | tuple[int, int, int]
        The expected shape of a sample, (rows, columns) for planar data and (rows, columns, planes) for volumetric.
    dtype : np.dtype
        The expected dtype of the pixels.
    """

    extractor_cls: type[ImagingExtractor]
    extractor_kwargs: dict
    num_samples: int
    sample_shape: tuple[int, int] | tuple[int, int, int]
    dtype: np.dtype

    @pytest.fixture
    def setup_extractor(self):
        """Build a fresh extractor for each test."""
        self.extractor = self.extractor_cls(**self.extractor_kwargs)
        return self.extractor

    def test_num_samples(self, setup_extractor):
        """`get_num_samples()` is the expected number of samples."""
        assert self.extractor.get_num_samples() == self.num_samples

    def test_sample_shape(self, setup_extractor):
        """`get_sample_shape()`, `get_image_shape()`, `is_volumetric` and `get_num_planes()` agree with `sample_shape`."""
        assert self.extractor.get_sample_shape() == self.sample_shape
        assert self.extractor.get_image_shape() == self.sample_shape[:2]
        is_volumetric = len(self.sample_shape) == 3
        assert self.extractor.is_volumetric == is_volumetric
        # Planar extractors disagree on get_num_planes(), some raise and some return 1
        if is_volumetric:
            assert self.extractor.get_num_planes() == self.sample_shape[2]

    def test_dtype(self, setup_extractor):
        """`get_dtype()` returns the expected dtype as a `np.dtype`."""
        # Equality alone would pass for a scalar type or a string too
        assert isinstance(self.extractor.get_dtype(), np.dtype)
        assert self.extractor.get_dtype() == self.dtype

    def test_series_shape_and_dtype(self, setup_extractor):
        """`get_series()` has the expected shape and dtype."""
        series = self.extractor.get_series()
        assert series.shape == (self.num_samples, *self.sample_shape)
        assert series.dtype == self.dtype

    def test_series_range(self, setup_extractor):
        """`get_series(start_sample, end_sample)` returns the same samples as the full read."""
        num_samples = self.extractor.get_num_samples()
        series = self.extractor.get_series(start_sample=1, end_sample=num_samples - 1)
        assert_array_equal(series, self.extractor.get_series()[1 : num_samples - 1])

    def test_samples(self, setup_extractor):
        """`get_samples()` returns the same samples as the full read, also when they are not contiguous."""
        sample_indices = [0, self.extractor.get_num_samples() - 1]
        assert_array_equal(self.extractor.get_samples(sample_indices), self.extractor.get_series()[sample_indices])

    def test_timestamps(self, setup_extractor):
        """`get_timestamps()` has one time per sample, and a range of it equals the same range of the full read."""
        num_samples = self.extractor.get_num_samples()
        timestamps = self.extractor.get_timestamps()
        assert timestamps.shape == (num_samples,)
        assert_array_equal(
            self.extractor.get_timestamps(start_sample=1, end_sample=num_samples - 1), timestamps[1 : num_samples - 1]
        )

    def test_slice_samples(self, setup_extractor):
        """`slice_samples()` returns the samples and the timestamps of the same range of the parent."""
        num_samples = self.extractor.get_num_samples()
        sliced_extractor = self.extractor.slice_samples(start_sample=1, end_sample=num_samples - 1)
        assert sliced_extractor.get_num_samples() == num_samples - 2
        assert_array_equal(
            sliced_extractor.get_series(), self.extractor.get_series(start_sample=1, end_sample=num_samples - 1)
        )
        assert_allclose(
            sliced_extractor.get_timestamps(), self.extractor.get_timestamps(start_sample=1, end_sample=num_samples - 1)
        )

    def test_slice_field_of_view(self, setup_extractor):
        """`slice_field_of_view()` returns the same rows and columns of the full read, on every plane."""
        num_rows, num_columns = self.extractor.get_image_shape()
        sliced_extractor = self.extractor.slice_field_of_view(
            row_start=1, row_end=num_rows - 1, column_start=1, column_end=num_columns - 1
        )
        expected_series = self.extractor.get_series()[:, 1:-1, 1:-1]
        assert sliced_extractor.get_sample_shape() == expected_series.shape[1:]
        assert_array_equal(sliced_extractor.get_series(), expected_series)

    def test_select_plane(self, setup_extractor):
        """Each extractor that `select_plane()` returns has the same plane of the volumetric series."""
        if not self.extractor.is_volumetric:
            pytest.skip("select_plane() applies only to volumetric extractors.")
        series = self.extractor.get_series()
        for plane_index in range(self.extractor.get_num_planes()):
            plane_extractor = self.extractor.select_plane(plane_index)
            assert_array_equal(plane_extractor.get_series(), series[..., plane_index])

    def test_available_channel_names_are_accepted(self):
        """Every name that `get_available_channel_names()` returns is accepted back as `channel_name`."""
        if not hasattr(self.extractor_cls, "get_available_channel_names"):
            pytest.skip(f"{self.extractor_cls.__name__} has no get_available_channel_names().")
        discovery_method = self.extractor_cls.get_available_channel_names
        discovery_parameters = inspect.signature(discovery_method).parameters
        discovery_kwargs = {key: value for key, value in self.extractor_kwargs.items() if key in discovery_parameters}
        channel_names = discovery_method(**discovery_kwargs)

        assert isinstance(channel_names, list)
        assert all(isinstance(channel_name, str) for channel_name in channel_names)
        for channel_name in channel_names:
            self.extractor_cls(**{**self.extractor_kwargs, "channel_name": channel_name})
