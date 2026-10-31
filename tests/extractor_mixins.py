"""Checks shared by the extractor test files, modeled on the interface test mixins of NeuroConv.

A test file subclasses a mixin once per fixture, sets the extractor class and the keyword arguments of its
constructor, and adds the checks specific to the format as methods of the subclass.
"""

import inspect

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from roiextractors import ImagingExtractor


class ImagingExtractorTestMixin:
    """Checks that every imaging extractor must pass.

    The expected values describe the file, so take them from its README or from another reader, never from the
    extractor.

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
    channel_names : list[str]
        The expected names that `get_available_channel_names()` returns, in order. Only on extractors that have it.
    has_time_vector : bool
        Whether the format records a time for each sample.
    """

    extractor_cls: type[ImagingExtractor]
    extractor_kwargs: dict
    num_samples: int
    sample_shape: tuple[int, int] | tuple[int, int, int]
    dtype: np.dtype
    channel_names: list[str]
    has_time_vector: bool

    # TODO: implement `check_series()` in every test class, as the docstring of `test_get_series()` describes.

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

    def test_get_series(self, setup_extractor):
        """`get_series()` returns the data of the fixture.

        It reads the data once, which is usually the slow part, and gathers here the checks that need it, so the
        suite stays fast. The getter checks never read the data. Here the shape and the dtype of the returned array
        are checked against the expected `num_samples`, `sample_shape` and `dtype`, and then the series goes to
        `check_series()`, where each test class compares it with the same data read by another library.
        """
        series = self.extractor.get_series()
        assert series.shape == (self.num_samples, *self.sample_shape)
        assert series.dtype == self.dtype
        self.check_series(series)

    def check_series(self, series):
        """Check the series that `get_series()` returned, with the data read by another library.

        Override it in a test class. How to read the raw data depends on the format (the pages of `tifffile`, an
        HDF5 dataset), so each class decides what to compare: the full array, or chosen samples against chosen
        pages. It can also check properties of the array specific to the format. It receives the series already
        read, so the data is read once. The goal is that every test class implements it.
        """

    def test_native_timestamps(self, setup_extractor):
        """The extractor reports whether the format records times, and returns one per sample when it does."""
        assert self.extractor.has_time_vector() == self.has_time_vector
        native_timestamps = self.extractor.get_native_timestamps()
        if self.has_time_vector:
            assert native_timestamps.shape == (self.num_samples,)
        else:
            assert native_timestamps is None

    # TODO: check `slice_samples()` here together with the fix for the `TypeError` it raises today when a bound is
    # left out (`_num_samples` is computed before the defaults are applied). The check should compare the number of
    # samples of the slice with the expected `num_samples`, leave a bound out, and read a range of the slice.

    def test_slice_field_of_view(self, setup_extractor):
        """`slice_field_of_view()` returns the same rows and columns of the full read, on every plane.

        The crop has different bounds and a different size on the rows and on the columns, so it catches an extractor
        that swaps them: the pixels come from another region, and the shape it reports is transposed. With the same
        bounds on both axes, a swap would select the same region of a square frame and pass. The crop asks for C, rows
        [1, 4) and columns [4, 6), and an extractor that swaps the bounds returns S, rows [4, 6) and columns [1, 4),
        shown on the top left corner of the frame:

            column  0 1 2 3 4 5 6 7
            row 0   . . . . . . . .
                1   . . . . C C . .
                2   . . . . C C . .
                3   . . . . C C . .
                4   . S S S . . . .
                5   . S S S . . . .
                6   . . . . . . . .
        """
        row_start, row_end, column_start, column_end = 1, 4, 4, 6
        sliced_extractor = self.extractor.slice_field_of_view(
            row_start=row_start, row_end=row_end, column_start=column_start, column_end=column_end
        )
        expected_series = self.extractor.get_series()[:, row_start:row_end, column_start:column_end]
        assert sliced_extractor.get_sample_shape() == expected_series.shape[1:]
        assert_array_equal(sliced_extractor.get_series(), expected_series)

    def test_select_plane(self, setup_extractor):
        """The extractor that `select_plane()` returns for the last plane has that plane of the volumetric series."""
        if not self.extractor.is_volumetric:
            pytest.skip("select_plane() applies only to volumetric extractors.")
        last_plane_index = self.sample_shape[2] - 1
        plane_extractor = self.extractor.select_plane(last_plane_index)
        assert plane_extractor.get_sample_shape() == self.sample_shape[:2]
        assert_array_equal(plane_extractor.get_series(), self.extractor.get_series()[..., last_plane_index])

    def test_available_channel_names(self):
        """`get_available_channel_names()` returns the expected channel names in order, and each is accepted back."""
        if not hasattr(self.extractor_cls, "get_available_channel_names"):
            pytest.skip(f"{self.extractor_cls.__name__} has no get_available_channel_names().")
        discovery_method = self.extractor_cls.get_available_channel_names
        discovery_parameters = inspect.signature(discovery_method).parameters
        discovery_kwargs = {key: value for key, value in self.extractor_kwargs.items() if key in discovery_parameters}
        assert discovery_method(**discovery_kwargs) == self.channel_names

        for channel_name in self.channel_names:
            self.extractor_cls(**{**self.extractor_kwargs, "channel_name": channel_name})

    def test_channel_name_is_required_with_several_channels(self):
        """Without `channel_name`, the extractor raises on data with several channels and builds on data with one."""
        if not hasattr(self.extractor_cls, "get_available_channel_names"):
            pytest.skip(f"{self.extractor_cls.__name__} has no get_available_channel_names().")
        kwargs_without_channel = {key: value for key, value in self.extractor_kwargs.items() if key != "channel_name"}
        if len(self.channel_names) > 1:
            # TODO: the error messages differ between formats. Unify them, for example so that all of them list the
            # available channel names, and match the message here.
            with pytest.raises(ValueError):
                self.extractor_cls(**kwargs_without_channel)
        else:
            self.extractor_cls(**kwargs_without_channel)
