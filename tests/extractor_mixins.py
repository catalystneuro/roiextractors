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

    The expected values describe the file, so take them from its README or from another reader, never from the
    extractor.

    Attributes
    ----------
    extractor_cls : type[ImagingExtractor]
        The extractor class, not an instance.
    extractor_kwargs : dict
        The keyword arguments of the constructor.
    expected_num_samples : int
        The expected number of samples.
    expected_sample_shape : tuple[int, int] | tuple[int, int, int]
        The expected shape of a sample, (rows, columns) for planar data and (rows, columns, planes) for volumetric.
    expected_dtype : np.dtype
        The expected dtype of the pixels.
    expected_channel_names : list[str]
        The expected names that `get_available_channel_names()` returns, in order. Only on extractors that have it.
    expected_has_time_vector : bool
        Whether the format records a time for each sample.
    """

    extractor_cls: type[ImagingExtractor]
    extractor_kwargs: dict
    expected_num_samples: int
    expected_sample_shape: tuple[int, int] | tuple[int, int, int]
    expected_dtype: np.dtype
    expected_channel_names: list[str]
    expected_has_time_vector: bool

    # TODO: implement `check_series()` in every test class, as the docstring of `test_get_series()` describes.

    @pytest.fixture
    def setup_extractor(self):
        """Build a fresh extractor for each test."""
        self.extractor = self.extractor_cls(**self.extractor_kwargs)
        return self.extractor

    def test_num_samples(self, setup_extractor):
        """`get_num_samples()` is the expected number of samples."""
        num_samples = self.extractor.get_num_samples()
        assert num_samples == self.expected_num_samples

    def test_sample_shape(self, setup_extractor):
        """The shape getters and `is_volumetric` agree with `expected_sample_shape`."""
        sample_shape = self.extractor.get_sample_shape()
        image_shape = self.extractor.get_image_shape()
        is_volumetric = self.extractor.is_volumetric
        expected_image_shape = self.expected_sample_shape[:2]
        expected_is_volumetric = len(self.expected_sample_shape) == 3
        assert sample_shape == self.expected_sample_shape
        assert image_shape == expected_image_shape
        assert is_volumetric == expected_is_volumetric
        # Planar extractors disagree on get_num_planes(), some raise and some return 1
        if expected_is_volumetric:
            num_planes = self.extractor.get_num_planes()
            expected_num_planes = self.expected_sample_shape[2]
            assert num_planes == expected_num_planes

    def test_dtype(self, setup_extractor):
        """`get_dtype()` returns the expected dtype as a `np.dtype`."""
        dtype = self.extractor.get_dtype()
        # Equality alone would pass for a scalar type or a string too
        assert isinstance(dtype, np.dtype)
        assert dtype == self.expected_dtype

    def test_get_series(self, setup_extractor):
        """`get_series()` returns the data of the fixture.

        It reads the data once, which is usually the slow part, and gathers here the checks that need it, so the
        suite stays fast. The getter checks never read the data. Here the shape and the dtype of the returned array
        are checked against `expected_num_samples`, `expected_sample_shape` and `expected_dtype`, and then the series
        goes to `check_series()`, where each test class compares it with the same data read by another library.
        """
        series = self.extractor.get_series()
        expected_shape = (self.expected_num_samples, *self.expected_sample_shape)
        assert series.shape == expected_shape
        assert series.dtype == self.expected_dtype
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
        has_time_vector = self.extractor.has_time_vector()
        native_timestamps = self.extractor.get_native_timestamps()
        assert has_time_vector == self.expected_has_time_vector
        if self.expected_has_time_vector:
            expected_shape = (self.expected_num_samples,)
            assert native_timestamps.shape == expected_shape
        else:
            assert native_timestamps is None

    def test_slice_samples(self, setup_extractor):
        """`slice_samples()` returns the samples and the timestamps of the same range of the parent.

        Each bound is left out once, so the slice must default to the start and to the end of the parent. The last
        slice is also read with a range, so its indices must be offset into the parent. The expected samples are cut
        from the full read of the parent, so they do not go through the range reads that the slice itself uses.
        """
        # Read the full series of the parent once and cut the expected samples from it, so the reference does not
        # use the range reads under test
        parent_series = self.extractor.get_series()
        parent_timestamps = self.extractor.get_timestamps()
        slice_length = 2

        # Leave the start out: the slice starts at the first sample of the parent
        first_slice = self.extractor.slice_samples(end_sample=slice_length)
        first_slice_num_samples = first_slice.get_num_samples()
        assert first_slice_num_samples == slice_length
        # A read with no bounds stops at the end of the slice, not at the end of the parent
        first_slice_series = first_slice.get_series()
        expected_first_slice_series = parent_series[:slice_length]
        assert_array_equal(first_slice_series, expected_first_slice_series)

        # Leave the end out: the slice runs to the last sample of the parent
        start_sample = self.expected_num_samples - slice_length
        last_slice = self.extractor.slice_samples(start_sample=start_sample)
        last_slice_num_samples = last_slice.get_num_samples()
        assert last_slice_num_samples == slice_length
        last_slice_series = last_slice.get_series()
        expected_last_slice_series = parent_series[start_sample:]
        assert_array_equal(last_slice_series, expected_last_slice_series)

        # A range of the slice is offset into the parent
        last_slice_second_sample = last_slice.get_series(start_sample=1, end_sample=2)
        expected_last_slice_second_sample = parent_series[start_sample + 1 : start_sample + 2]
        assert_array_equal(last_slice_second_sample, expected_last_slice_second_sample)

        # The slice carries the times of the parent, the native ones where the format records them
        last_slice_timestamps = last_slice.get_timestamps()
        expected_last_slice_timestamps = parent_timestamps[start_sample:]
        assert_allclose(last_slice_timestamps, expected_last_slice_timestamps)

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
        sliced_sample_shape = sliced_extractor.get_sample_shape()
        sliced_series = sliced_extractor.get_series()
        full_series = self.extractor.get_series()
        expected_series = full_series[:, row_start:row_end, column_start:column_end]
        expected_sample_shape = expected_series.shape[1:]
        assert sliced_sample_shape == expected_sample_shape
        assert_array_equal(sliced_series, expected_series)

    def test_select_plane(self, setup_extractor):
        """The extractor that `select_plane()` returns for the last plane has that plane of the volumetric series."""
        if not self.extractor.is_volumetric:
            pytest.skip("select_plane() applies only to volumetric extractors.")
        last_plane_index = self.expected_sample_shape[2] - 1
        plane_extractor = self.extractor.select_plane(last_plane_index)
        plane_sample_shape = plane_extractor.get_sample_shape()
        plane_series = plane_extractor.get_series()
        full_series = self.extractor.get_series()
        expected_plane_sample_shape = self.expected_sample_shape[:2]
        expected_plane_series = full_series[..., last_plane_index]
        assert plane_sample_shape == expected_plane_sample_shape
        assert_array_equal(plane_series, expected_plane_series)

    def test_available_channel_names(self):
        """`get_available_channel_names()` returns the expected channel names in order, and each is accepted back."""
        if not hasattr(self.extractor_cls, "get_available_channel_names"):
            pytest.skip(f"{self.extractor_cls.__name__} has no get_available_channel_names().")
        discovery_method = self.extractor_cls.get_available_channel_names
        discovery_parameters = inspect.signature(discovery_method).parameters
        discovery_kwargs = {key: value for key, value in self.extractor_kwargs.items() if key in discovery_parameters}
        channel_names = discovery_method(**discovery_kwargs)
        assert channel_names == self.expected_channel_names

        for channel_name in self.expected_channel_names:
            self.extractor_cls(**{**self.extractor_kwargs, "channel_name": channel_name})

    def test_channel_name_is_required_with_several_channels(self):
        """Without `channel_name`, the extractor raises on data with several channels and builds on data with one."""
        if not hasattr(self.extractor_cls, "get_available_channel_names"):
            pytest.skip(f"{self.extractor_cls.__name__} has no get_available_channel_names().")
        kwargs_without_channel = {key: value for key, value in self.extractor_kwargs.items() if key != "channel_name"}
        if len(self.expected_channel_names) > 1:
            # TODO: the error messages differ between formats. Unify them, for example so that all of them list the
            # available channel names, and match the message here.
            with pytest.raises(ValueError):
                self.extractor_cls(**kwargs_without_channel)
        else:
            self.extractor_cls(**kwargs_without_channel)
