"""Checks shared by the extractor test files, modeled on the interface test mixins of NeuroConv.

A test file subclasses a mixin once per fixture, sets the extractor class and the keyword arguments of its
constructor, and adds the checks specific to the format as methods of the subclass.
"""

import inspect
import pickle
import re
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from roiextractors import ImagingExtractor
from roiextractors._metadata import Metadata


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
    expected_native_timestamps : list[float] | None
        The first times in seconds that `get_native_timestamps()` returns, read from the file with the header field
        in a comment, or `None` when the format records no time for each sample. Three samples are enough to catch a
        wrong start, step or unit. On volumetric data a format gives each sample the time of one of its planes, so the
        comment names the plane.
    expected_plane_native_timestamps : list[list[float]]
        On volumetric fixtures with `expected_native_timestamps`, the first times in seconds that
        `select_plane(plane_index).get_native_timestamps()` returns, one list for each plane in the order of
        `plane_index`, read from the file with the header field in a comment. Each plane has the times of its own
        frames.
    expected_sampling_frequency : float
        The expected rate of the samples in Hz, along the first axis of `get_series()`, so the volume rate on
        volumetric data.
    expected_metadata : dict
        The fields of `metadata` that are set, as `metadata.to_populated_dict(include_extra=False)` returns them, read
        from the file with the header field in a comment. Every fixture declares it, as an empty dict for a format
        whose extractor fills no field yet.
    """

    extractor_cls: type[ImagingExtractor]
    extractor_kwargs: dict
    expected_num_samples: int
    expected_sample_shape: tuple[int, int] | tuple[int, int, int]
    expected_dtype: np.dtype
    expected_channel_names: list[str]
    expected_native_timestamps: list[float] | None
    expected_plane_native_timestamps: list[list[float]]
    expected_sampling_frequency: float
    expected_metadata: dict

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
        if expected_is_volumetric:
            num_planes = self.extractor.get_num_planes()
            volume_shape = self.extractor.get_volume_shape()
            expected_num_planes = self.expected_sample_shape[2]
            assert num_planes == expected_num_planes
            assert volume_shape == self.expected_sample_shape
        else:
            # The extractors that returned 1 plane on planar data warn until they raise, in or after May 2027
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                with pytest.raises((ValueError, FutureWarning)):
                    self.extractor.get_num_planes()
                with pytest.raises((ValueError, FutureWarning)):
                    self.extractor.get_volume_shape()

    def test_dtype(self, setup_extractor):
        """`get_dtype()` returns the expected dtype as a `np.dtype`."""
        dtype = self.extractor.get_dtype()
        # Equality alone would pass for a scalar type or a string too
        assert isinstance(dtype, np.dtype)
        assert dtype == self.expected_dtype

    def test_sampling_frequency(self, setup_extractor):
        """`get_sampling_frequency()` is the expected rate of the samples."""
        sampling_frequency = self.extractor.get_sampling_frequency()
        # A rate taken from timestamps can differ from the value of the file in the last digits
        assert sampling_frequency == pytest.approx(self.expected_sampling_frequency)

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

        Every test class must override it, and the mixin raises `NotImplementedError` otherwise. How to read the raw
        data depends on the format (the pages of `tifffile`, an HDF5 dataset), so each class decides what to compare:
        the full array, or chosen samples against chosen pages. It can also check properties of the array specific to
        the format. It receives the series already read, so the data is read once.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement check_series() and compare the series with the data read by "
            "another library"
        )

    def test_native_timestamps(self, setup_extractor):
        """The native timestamps are one per sample and start with `expected_native_timestamps`, or are `None`.

        A range of the timestamps is the same range of the full read, for the native timestamps and for
        `get_timestamps()`, which computes them from the rate when the format records none.
        """
        # Asked before the timestamps are read, as the extractor must answer without reading them
        has_time_vector = self.extractor.has_time_vector()
        native_timestamps = self.extractor.get_native_timestamps()
        timestamps = self.extractor.get_timestamps()
        start_sample, end_sample = 1, 3
        if self.expected_native_timestamps is None:
            assert not has_time_vector
            assert native_timestamps is None
        else:
            assert has_time_vector
            expected_shape = (self.expected_num_samples,)
            assert native_timestamps.shape == expected_shape
            # Without times set, `get_timestamps()` returns the native timestamps
            assert_array_equal(timestamps, native_timestamps)

            num_expected_samples = len(self.expected_native_timestamps)
            first_native_timestamps = native_timestamps[:num_expected_samples]
            assert_allclose(first_native_timestamps, self.expected_native_timestamps)

            range_native_timestamps = self.extractor.get_native_timestamps(
                start_sample=start_sample, end_sample=end_sample
            )
            assert_array_equal(range_native_timestamps, native_timestamps[start_sample:end_sample])

        range_timestamps = self.extractor.get_timestamps(start_sample=start_sample, end_sample=end_sample)
        assert_array_equal(range_timestamps, timestamps[start_sample:end_sample])

    def test_set_times(self, setup_extractor):
        """Times set with `set_times()` replace the native ones in `get_timestamps()`, not in `get_native_timestamps()`.

        The set times must survive a lookup with `time_to_sample_indices()` and carry over to a slice.
        """
        native_timestamps = self.extractor.get_native_timestamps()
        # Far from the times of any file, so a native time returned by mistake cannot match
        set_timestamps = 100.0 + 0.5 * np.arange(self.expected_num_samples)
        self.extractor.set_times(set_timestamps)
        assert self.extractor.has_time_vector()
        assert_array_equal(self.extractor.get_timestamps(), set_timestamps)
        native_timestamps_after_set_times = self.extractor.get_native_timestamps()
        if native_timestamps is None:
            assert native_timestamps_after_set_times is None
        else:
            assert_array_equal(native_timestamps_after_set_times, native_timestamps)

        # The lookup once cached the native timestamps over the set ones
        self.extractor.time_to_sample_indices(set_timestamps[1])
        assert_array_equal(self.extractor.get_timestamps(), set_timestamps)

        slice_extractor = self.extractor.slice_samples(start_sample=1, end_sample=self.expected_num_samples)
        assert_array_equal(slice_extractor.get_timestamps(), set_timestamps[1:])

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
        """`select_plane()` returns the last plane of the volumetric series and raises for -1 and `num_planes`.

        Each plane has the times of its own frames, which `expected_plane_native_timestamps` gives, or none when the
        format records no time for each sample.
        """
        if not self.extractor.is_volumetric:
            with pytest.raises(ValueError):
                self.extractor.select_plane(0)
            return
        last_plane_index = self.expected_sample_shape[2] - 1
        plane_extractor = self.extractor.select_plane(last_plane_index)
        plane_sample_shape = plane_extractor.get_sample_shape()
        plane_series = plane_extractor.get_series()
        full_series = self.extractor.get_series()
        expected_plane_sample_shape = self.expected_sample_shape[:2]
        expected_plane_series = full_series[..., last_plane_index]
        assert plane_sample_shape == expected_plane_sample_shape
        assert_array_equal(plane_series, expected_plane_series)

        expected_num_planes = self.expected_sample_shape[2]
        below_range_message = f"plane_index (-1) must satisfy 0 <= plane_index < num_planes ({expected_num_planes})."
        with pytest.raises(ValueError, match=re.escape(below_range_message)):
            self.extractor.select_plane(-1)
        above_range_message = (
            f"plane_index ({expected_num_planes}) must satisfy 0 <= plane_index < num_planes ({expected_num_planes})."
        )
        with pytest.raises(ValueError, match=re.escape(above_range_message)):
            self.extractor.select_plane(expected_num_planes)

        # A format gives a volume the time of one of its planes, and each plane must have the times of its own frames
        for plane_index in range(expected_num_planes):
            plane_native_timestamps = self.extractor.select_plane(plane_index).get_native_timestamps()
            if self.expected_native_timestamps is None:
                assert plane_native_timestamps is None
            else:
                expected_shape = (self.expected_num_samples,)
                assert plane_native_timestamps.shape == expected_shape
                expected_plane_native_timestamps = self.expected_plane_native_timestamps[plane_index]
                num_expected_samples = len(expected_plane_native_timestamps)
                first_plane_native_timestamps = plane_native_timestamps[:num_expected_samples]
                assert_allclose(first_plane_native_timestamps, expected_plane_native_timestamps)

    def test_pickle_round_trip(self, setup_extractor):
        """A pickled extractor comes back as the same class, with the same dtype, data and timestamps.

        A worker process receives the extractor this way. An extractor that holds an open file or a handle fails at
        `pickle.dumps`, and a constructor argument the extractor does not record comes back as other data or times.
        """
        unpickled = pickle.loads(pickle.dumps(self.extractor))

        assert type(unpickled) is type(self.extractor)
        assert unpickled.get_dtype() == self.extractor.get_dtype()
        assert_array_equal(unpickled.get_series(), self.extractor.get_series())
        assert_array_equal(unpickled.get_timestamps(), self.extractor.get_timestamps())

    # The warning that the metadata API is experimental is checked in `tests/test_internals/test_metadata.py`
    @pytest.mark.filterwarnings("ignore:The metadata API is experimental:UserWarning")
    def test_pickle_does_not_grow_after_reading(self, setup_extractor):
        """Reading the data, the timestamps or the metadata does not add what was read to the pickle.

        A worker can read those from the file itself, so a cache that travels with the extractor is sent for nothing.
        """
        payload_before_reading = pickle.dumps(self.extractor)

        self.extractor.get_series()
        self.extractor.get_timestamps()
        self.extractor.metadata

        payload_after_reading = pickle.dumps(self.extractor)
        assert len(payload_after_reading) == len(payload_before_reading)

    def test_pickle_round_trip_of_overridden_views(self, setup_extractor):
        """A view made by the format's own `slice_field_of_view` or `select_plane` comes back as that view.

        The generic views of `ImagingExtractor` are the same code for every format and are tested with the mock in
        `tests/test_internals/test_serialization.py`, so this runs only on formats that override a view method.
        """
        extractor_class = type(self.extractor)
        is_slice_field_of_view_overridden = (
            extractor_class.slice_field_of_view is not ImagingExtractor.slice_field_of_view
        )
        is_select_plane_overridden = extractor_class.select_plane is not ImagingExtractor.select_plane

        views = []
        if is_slice_field_of_view_overridden:
            views.append(self.extractor.slice_field_of_view(row_start=1, row_end=4, column_start=4, column_end=6))
        if self.extractor.is_volumetric and is_select_plane_overridden:
            last_plane_index = self.expected_sample_shape[2] - 1
            views.append(self.extractor.select_plane(last_plane_index))
        if not views:
            pytest.skip("The views of this format are the generic ones, tested with the mock")

        for view in views:
            unpickled_view = pickle.loads(pickle.dumps(view))
            assert type(unpickled_view) is type(view)
            assert_array_equal(unpickled_view.get_series(), view.get_series())
            assert_array_equal(unpickled_view.get_timestamps(), view.get_timestamps())

    # The warning that the metadata API is experimental is checked in `tests/test_internals/test_metadata.py`
    @pytest.mark.filterwarnings("ignore:The metadata API is experimental:UserWarning")
    def test_metadata(self, setup_extractor):
        """`metadata` has the fields of `expected_metadata`, and each view has the metadata of its part.

        A slice of samples has the metadata of the extractor. A crop from `slice_field_of_view()` has the width and the
        height of its own rows and columns. A plane from `select_plane()` has its own z offset.
        """
        # Read before the views are built, so a view that copies the extractor gets the metadata of the full
        # extractor and must replace it
        metadata = self.extractor.metadata
        assert isinstance(metadata, Metadata)
        check_metadata(metadata, self.expected_metadata)

        sample_slice = self.extractor.slice_samples(start_sample=1)
        assert sample_slice.metadata == metadata

        # The crop of `test_slice_field_of_view`, with a different size on the rows and on the columns
        row_start, row_end, column_start, column_end = 1, 4, 4, 6
        cropped_extractor = self.extractor.slice_field_of_view(
            row_start=row_start, row_end=row_end, column_start=column_start, column_end=column_end
        )
        expected_cropped_metadata = dict(self.expected_metadata)
        num_rows, num_columns = self.expected_sample_shape[:2]
        if "field_of_view_height_um" in self.expected_metadata:
            height_um = self.expected_metadata["field_of_view_height_um"]
            expected_cropped_metadata["field_of_view_height_um"] = height_um * (row_end - row_start) / num_rows
        if "field_of_view_width_um" in self.expected_metadata:
            width_um = self.expected_metadata["field_of_view_width_um"]
            expected_cropped_metadata["field_of_view_width_um"] = width_um * (column_end - column_start) / num_columns
        check_metadata(cropped_extractor.metadata, expected_cropped_metadata)

        if not self.extractor.is_volumetric:
            return
        num_planes = self.expected_sample_shape[2]
        for plane_index in range(num_planes):
            plane_metadata = self.extractor.select_plane(plane_index).metadata
            expected_plane_metadata = dict(self.expected_metadata)
            if "plane_z_offsets_um" in self.expected_metadata:
                plane_z_offset_um = self.expected_metadata["plane_z_offsets_um"][plane_index]
                expected_plane_metadata["plane_z_offsets_um"] = [plane_z_offset_um]
            check_metadata(plane_metadata, expected_plane_metadata)

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
            expected_message = (
                "channel_name must be specified for data with more than one channel. "
                f"Available channels: {self.expected_channel_names}"
            )
            with pytest.raises(ValueError, match=re.escape(expected_message)):
                self.extractor_cls(**kwargs_without_channel)
        else:
            self.extractor_cls(**kwargs_without_channel)

    def test_wrong_channel_name_raises(self):
        """A `channel_name` that is not an available channel raises, also on data with one channel."""
        if not hasattr(self.extractor_cls, "get_available_channel_names"):
            pytest.skip(f"{self.extractor_cls.__name__} has no get_available_channel_names().")
        wrong_channel_name = "not_a_channel"
        expected_message = (
            f"Channel '{wrong_channel_name}' not found. Available channels: {self.expected_channel_names}"
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            self.extractor_cls(**{**self.extractor_kwargs, "channel_name": wrong_channel_name})


def check_metadata(metadata: Metadata, expected_metadata: dict):
    """Check that the fields of `metadata` that are set are those of `expected_metadata`.

    The numbers are compared approximately, as the sizes of a field of view are products of the pixel size and the
    expected values are written with the digits of the file.
    """
    populated_metadata = metadata.to_populated_dict(include_extra=False)
    assert populated_metadata.keys() == expected_metadata.keys()
    for field_name, expected_value in expected_metadata.items():
        value = populated_metadata[field_name]
        if isinstance(expected_value, (float, int, list)):
            assert value == pytest.approx(expected_value), field_name
        else:
            assert value == expected_value, field_name
