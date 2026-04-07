"""Base class definitions for all ImagingExtractors.

Classes
-------
ImagingExtractor
    Abstract class that contains all the meta-data and input data from the imaging data.
"""

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from math import prod

import numpy as np
from numpy.typing import ArrayLike

from .core_utils import _convert_bytes_to_str, _convert_seconds_to_str


class ImagingExtractor(ABC):
    """Abstract class that contains all the meta-data and input data from the imaging data."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the ImagingExtractor object."""
        self._args = args
        self._kwargs = kwargs
        self._times = None
        self.name = self.__class__.__name__
        self.is_volumetric = False

    def _repr_text(self) -> str:
        """Generate text representation of the ImagingExtractor object."""
        num_samples = self.get_num_samples()
        sample_shape = self.get_sample_shape()
        dtype = self.get_dtype()
        sf_hz = self.get_sampling_frequency()

        # Format sampling frequency
        if not sf_hz.is_integer():
            sampling_frequency_repr = f"{sf_hz:f} Hz"
        else:
            sampling_frequency_repr = f"{sf_hz:0.1f}Hz"

        # Calculate duration
        duration = num_samples / sf_hz
        duration_repr = _convert_seconds_to_str(duration)

        # Calculate memory size using product of all dimensions in image_size
        memory_size = num_samples * prod(sample_shape) * dtype.itemsize
        memory_repr = _convert_bytes_to_str(memory_size)

        # Format shape string based on whether data is volumetric or not
        sample_shape_repr = f"{sample_shape[0]} rows x {sample_shape[1]} columns "
        if self.is_volumetric:
            sample_shape_repr += f"x {sample_shape[2]} planes"

        return (
            f"{self.name}\n"
            f"  Number of samples: {num_samples:,} \n"
            f"  Sample shape: {sample_shape_repr} \n"
            f"  Sampling rate: {sampling_frequency_repr}\n"
            f"  Duration: {duration_repr}\n"
            f"  Imaging data memory: {memory_repr} ({dtype} dtype)"
        )

    def __repr__(self):
        return self._repr_text()

    @abstractmethod
    def get_image_shape(self) -> tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        pass

    def get_frame_shape(self) -> tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        frame_shape: tuple
            Shape of the video frame (num_rows, num_columns).

        Notes
        -----
        This method is equivalent to get_image_shape()
        """
        return self.get_image_shape()

    def get_sample_shape(self) -> tuple[int, int] | tuple[int, int, int]:
        """
        Get the shape of a single sample elements from the series.

        If the series is volumetric, the shape is the shape of the volume and otherwise
        returns the shape of a single frame/image.

        Returns
        -------
        sample_shape: tuple
            Shape of a single sample from the time series (num_rows, num_columns, num_planes)
            if volumetric (num_rows, num_columns) otherwise
        """
        if self.is_volumetric:
            return self.get_volume_shape()
        else:
            return self.get_frame_shape()

    def get_num_planes(self) -> int:
        """Get the number of depth planes.

        Returns
        -------
        num_planes: int
            The number of depth planes.

        Raises
        ------
        NotImplementedError
            If the extractor is not volumetric.
        """
        if not self.is_volumetric:
            raise NotImplementedError(
                "This extractor is not volumetric. "
                "The get_num_planes method is only available for volumetric extractors."
            )
        raise NotImplementedError("This method must be implemented by volumetric extractor subclasses.")

    def get_volume_shape(self) -> tuple[int, int, int]:
        """Get the shape of the volumetric video (num_rows, num_columns, num_planes).

        Returns
        -------
        video_shape: tuple
            Shape of the volumetric video (num_rows, num_columns, num_planes).

        Raises
        ------
        NotImplementedError
            If the extractor is not volumetric or does not implement get_num_planes method.
        """
        if not self.is_volumetric:
            raise NotImplementedError(
                "This extractor is not volumetric. "
                "The get_volume_shape method is only available for volumetric extractors."
            )

        if not hasattr(self, "get_num_planes") or not callable(getattr(self, "get_num_planes")):
            raise NotImplementedError(
                "This extractor does not implement get_num_planes method. "
                "The get_volume_shape method requires the get_num_planes method to be implemented."
            )

        frame_shape = self.get_frame_shape()
        return (frame_shape[0], frame_shape[1], self.get_num_planes())

    @abstractmethod
    def get_num_samples(self) -> int:
        """Get the number of samples in the video.

        Returns
        -------
        num_samples: int
            Number of samples in the video.
        """
        pass

    @abstractmethod
    def get_sampling_frequency(self) -> float:
        """Get the sampling frequency in Hz.

        Returns
        -------
        sampling_frequency: float
            Sampling frequency in Hz.
        """
        pass

    def get_channel_names(self) -> list:
        """Get the channel names in the recoding.

        Returns
        -------
        channel_names: list
            List of strings of channel names
        """
        return [f"channel_{i}" for i in range(self.get_num_channels())]

    def get_dtype(self) -> np.dtype:
        """Get the data type of the video.

        Returns
        -------
        dtype: dtype
            Data type of the video.
        """
        return self.get_series(start_sample=0, end_sample=2).dtype

    @abstractmethod
    def get_series(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
        """Get the series of samples.

        Parameters
        ----------
        start_sample: int, optional
            Start sample index (inclusive).
        end_sample: int, optional
            End sample index (exclusive).

        Returns
        -------
        series: numpy.ndarray
            The series of samples.

        Notes
        -----
        Importantly, we follow the convention that the dimensions of the array are returned in their matrix order,
        More specifically:
        (time, height, width)

        Which is equivalent to:
        (samples, rows, columns)

        For volumetric data, the dimensions are:
        (time, height, width, planes)

        Which is equivalent to:
        (samples, rows, columns, planes)

        Note that this does not match the cartesian convention:
        (t, x, y)

        Where x is the columns width or and y is the rows or height.
        """
        pass

    def get_samples(self, sample_indices: ArrayLike) -> np.ndarray:
        """Get specific samples from indices (not necessarily continuous).

        Parameters
        ----------
        sample_indices: array-like
            Indices of samples to return.

        Returns
        -------
        samples: numpy.ndarray
            The samples.
        """
        sample_indices = np.asarray(sample_indices)
        assert (
            max(sample_indices) < self.get_num_samples()
        ), "'sample_indices' range beyond number of available samples!"
        if np.all(np.diff(sample_indices) == 1):
            return self.get_series(start_sample=sample_indices[0], end_sample=sample_indices[-1] + 1)
        relative_indices = np.array(sample_indices) - sample_indices[0]
        series = self.get_series(start_sample=sample_indices[0], end_sample=sample_indices[-1] + 1)
        return series[relative_indices]

    @abstractmethod
    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        """
        Retrieve the original unaltered timestamps for the data in this interface.

        This function should retrieve the data on-demand by re-initializing the IO.
        Can be overridden to return None if the extractor does not have native timestamps.

        Parameters
        ----------
        start_sample : int, optional
            The starting sample index. If None, starts from the beginning.
        end_sample : int, optional
            The ending sample index. If None, goes to the end.

        Returns
        -------
        timestamps: numpy.ndarray or None
            The timestamps for the data stream, or None if native timestamps are not available.
        """
        return None

    def get_timestamps(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
        """
        Retrieve the timestamps for the data in this extractor.

        Parameters
        ----------
        start_sample : int, optional
            The starting sample index. If None, starts from the beginning.
        end_sample : int, optional
            The ending sample index. If None, goes to the end.

        Returns
        -------
        timestamps: numpy.ndarray
            The timestamps for the data stream.
        """
        # Set defaults
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self.get_num_samples()

        # Return cached timestamps if available
        if self._times is not None:
            return self._times[start_sample:end_sample]

        # See if native timetstamps are available from the format
        native_timestamps = self.get_native_timestamps()
        if native_timestamps is not None:
            self._times = native_timestamps  # Cache the native timestamps
            return native_timestamps[start_sample:end_sample]

        # Fallback to calculated timestamps from sampling frequency
        sample_indices = np.arange(start_sample, end_sample)
        return sample_indices / self.get_sampling_frequency()

    def time_to_sample_indices(self, times: float | ArrayLike) -> float | np.ndarray:
        """Convert user-inputted times (in seconds) to sample indices.

        Parameters
        ----------
        times: float or array-like
            The times (in seconds) to be converted to sample indices.

        Returns
        -------
        sample_indices: float or array-like
            The corresponding sample indices.
        """
        # Default implementation
        if self._times is None:
            return np.round(times * self.get_sampling_frequency()).astype("int64")
        else:
            return np.searchsorted(self._times, times).astype("int64")

    def set_times(self, times: ArrayLike) -> None:
        """Set the recording times (in seconds) for each frame.

        Parameters
        ----------
        times: array-like
            The times in seconds for each frame.

        Raises
        ------
        ValueError
            If the length of 'times' does not match the number of samples.
        """
        num_samples = self.get_num_samples()
        num_timestamps = len(times)

        if num_timestamps != num_samples:
            raise ValueError(
                f"Mismatch between the number of samples and timestamps: "
                f"{num_samples} samples, but {num_timestamps} timestamps provided. "
                "Ensure the length of 'times' matches the number of samples."
            )

        self._times = np.array(times).astype("float64", copy=False)

    def has_time_vector(self) -> bool:
        """Detect if the ImagingExtractor has a time vector set or not.

        Returns
        -------
        has_times: bool
            True if the ImagingExtractor has a time vector set, otherwise False.
        """
        return self._times is not None

    def copy_times(self, extractor) -> None:
        """Copy times from another extractor.

        Parameters
        ----------
        extractor
            The extractor from which the times will be copied.
        """
        if extractor._times is not None:
            self.set_times(deepcopy(extractor._times))

    def slice_samples(self, start_sample: int | None = None, end_sample: int | None = None):
        """Return a new ImagingExtractor ranging from the start_sample to the end_sample.

        Parameters
        ----------
        start_sample: int, optional
            Start sample index (inclusive).
        end_sample: int, optional
            End sample index (exclusive).

        Returns
        -------
        imaging: SampleSlicedImagingExtractor
            The sliced ImagingExtractor object.
        """
        return SampleSlicedImagingExtractor(parent_imaging=self, start_sample=start_sample, end_sample=end_sample)

    def slice_field_of_view(
        self,
        row_start: int | None = None,
        row_end: int | None = None,
        column_start: int | None = None,
        column_end: int | None = None,
    ):
        """Return a new ImagingExtractor with a spatially sliced field of view.

        Parameters
        ----------
        row_start: int, optional
            Starting row index (inclusive). Default is 0.
        row_end: int, optional
            Ending row index (exclusive). Default is the full height.
        column_start: int, optional
            Starting column index (inclusive). Default is 0.
        column_end: int, optional
            Ending column index (exclusive). Default is the full width.

        Returns
        -------
        imaging: FieldOfViewSlicedImagingExtractor
            The spatially sliced ImagingExtractor object.

        Notes
        -----
        This method creates a lazy view of the imaging data with a cropped field of view.
        The slicing is applied to the spatial dimensions (rows and columns) of each frame.
        For volumetric data, all depth planes are preserved with the same spatial crop applied.

        Examples
        --------
        >>> # Crop to center 100x100 region starting at (50, 50)
        >>> cropped_extractor = extractor.slice_field_of_view(row_start=50, row_end=150,
        ...                                                    column_start=50, column_end=150)
        >>>
        >>> # Get top-left quadrant
        >>> height, width = extractor.get_image_shape()
        >>> quadrant = extractor.slice_field_of_view(row_end=height//2, column_end=width//2)
        >>>
        >>> # Compose with temporal slicing
        >>> subset = extractor.slice_samples(0, 1000).slice_field_of_view(100, 200, 100, 200)
        """
        return _FieldOfViewSlicedImagingExtractor(
            parent_imaging=self,
            row_start=row_start,
            row_end=row_end,
            column_start=column_start,
            column_end=column_end,
        )


class SampleSlicedImagingExtractor(ImagingExtractor):
    """Class to get a lazy sample slice.

    Do not use this class directly but use `.slice_samples(...)` on an ImagingExtractor object.
    """

    extractor_name = "SampleSlicedImagingExtractor"

    def __init__(
        self, parent_imaging: ImagingExtractor, start_sample: int | None = None, end_sample: int | None = None
    ):
        """Initialize an ImagingExtractor whose samples subset the parent.

        Subset is exclusive on the right bound, that is, the indexes of this ImagingExtractor range over
        [0, ..., end_sample-start_sample-1], which is used to resolve the index mapping in `get_samples(sample_indices=[...])`.

        Parameters
        ----------
        parent_imaging : ImagingExtractor
            The ImagingExtractor object to subset the samples of.
        start_sample : int, optional
            The left bound of the samples to subset.
            The default is the start sample of the parent.
        end_sample : int, optional
            The right bound of the samples, exclusively, to subset.
            The default is end sample of the parent.
        """
        self._parent_imaging = parent_imaging
        self._start_sample = start_sample
        self._end_sample = end_sample
        self._num_samples = self._end_sample - self._start_sample

        parent_size = self._parent_imaging.get_num_samples()
        if start_sample is None:
            start_sample = 0
        else:
            assert 0 <= start_sample < parent_size
        if end_sample is None:
            end_sample = parent_size
        else:
            assert 0 < end_sample <= parent_size
        assert end_sample > start_sample, "'start_sample' must be smaller than 'end_sample'!"

        super().__init__()
        if getattr(self._parent_imaging, "_times") is not None:
            self._times = self._parent_imaging._times[start_sample:end_sample]

        # Inherit volumetric properties from parent
        self.is_volumetric = self._parent_imaging.is_volumetric

    def get_samples(self, sample_indices: ArrayLike) -> np.ndarray:
        assert max(sample_indices) < self._num_samples, "'sample_indices' range beyond number of available samples!"
        mapped_sample_indices = np.array(sample_indices) + self._start_sample
        return self._parent_imaging.get_samples(sample_indices=mapped_sample_indices)

    def get_series(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
        assert start_sample is None or start_sample >= 0, (
            f"'start_sample' must be greater than or equal to zero! Received '{start_sample}'.\n"
            "Negative slicing semantics are not supported."
        )
        start_sample_shifted = (start_sample or 0) + self._start_sample
        end_sample_shifted = end_sample
        if end_sample is not None:
            end_sample_shifted = end_sample + self._start_sample
        return self._parent_imaging.get_series(start_sample=start_sample_shifted, end_sample=end_sample_shifted)

    def get_image_shape(self) -> tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._parent_imaging.get_image_shape()

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_sampling_frequency(self) -> float:
        return self._parent_imaging.get_sampling_frequency()

    def get_channel_names(self) -> list:
        warnings.warn(
            "get_channel_names is deprecated and will be removed in May 2026 or after.",
            category=FutureWarning,
            stacklevel=2,
        )
        return self._parent_imaging.get_channel_names()

    def get_num_planes(self) -> int:
        """Get the number of depth planes.

        Returns
        -------
        num_planes: int
            The number of depth planes.

        Raises
        ------
        NotImplementedError
            If the parent extractor is not volumetric.
        """
        if not self.is_volumetric:
            raise NotImplementedError(
                "This extractor is not volumetric. "
                "The get_num_planes method is only available for volumetric extractors."
            )
        return self._parent_imaging.get_num_planes()

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        # Get the full original timestamps from parent, but return only our slice range
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self.get_num_samples()

        # Map relative indices to absolute indices in the parent
        actual_start = self._start_sample + start_sample
        actual_end = self._start_sample + end_sample

        # Get timestamps from parent for our specific range
        return self._parent_imaging.get_native_timestamps(start_sample=actual_start, end_sample=actual_end)

    def has_time_vector(self) -> bool:
        # Override to check parent imaging for time vector
        return self._parent_imaging.has_time_vector()


class _FieldOfViewSlicedImagingExtractor(ImagingExtractor):
    """Class to get a lazy field of view slice.

    Do not use this class directly but use `.slice_field_of_view(...)` on an ImagingExtractor object.
    """

    extractor_name = "FieldOfViewSlicedImagingExtractor"

    def __init__(
        self,
        parent_imaging: ImagingExtractor,
        row_start: int | None = None,
        row_end: int | None = None,
        column_start: int | None = None,
        column_end: int | None = None,
    ):
        """Initialize an ImagingExtractor with a spatially sliced field of view.

        Parameters
        ----------
        parent_imaging : ImagingExtractor
            The ImagingExtractor object to spatially slice.
        row_start : int, optional
            The starting row index (inclusive). Default is 0.
        row_end : int, optional
            The ending row index (exclusive). Default is the full height.
        column_start : int, optional
            The starting column index (inclusive). Default is 0.
        column_end : int, optional
            The ending column index (exclusive). Default is the full width.
        """
        self._parent_imaging = parent_imaging

        # Get parent image shape to determine defaults
        parent_height, parent_width = parent_imaging.get_image_shape()

        # Set defaults and validate
        if row_start is None:
            row_start = 0
        if row_end is None:
            row_end = parent_height
        if column_start is None:
            column_start = 0
        if column_end is None:
            column_end = parent_width

        # Validation
        assert (
            0 <= row_start < parent_height
        ), f"'row_start' ({row_start}) must be >= 0 and < parent height ({parent_height})"
        assert 0 < row_end <= parent_height, f"'row_end' ({row_end}) must be > 0 and <= parent height ({parent_height})"
        assert row_end > row_start, f"'row_end' ({row_end}) must be greater than 'row_start' ({row_start})"
        assert (
            0 <= column_start < parent_width
        ), f"'column_start' ({column_start}) must be >= 0 and < parent width ({parent_width})"
        assert (
            0 < column_end <= parent_width
        ), f"'column_end' ({column_end}) must be > 0 and <= parent width ({parent_width})"
        assert (
            column_end > column_start
        ), f"'column_end' ({column_end}) must be greater than 'column_start' ({column_start})"

        # Store slices
        self._row_slice = slice(row_start, row_end)
        self._column_slice = slice(column_start, column_end)

        # Calculate new image shape
        self._image_shape = (row_end - row_start, column_end - column_start)

        super().__init__()

        # Copy time vector from parent if it exists
        if getattr(self._parent_imaging, "_times") is not None:
            self._times = self._parent_imaging._times

        # Inherit volumetric properties from parent
        self.is_volumetric = self._parent_imaging.is_volumetric

    def get_image_shape(self) -> tuple[int, int]:
        """Get the shape of the spatially sliced video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the sliced video frame (num_rows, num_columns).
        """
        return self._image_shape

    def get_series(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
        """Get the spatially sliced series of samples.

        Parameters
        ----------
        start_sample: int, optional
            Start sample index (inclusive).
        end_sample: int, optional
            End sample index (exclusive).

        Returns
        -------
        series: numpy.ndarray
            The spatially sliced series with shape (samples, rows, columns) for 2D data
            or (samples, rows, columns, planes) for volumetric data.
        """
        # Get full data from parent
        data = self._parent_imaging.get_series(start_sample=start_sample, end_sample=end_sample)

        # Apply spatial slicing
        if self.is_volumetric:
            # Volumetric data: (time, height, width, planes)
            return data[:, self._row_slice, self._column_slice, :]
        else:
            # 2D data: (time, height, width)
            return data[:, self._row_slice, self._column_slice]

    def get_samples(self, sample_indices: ArrayLike) -> np.ndarray:
        """Get specific spatially sliced samples from indices.

        Parameters
        ----------
        sample_indices: array-like
            Indices of samples to return.

        Returns
        -------
        samples: numpy.ndarray
            The spatially sliced samples.
        """
        # Get samples from parent
        data = self._parent_imaging.get_samples(sample_indices=sample_indices)

        # Apply spatial slicing
        if self.is_volumetric:
            return data[:, self._row_slice, self._column_slice, :]
        else:
            return data[:, self._row_slice, self._column_slice]

    def get_num_samples(self) -> int:
        """Get the number of samples.

        Returns
        -------
        num_samples: int
            Number of samples in the video.
        """
        return self._parent_imaging.get_num_samples()

    def get_sampling_frequency(self) -> float:
        """Get the sampling frequency in Hz.

        Returns
        -------
        sampling_frequency: float
            Sampling frequency in Hz.
        """
        return self._parent_imaging.get_sampling_frequency()

    def get_channel_names(self) -> list:
        """Get the channel names.

        Returns
        -------
        channel_names: list
            List of strings of channel names.
        """
        return self._parent_imaging.get_channel_names()

    def get_num_planes(self) -> int:
        """Get the number of depth planes.

        Returns
        -------
        num_planes: int
            The number of depth planes.

        Raises
        ------
        NotImplementedError
            If the parent extractor is not volumetric.
        """
        if not self.is_volumetric:
            raise NotImplementedError(
                "This extractor is not volumetric. "
                "The get_num_planes method is only available for volumetric extractors."
            )
        return self._parent_imaging.get_num_planes()

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        """Get native timestamps from parent (FOV slicing doesn't affect temporal metadata).

        Parameters
        ----------
        start_sample : int, optional
            The starting sample index.
        end_sample : int, optional
            The ending sample index.

        Returns
        -------
        timestamps: numpy.ndarray or None
            The timestamps for the data stream.
        """
        return self._parent_imaging.get_native_timestamps(start_sample=start_sample, end_sample=end_sample)

    def has_time_vector(self) -> bool:
        """Check if parent has time vector.

        Returns
        -------
        has_times: bool
            True if the parent ImagingExtractor has a time vector set.
        """
        return self._parent_imaging.has_time_vector()
