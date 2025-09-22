"""Base segmentation extractors.

Classes
-------
SegmentationExtractor
    Abstract class that contains all the meta-data and output data from the ROI segmentation operation when applied to
    the pre-processed data. It also contains methods to read from and write to various data formats output from the
    processing pipelines like SIMA, CaImAn, Suite2p, CNMF-E.
SampleSlicedSegmentationExtractor
    Class to get a lazy sample slice.
FrameSliceSegmentationExtractor
    Class to get a lazy frame slice.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .extraction_tools import ArrayType, FloatType, IntType, _pixel_mask_extractor


@dataclass
class RoiResponse:
    """Represents a fluorescence response (trace) with its metadata."""

    response_type: Literal["raw", "dff", "neuropil", "baseline", "background", "denoised", "deconvolved"]
    data: ArrayLike  # Shape: (num_samples, num_rois)
    roi_ids: list[str | int]  # Which ROIs this trace corresponds to


@dataclass
class RoiRepresentation:
    """Represents the spatial representation of an ROI (mask)."""

    roi_id: str | int  # Single ROI identifier
    data: ArrayLike  # Dense array or sparse coordinate list
    representation_type: Literal["dense", "sparse"]
    field_of_view_shape: tuple[int, ...]  # Shape of imaging space: (height, width) or (depth, height, width)
    # sparse: list of (x, y, weight) or (x, y, z, weight) tuples
    # dense: 2D or 3D array

    @property
    def is_volumetric(self) -> bool:
        """True if this is 3D volumetric data, False for 2D."""
        return len(self.field_of_view_shape) == 3

    def to_dense(self) -> np.ndarray:
        """Convert sparse representation to dense format.

        Returns
        -------
        np.ndarray
            Dense representation as a 2D or 3D array matching field_of_view_shape.
        """
        if self.representation_type == "dense":
            return np.array(self.data)

        # Convert sparse to dense
        dense_mask = np.zeros(self.field_of_view_shape, dtype=np.float32)
        coords = np.array(self.data)

        if self.is_volumetric:
            # 3D coordinates: (x, y, z, weight)
            x_coords = coords[:, 0].astype(int)
            y_coords = coords[:, 1].astype(int)
            z_coords = coords[:, 2].astype(int)
            weights = coords[:, 3] if coords.shape[1] > 3 else np.ones(len(coords))

            # Ensure coordinates are within bounds
            valid_indices = (
                (x_coords >= 0)
                & (x_coords < self.field_of_view_shape[2])
                & (y_coords >= 0)
                & (y_coords < self.field_of_view_shape[1])
                & (z_coords >= 0)
                & (z_coords < self.field_of_view_shape[0])
            )
            dense_mask[z_coords[valid_indices], y_coords[valid_indices], x_coords[valid_indices]] = weights[
                valid_indices
            ]
        else:
            # 2D coordinates: (x, y, weight)
            x_coords = coords[:, 0].astype(int)
            y_coords = coords[:, 1].astype(int)
            weights = coords[:, 2] if coords.shape[1] > 2 else np.ones(len(coords))

            # Ensure coordinates are within bounds
            valid_indices = (
                (x_coords >= 0)
                & (x_coords < self.field_of_view_shape[1])
                & (y_coords >= 0)
                & (y_coords < self.field_of_view_shape[0])
            )
            dense_mask[y_coords[valid_indices], x_coords[valid_indices]] = weights[valid_indices]

        return dense_mask

    def to_sparse(self) -> list[tuple]:
        """Convert dense representation to sparse format.

        Returns
        -------
        list[tuple]
            Sparse representation as [(x, y, weight), ...] for 2D or [(x, y, z, weight), ...] for 3D.
        """
        if self.representation_type == "sparse":
            return list(self.data)

        # Convert dense to sparse
        dense_array = np.array(self.data)

        if self.is_volumetric:
            # 3D case
            z_coords, y_coords, x_coords = np.nonzero(dense_array)
            weights = dense_array[z_coords, y_coords, x_coords]
            return [(int(x), int(y), int(z), float(w)) for x, y, z, w in zip(x_coords, y_coords, z_coords, weights)]
        else:
            # 2D case
            y_coords, x_coords = np.nonzero(dense_array)
            weights = dense_array[y_coords, x_coords]
            return [(int(x), int(y), float(w)) for x, y, w in zip(x_coords, y_coords, weights)]


class SegmentationExtractor(ABC):
    """Abstract segmentation extractor class.

    An abstract class that contains all the meta-data and output data from
    the ROI segmentation operation when applied to the pre-processed data.
    It also contains methods to read from and write to various data formats
    output from the processing pipelines like SIMA, CaImAn, Suite2p, CNMF-E.
    All the methods with @abstract decorator have to be defined by the
    format specific classes that inherit from this.
    """

    def __init__(self):
        """Create a new SegmentationExtractor for a specific data format (unique to each child SegmentationExtractor)."""
        self._sampling_frequency = None
        self._times = None
        self._channel_names = ["OpticalChannel"]
        self._num_planes = 1

        self._roi_ids = None  # Optional explicit ROI IDs
        self._roi_responses: list[RoiResponse] = []
        self._roi_representations: dict[int | str, RoiRepresentation] = {}  # {roi_id: RoiRepresentation}
        self._summary_images: dict = {}  # Summary images (mean, correlation, max projection, etc.)
        self._properties = {}

    @abstractmethod
    def get_accepted_list(self) -> list:
        """Get a list of accepted ROI ids.

        Returns
        -------
        accepted_list: list
            List of accepted ROI ids.
        """
        pass

    @abstractmethod
    def get_rejected_list(self) -> list:
        """Get a list of rejected ROI ids.

        Returns
        -------
        rejected_list: list
            List of rejected ROI ids.
        """
        pass

    @abstractmethod
    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Get the original timestamps from the data source.

        Parameters
        ----------
        start_sample : int, optional
            Start sample index (inclusive).
        end_sample : int, optional
            End sample index (exclusive).

        Returns
        -------
        timestamps : np.ndarray or None
            The original timestamps in seconds, or None if not available.
        """
        pass

    @abstractmethod
    def get_frame_shape(self) -> ArrayType:
        """Get frame size of movie (height, width).

        Returns
        -------
        frame_shape: array_like
            2-D array: image height x image width
        """
        pass

    def get_image_size(self) -> ArrayType:
        """Get frame size of movie (height, width).

        .. deprecated:: on or after January 2026
           Use :meth:`get_frame_shape` instead.

        Returns
        -------
        no_rois: array_like
            2-D array: image height x image width
        """
        warnings.warn(
            "get_image_size is deprecated and will be removed on or after January 2026. "
            "Use get_frame_shape instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_frame_shape()

    def get_num_samples(self) -> int:
        """Get the number of samples in the recording (duration of recording).

        Returns
        -------
        num_samples: int
            Number of samples in the recording.
        """
        for trace in self.get_traces_dict().values():
            if trace is not None and len(trace.shape) > 0:
                return trace.shape[0]

    def get_num_frames(self) -> int:
        """Get the number of frames in the recording (duration of recording).

        .. deprecated:: on or after January 2026
           Use :meth:`get_num_samples` instead.

        Returns
        -------
        num_frames: int
            Number of frames in the recording.
        """
        warnings.warn(
            "get_num_frames is deprecated and will be removed on or after January 2026. "
            "Use get_num_samples instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_num_samples()

    def get_roi_locations(self, roi_ids=None) -> np.ndarray:
        """Get the locations of the Regions of Interest (ROIs).

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs requested.

        Returns
        -------
        roi_locs: numpy.ndarray
            2-D array: 2 X no_ROIs. The pixel ids (x,y) where the centroid of the ROI is.
        """
        if roi_ids is None:
            roi_ids = self.get_roi_ids()

        roi_location = np.zeros([2, len(roi_ids)], dtype="int")
        for roi_index, roi_id in enumerate(roi_ids):
            image_mask = self.get_roi_image_masks(roi_ids=[roi_id])
            temp = np.where(image_mask == np.amax(image_mask))
            roi_location[:, roi_index] = np.array([np.median(temp[0]), np.median(temp[1])]).T
        return roi_location

    def get_roi_ids(self) -> list:
        """Get the list of ROI ids.

        Returns
        -------
        roi_ids: list
            List of roi ids.
        """
        # Check if explicit ROI IDs were set
        if self._roi_ids is not None:
            return list(self._roi_ids)

        # Generate sequential indices
        return list(range(self.get_num_rois()))

    def get_roi_image_masks(self, roi_ids=None) -> np.ndarray:
        """Get the image masks extracted from segmentation algorithm.

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs requested.

        Returns
        -------
        image_masks: numpy.ndarray
            3-D array(val 0 or 1): image_height X image_width X length(roi_ids)
        """
        all_roi_ids = self.get_roi_ids()
        if roi_ids is None:
            roi_ids = all_roi_ids

        # Build dense masks from representations
        masks = []
        for roi_id in roi_ids:
            if roi_id in self._roi_representations:
                rep = self._roi_representations[roi_id]
                dense_mask = rep.to_dense()
                masks.append(dense_mask)

        if masks:
            return np.stack(masks, axis=2)
        else:
            # Return empty array with correct shape
            frame_shape = self.get_frame_shape()
            return np.zeros((*frame_shape, 0))

    def get_roi_pixel_masks(self, roi_ids=None) -> np.array:
        """Get the weights applied to each of the pixels of the mask.

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs requested.

        Returns
        -------
        pixel_masks: list
            List of length number of rois, each element is a 2-D array with shape (number_of_non_zero_pixels, 3).
            Columns 1 and 2 are the x and y coordinates of the pixel, while the third column represents the weight of
            the pixel.
        """
        if roi_ids is None:
            roi_ids = range(self.get_num_rois())

        return _pixel_mask_extractor(self.get_roi_image_masks(roi_ids=roi_ids), roi_ids)

    def get_background_ids(self) -> list:
        """Get the list of background components ids.

        Returns
        -------
        background_components_ids: list
            List of background components ids.
        """
        return list(range(self.get_num_background_components()))

    def get_background_image_masks(self, background_ids=None) -> np.ndarray:
        """Get the background image masks extracted from segmentation algorithm.

        Parameters
        ----------
        background_ids: array_like
            A list or 1D array of ids of the background components. Length is the number of background components requested.

        Returns
        -------
        background_image_masks: numpy.ndarray
            3-D array(val 0 or 1): image_height X image_width X length(background_ids)
        """
        # Get background ROI IDs
        all_background_ids = [rid for rid in self._roi_representations.keys() if str(rid).startswith("background")]

        if background_ids is None:
            background_ids = all_background_ids

        # Build dense masks from background representations
        masks = []
        for bg_id in background_ids:
            if bg_id in self._roi_representations:
                rep = self._roi_representations[bg_id]
                dense_mask = rep.to_dense()
                masks.append(dense_mask)

        if masks:
            return np.stack(masks, axis=2)
        else:
            # Return empty array with correct shape
            frame_shape = self.get_frame_shape()
            return np.zeros((*frame_shape, 0))

    def get_background_pixel_masks(self, background_ids=None) -> np.array:
        """Get the weights applied to each of the pixels of the mask.

        Parameters
        ----------
        background_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs requested.

        Returns
        -------
        pixel_masks: list
            List of length number of rois, each element is a 2-D array with shape (number_of_non_zero_pixels, 3).
            Columns 1 and 2 are the x and y coordinates of the pixel, while the third column represents the weight of
            the pixel.
        """
        if background_ids is None:
            background_ids = range(self.get_num_background_components())

        return _pixel_mask_extractor(self.get_background_image_masks(background_ids=background_ids), background_ids)

    def slice_samples(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None):
        """Return a new SegmentationExtractor ranging from the start_sample to the end_sample.

        Parameters
        ----------
        start_sample: int, optional
            Start sample index (inclusive).
        end_sample: int, optional
            End sample index (exclusive).

        Returns
        -------
        segmentation: SampleSlicedSegmentationExtractor
            The sliced SegmentationExtractor object.
        """
        return SampleSlicedSegmentationExtractor(
            parent_segmentation=self, start_sample=start_sample, end_sample=end_sample
        )

    def frame_slice(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None):
        """Return a new SegmentationExtractor ranging from the start_frame to the end_frame.

        Parameters
        ----------
        start_frame: int, optional
            Start frame index (inclusive).
        end_frame: int, optional
            End frame index (exclusive).

        Returns
        -------
        frame_slice_segmentation_extractor: FrameSliceSegmentationExtractor
            The frame slice segmentation extractor object.

        Deprecated
        ----------
        This method will be removed on or after January 2026.
        Use slice_samples() instead.
        """
        warnings.warn(
            "frame_slice() is deprecated and will be removed on or after January 2026. " "Use slice_samples() instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.slice_samples(start_sample=start_frame, end_sample=end_frame)

    def get_traces(
        self,
        roi_ids: ArrayType = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        name: str = "raw",
    ) -> ArrayType:
        """Get the traces of each ROI specified by roi_ids.

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs requested.
        start_frame: int
            The starting frame of the trace.
        end_frame: int
            The ending frame of the trace.
        name: str
            The name of the trace to retrieve ex. 'raw', 'dff', 'neuropil', 'deconvolved'

        Returns
        -------
        traces: array_like
            2-D array (ROI x timepoints)
        """
        # Find the requested trace type
        for response in self._roi_responses:
            if response.response_type == name:
                data = response.data
                if roi_ids is not None:
                    indices = [response.roi_ids.index(rid) for rid in roi_ids]
                    data = data[:, indices]
                return np.array(data[start_frame:end_frame, :])

        # Return None if trace type not found, this is for backwards compatibility
        return None

    def get_traces_dict(self) -> dict:
        """Get traces as a dictionary with key as the name of the ROiResponseSeries.

        Returns
        -------
        _roi_response_dict: dict
            dictionary with key, values representing different types of RoiResponseSeries:
                Raw Fluorescence, DeltaFOverF, Denoised, Neuropil, Deconvolved, Background, etc.
        """
        # Build dict from available traces
        result = {r.response_type: r.data for r in self._roi_responses}

        # Ensure all expected trace types are present in the dictionary
        # This is for backwards compatibility
        expected_types = ["raw", "dff", "neuropil", "deconvolved", "denoised", "baseline", "background"]
        for trace_type in expected_types:
            if trace_type not in result:
                result[trace_type] = None

        return result

    def get_images_dict(self) -> dict:
        """Get images as a dictionary with key as the name of the ROIResponseSeries.

        Returns
        -------
        _roi_image_dict: dict
            dictionary with key, values representing different types of Images used in segmentation:
                Mean, Correlation image, Maximum projection, etc.
        """
        return dict(self._summary_images)

    def get_image(self, name: str = "correlation") -> ArrayType:
        """Get specific images: mean or correlation.

        Parameters
        ----------
        name:str
            name of the type of image to retrieve

        Returns
        -------
        images: numpy.ndarray
        """
        if name not in self.get_images_dict():
            raise ValueError(f"could not find {name} image, enter one of {list(self.get_images_dict().keys())}")
        return self.get_images_dict().get(name)

    def get_sampling_frequency(self) -> float:
        """Get the sampling frequency in Hz.

        Returns
        -------
        sampling_frequency: float
            Sampling frequency of the recording in Hz.
        """
        if self._sampling_frequency is not None:
            return float(self._sampling_frequency)

        return self._sampling_frequency

    def get_num_rois(self) -> int:
        """Get total number of Regions of Interest (ROIs) in the acquired images.

        Returns
        -------
        num_rois: int
            The number of ROIs extracted.
        """
        # Check ROI responses first
        if self._roi_responses:
            for response in self._roi_responses:
                return len(response.roi_ids)

        # Check ROI representations
        if self._roi_representations:
            # Count cell ROIs (exclude background ROIs)
            cell_rois = [rid for rid in self._roi_representations.keys() if not str(rid).startswith("background")]
            return len(cell_rois)

        # If no data is available, return 0
        return 0

    def get_num_background_components(self) -> int:
        """Get total number of background components in the acquired images.

        Returns
        -------
        num_background_components: int
            The number of background components extracted.
        """
        # Count background ROI representations
        background_rois = [rid for rid in self._roi_representations.keys() if str(rid).startswith("background")]
        if background_rois:
            return len(background_rois)

        # Check neuropil responses
        for response in self._roi_responses:
            if response.response_type == "neuropil":
                return len(response.roi_ids)

        return 0

    def get_channel_names(self) -> List[str]:
        """Get names of channels in the pipeline.

        Returns
        -------
        _channel_names: list
            names of channels (str)
        """
        return self._channel_names

    def get_num_channels(self) -> int:
        """Get number of channels in the pipeline.

        Returns
        -------
        num_of_channels: int
            number of channels
        """
        return len(self._channel_names)

    def get_num_planes(self) -> int:
        """Get the default number of planes of imaging for the segmentation extractor.

        Notes
        -----
        Defaults to 1 for all but the MultiSegmentationExtractor.

        Returns
        -------
        self._num_planes: int
            number of planes
        """
        return self._num_planes

    def set_times(self, times: ArrayType):
        """Set the recording times in seconds for each frame.

        Parameters
        ----------
        times: array-like
            The times in seconds for each frame

        Notes
        -----
        Operates on _times attribute of the SegmentationExtractor object.
        """
        assert len(times) == self.get_num_samples(), "'times' should have the same length of the number of samples!"
        self._times = np.array(times, dtype=np.float64)

    def has_time_vector(self) -> bool:
        """Detect if the SegmentationExtractor has a time vector set or not.

        Returns
        -------
        has_time_vector: bool
            True if the SegmentationExtractor has a time vector set, otherwise False.
        """
        return self._times is not None

    def frame_to_time(self, frames: Union[IntType, ArrayType]) -> Union[FloatType, ArrayType]:
        """Get the timing of frames in unit of seconds.

        Parameters
        ----------
        frames: int or array-like
            The frame or frames to be converted to times

        Returns
        -------
        times: float or array-like
            The corresponding times in seconds

        Deprecated
        ----------
        This method will be removed on or after January 2026.
        Use sample_indices_to_time() instead.
        """
        warnings.warn(
            "frame_to_time() is deprecated and will be removed on or after January 2026. "
            "Use sample_indices_to_time() instead.",
            FutureWarning,
            stacklevel=2,
        )
        if self._times is None:
            return frames / self.get_sampling_frequency()
        else:
            return self._times[frames]

    def get_timestamps(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
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

    def sample_indices_to_time(self, sample_indices: Union[FloatType, np.ndarray]) -> Union[FloatType, np.ndarray]:
        """Convert user-inputted sample indices to times with units of seconds.

        .. deprecated:: on or after January 2026
           Use :meth:`get_timestamps` instead.

        Parameters
        ----------
        sample_indices: int or array-like
            The sample indices to be converted to times.

        Returns
        -------
        times: float or array-like
            The corresponding times in seconds.
        """
        warnings.warn(
            "sample_indices_to_time is deprecated and will be removed on or after January 2026. "
            "Use get_timestamps instead.",
            FutureWarning,
            stacklevel=2,
        )

        # Handle scalar vs array input
        is_scalar = np.isscalar(sample_indices)
        if is_scalar:
            sample_indices = np.array([sample_indices])
        else:
            sample_indices = np.asarray(sample_indices)

        # Get the range of samples we need
        start_sample = int(np.min(sample_indices))
        end_sample = int(np.max(sample_indices)) + 1

        # Get timestamps for the range
        timestamps = self.get_timestamps(start_sample=start_sample, end_sample=end_sample)

        # Map the requested indices to the timestamps
        result = timestamps[sample_indices - start_sample]

        return result[0] if is_scalar else result

    def set_property(self, key: str, values: ArrayType, ids: ArrayType):
        """Set property values for ROIs.

        Parameters
        ----------
        key: str
            The name of the property.
        values: array-like
            Array of property values. Must have same length as ids and num_rois.
        ids: array-like
            Array of ROI ids corresponding to the values. Must have same length as values and num_rois.
        """
        values = np.asarray(values)
        ids = list(ids)
        num_rois = self.get_num_rois()

        # Check that all arrays have the correct length
        if len(values) != num_rois or len(ids) != num_rois:
            raise ValueError(
                f"Length of values ({len(values)}) and ids ({len(ids)}) must match number of ROIs ({num_rois})"
            )

        # Verify that the provided ids match the extractor's ROI ids
        extractor_roi_ids = self.get_roi_ids()
        if set(ids) != set(extractor_roi_ids):
            raise ValueError("Provided ids must match the extractor's ROI ids")

        # Create property array with values in the correct order
        property_array = np.empty(num_rois, dtype=values.dtype)
        for roi_index, roi_id in enumerate(extractor_roi_ids):
            id_index = ids.index(roi_id)
            property_array[roi_index] = values[id_index]

        self._properties[key] = property_array

    def get_property(self, key: str, ids: ArrayType) -> ArrayType:
        """Get property values for ROIs.

        Parameters
        ----------
        key: str
            The name of the property.
        ids: array-like
            Array of ROI ids to get property values for.

        Returns
        -------
        values: array-like
            Array of property values for the specified ROIs.
        """
        ids = np.asarray(ids)
        if key not in self._properties:
            available_keys = list(self._properties.keys())
            raise KeyError(f"Property '{key}' not found. Available properties: {available_keys}")

        # Check that all requested ROI ids exist in extractor
        all_roi_ids = self.get_roi_ids()
        for roi_id in ids:
            if roi_id not in all_roi_ids:
                raise ValueError(f"ROI id {roi_id} not found in extractor. Available ROI ids: {all_roi_ids}")

        # Map ids to indices and get values
        values = []
        for roi_id in ids:
            roi_index = all_roi_ids.index(roi_id)
            values.append(self._properties[key][roi_index])

        return np.array(values)

    def get_property_keys(self) -> list[str]:
        """Get list of available property keys.

        Returns
        -------
        keys: list
            List of property names.
        """
        return list(self._properties.keys())


class SampleSlicedSegmentationExtractor(SegmentationExtractor):
    """Class to get a lazy sample slice.

    Do not use this class directly but use `.slice_samples(...)` on a SegmentationExtractor object.
    """

    extractor_name = "SampleSlicedSegmentationExtractor"

    def __init__(
        self,
        parent_segmentation: SegmentationExtractor,
        start_sample: Optional[int] = None,
        end_sample: Optional[int] = None,
    ):
        """Initialize a SegmentationExtractor whose samples subset the parent.

        Subset is exclusive on the right bound, that is, the indexes of this SegmentationExtractor range over
        [0, ..., end_sample-start_sample-1], which is used to resolve the index mapping in `get_traces(...)`.

        Parameters
        ----------
        parent_segmentation : SegmentationExtractor
            The SegmentationExtractor object to subset the samples of.
        start_sample : int, optional
            The left bound of the samples to subset.
            The default is the start sample of the parent.
        end_sample : int, optional
            The right bound of the samples, exclusively, to subset.
            The default is end sample of the parent.
        """
        self._parent_segmentation = parent_segmentation
        self._start_sample = start_sample
        self._end_sample = end_sample
        self._num_samples = self._end_sample - self._start_sample

        parent_size = self._parent_segmentation.get_num_samples()
        if start_sample is None:
            start_sample = 0
        else:
            assert 0 <= start_sample < parent_size
        if end_sample is None:
            end_sample = parent_size
        else:
            assert 0 < end_sample <= parent_size
        assert end_sample > start_sample, "'start_sample' must be smaller than 'end_sample'!"

        # Copy image masks if they exist
        if hasattr(self._parent_segmentation, "_image_masks"):
            self._image_masks = self._parent_segmentation._image_masks

        if hasattr(self._parent_segmentation, "_background_image_masks"):
            self._background_image_masks = self._parent_segmentation._background_image_masks

        super().__init__()

        # Inherit data model attributes from parent
        self._roi_ids = self._parent_segmentation._roi_ids

        # Copy ROI responses but slice their data appropriately
        self._roi_responses = []
        for roi_response in self._parent_segmentation._roi_responses:
            # Slice the trace data for this time window
            sliced_data = roi_response.data[start_sample:end_sample, :]
            sliced_response = RoiResponse(roi_response.response_type, sliced_data, roi_response.roi_ids)
            self._roi_responses.append(sliced_response)

        # Copy ROI representations (spatial masks don't change with time slicing)
        self._roi_representations = self._parent_segmentation._roi_representations.copy()

        # Copy images (spatial images don't change with time slicing)
        self._summary_images = self._parent_segmentation._summary_images.copy()
        # Preserve parent's channel names and other attributes
        self._channel_names = self._parent_segmentation.get_channel_names()
        self._num_planes = self._parent_segmentation.get_num_planes()
        if getattr(self._parent_segmentation, "_times") is not None:
            self._times = self._parent_segmentation._times[start_sample:end_sample]

    def get_accepted_list(self) -> list:
        return self._parent_segmentation.get_accepted_list()

    def get_rejected_list(self) -> list:
        return self._parent_segmentation.get_rejected_list()

    def get_frame_shape(self) -> Tuple[int, int]:
        return tuple(self._parent_segmentation.get_frame_shape())

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_num_frames(self) -> int:
        """Get the number of frames in the recording (duration of recording).

        .. deprecated:: on or after January 2026
           Use :meth:`get_num_samples` instead.

        Returns
        -------
        num_frames: int
            Number of frames in the recording.
        """
        warnings.warn(
            "get_num_frames is deprecated and will be removed on or after January 2026. "
            "Use get_num_samples instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_num_samples()

    def get_traces(
        self,
        roi_ids: Optional[Iterable[int]] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        name: str = "raw",
    ) -> np.ndarray:
        assert start_frame is None or start_frame >= 0, (
            f"'start_frame' must be greater than or equal to zero! Received '{start_frame}'.\n"
            "Negative slicing semantics are not supported."
        )

        # If no start_frame/end_frame specified, return the full sliced range
        if start_frame is None and end_frame is None:
            start_frame_shifted = self._start_sample
            end_frame_shifted = self._end_sample
        else:
            # If start_frame/end_frame are specified, they are relative to the sliced range
            start_frame_shifted = (start_frame or 0) + self._start_sample
            end_frame_shifted = end_frame
            if end_frame is not None:
                end_frame_shifted = end_frame + self._start_sample
            else:
                end_frame_shifted = self._end_sample

        # Check if the trace exists in the parent
        parent_traces_dict = self._parent_segmentation.get_traces_dict()
        if parent_traces_dict.get(name) is None:
            return None

        return self._parent_segmentation.get_traces(
            roi_ids=roi_ids,
            start_frame=start_frame_shifted,
            end_frame=end_frame_shifted,
            name=name,
        )

    def get_traces_dict(self) -> dict:
        result = {}
        for trace_name, trace_data in self._parent_segmentation.get_traces_dict().items():
            if trace_data is not None:
                result[trace_name] = self._parent_segmentation.get_traces(
                    start_frame=self._start_sample, end_frame=self._end_sample, name=trace_name
                )
            else:
                result[trace_name] = None
        return result

    def get_num_rois(self) -> int:
        return self._parent_segmentation.get_num_rois()

    def get_num_background_components(self) -> int:
        return self._parent_segmentation.get_num_background_components()

    def get_images_dict(self) -> dict:
        return self._parent_segmentation.get_images_dict()

    def get_image(self, name: str = "correlation") -> ArrayType:
        return self._parent_segmentation.get_image(name=name)

    def get_sampling_frequency(self) -> float:
        return self._parent_segmentation.get_sampling_frequency()

    def get_channel_names(self) -> list[str]:
        return self._parent_segmentation.get_channel_names()

    def get_num_channels(self) -> int:
        return self._parent_segmentation.get_num_channels()

    def get_num_planes(self) -> int:
        return self._parent_segmentation.get_num_planes()

    def get_roi_pixel_masks(self, roi_ids: Optional[ArrayLike] = None) -> list[np.ndarray]:
        return self._parent_segmentation.get_roi_pixel_masks(roi_ids=roi_ids)

    def get_background_pixel_masks(self, background_ids: Optional[ArrayLike] = None) -> list[np.ndarray]:
        return self._parent_segmentation.get_background_pixel_masks(background_ids=background_ids)

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # Adjust the sample indices to account for the slice offset
        start_sample = start_sample or 0
        end_sample = end_sample or self.get_num_samples()

        # Map slice-relative indices to parent indices
        parent_start = self._start_sample + start_sample
        parent_end = self._start_sample + end_sample

        return self._parent_segmentation.get_native_timestamps(start_sample=parent_start, end_sample=parent_end)


class FrameSliceSegmentationExtractor(SampleSlicedSegmentationExtractor):
    """Class to get a lazy frame slice.

    Do not use this class directly but use `.frame_slice(...)` on a SegmentationExtractor object.

    Deprecated
    ----------
    This class will be removed on or after January 2026.
    Use SampleSlicedSegmentationExtractor instead.
    """

    extractor_name = "FrameSliceSegmentationExtractor"

    def __init__(
        self,
        parent_segmentation: SegmentationExtractor,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ):
        """Create a new FrameSliceSegmentationExtractor from parent SegmentationExtractor.

        Parameters
        ----------
        parent_segmentation: SegmentationExtractor
            The parent SegmentationExtractor object.
        start_frame: int, optional
            The starting frame of the new SegmentationExtractor.
        end_frame: int, optional
            The ending frame of the new SegmentationExtractor.

        Deprecated
        ----------
        This class will be removed on or after January 2026.
        Use SampleSlicedSegmentationExtractor instead.
        """
        warnings.warn(
            "FrameSliceSegmentationExtractor is deprecated and will be removed on or after January 2026. "
            "Use SampleSlicedSegmentationExtractor instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(parent_segmentation=parent_segmentation, start_sample=start_frame, end_sample=end_frame)
