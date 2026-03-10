"""Base segmentation extractors.

Classes
-------
SegmentationExtractor
    Abstract class that contains all the meta-data and output data from the ROI segmentation operation when applied to
    the pre-processed data. It also contains methods to read from various data formats output from the
    processing pipelines like SIMA, CaImAn, Suite2p, CNMF-E.
SampleSlicedSegmentationExtractor
    Class to get a lazy sample slice.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .extraction_tools import ArrayType


# TODO make public once API stabilizes.
@dataclass
class _RoiResponse:
    """Represents a fluorescence response (trace) with its metadata."""

    response_type: str
    data: ArrayLike  # Shape: (num_samples, num_rois)
    roi_ids: list[str | int]


class _ROIMasks:
    """Internal container for all ROI spatial representations in native NWB-compatible format.

    Stores all ROI masks (cells + background/neuropil) together with their ID mapping.
    The representation format matches NWB standards for efficient reading/writing.

    Note: This is a private class. Users should access ROI masks through SegmentationExtractor
    methods like get_roi_image_masks() and get_roi_pixel_masks().

    Attributes
    ----------
    data : ArrayLike
        Native format data:
        - "nwb-image_mask": (height, width, n_rois) dense array, possibly lazy (DatasetView/h5py.Dataset)
        - "nwb-pixel_mask": list of (n_pixels, 3) arrays with columns [y, x, weight]
        - "nwb-voxel_mask": list of (n_voxels, 4) arrays with columns [y, x, z, weight]
    mask_tpe : Literal["nwb-image_mask", "nwb-pixel_mask", "nwb-voxel_mask"]
        Type of NWB-compatible representation.
    field_of_view_shape : tuple[int, ...]
        Shape of imaging FOV: (height, width) for 2D or (depth, height, width) for 3D.
    roi_id_map : dict[str | int, int]
        Maps ROI ID -> index in data structure.
        - For dense: roi_id -> slice index along last axis
        - For sparse lists: roi_id -> list index
        Examples: {0: 0, 1: 1, "background0": 2, "background1": 3}
    """

    def __init__(
        self,
        data: ArrayLike,
        mask_tpe: Literal["nwb-image_mask", "nwb-pixel_mask", "nwb-voxel_mask"],
        field_of_view_shape: tuple[int, ...],
        roi_id_map: dict[str | int, int],
    ):
        """Initialize ROI representations container.

        Parameters
        ----------
        data : ArrayLike
            ROI mask data in native format.
        mask_tpe : Literal["nwb-image_mask", "nwb-pixel_mask", "nwb-voxel_mask"]
            Format type following NWB conventions.
        field_of_view_shape : tuple[int, ...]
            Shape of the imaging field of view (height, width) or (depth, height, width).
        roi_id_map : dict[str | int, int]
            Mapping from ROI ID to index in data structure.
        """
        self.data = data
        self.mask_tpe = mask_tpe
        self.field_of_view_shape = field_of_view_shape
        self.roi_id_map = roi_id_map

    @property
    def is_volumetric(self) -> bool:
        """True if this is 3D volumetric data, False for 2D."""
        return len(self.field_of_view_shape) == 3

    @property
    def num_rois(self) -> int:
        """Total number of ROIs in this container."""
        return len(self.roi_id_map)

    def get_roi_ids(self) -> list[str | int]:
        """Get all ROI IDs in this container.

        Returns
        -------
        list[str | int]
            List of all ROI IDs (cells + background).
        """
        return list(self.roi_id_map.keys())

    def get_roi_image_mask(self, roi_id: str | int) -> np.ndarray:
        """Get dense image mask for a single ROI.

        Parameters
        ----------
        roi_id : str | int
            The ROI identifier.

        Returns
        -------
        np.ndarray
            Dense 2D or 3D array matching field_of_view_shape.
        """
        index = self.roi_id_map[roi_id]

        if self.mask_tpe == "nwb-image_mask":
            # Extract slice from dense stack
            return np.asarray(self.data[:, :, index])

        elif self.mask_tpe == "nwb-pixel_mask":
            # Convert sparse pixel list to dense
            dense_mask = np.zeros(self.field_of_view_shape, dtype=np.float32)
            pixel_data = self.data[index]  # (n_pixels, 3): [y, x, weight]
            if len(pixel_data) > 0:
                y_coords = pixel_data[:, 0].astype(int)
                x_coords = pixel_data[:, 1].astype(int)
                weights = pixel_data[:, 2]
                dense_mask[y_coords, x_coords] = weights
            return dense_mask

        elif self.mask_tpe == "nwb-voxel_mask":
            # Convert sparse voxel list to dense 3D
            dense_mask = np.zeros(self.field_of_view_shape, dtype=np.float32)
            voxel_data = self.data[index]  # (n_voxels, 4): [y, x, z, weight]
            if len(voxel_data) > 0:
                y_coords = voxel_data[:, 0].astype(int)
                x_coords = voxel_data[:, 1].astype(int)
                z_coords = voxel_data[:, 2].astype(int)
                weights = voxel_data[:, 3]
                dense_mask[y_coords, x_coords, z_coords] = weights
            return dense_mask

    def get_roi_pixel_mask(self, roi_id: str | int) -> np.ndarray:
        """Get sparse pixel mask for a single ROI.

        Parameters
        ----------
        roi_id : str | int
            The ROI identifier.

        Returns
        -------
        np.ndarray
            Array with shape (n_pixels, 3) with columns [y, x, weight].
            For 3D: (n_voxels, 4) with columns [y, x, z, weight].
        """
        index = self.roi_id_map[roi_id]

        if self.mask_tpe == "nwb-pixel_mask":
            return np.asarray(self.data[index])

        elif self.mask_tpe == "nwb-voxel_mask":
            return np.asarray(self.data[index])

        else:
            # Convert dense to sparse
            dense_mask = self.get_roi_image_mask(roi_id)
            if self.is_volumetric:
                # 3D case
                y_coords, x_coords, z_coords = np.nonzero(dense_mask)
                weights = dense_mask[y_coords, x_coords, z_coords]
                return np.column_stack([y_coords, x_coords, z_coords, weights])
            else:
                # 2D case
                y_coords, x_coords = np.nonzero(dense_mask)
                weights = dense_mask[y_coords, x_coords]
                return np.column_stack([y_coords, x_coords, weights])


class SegmentationExtractor(ABC):
    """Abstract segmentation extractor class.

    An abstract class that contains all the meta-data and output data from
    the ROI segmentation operation when applied to the pre-processed data.
    It also contains methods to read from various data formats output from the
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
        self._roi_ids: list[str | int] | None = None
        self._roi_responses: list[_RoiResponse] = []
        self._summary_images = {}
        self._roi_masks: _ROIMasks | None = None
        self._properties = {}

    def get_accepted_list(self) -> list:
        """Get a list of accepted ROI ids.

        .. deprecated::
            `get_accepted_list` is deprecated and will be removed in May 2026.
            Use `get_property()` instead to access format-specific acceptance data.

        Returns
        -------
        accepted_list: list
            List of accepted ROI ids.
        """
        warnings.warn(
            "get_accepted_list is deprecated and will be removed in May 2026. "
            "Use get_property() instead to access format-specific acceptance data.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Default: all ROIs accepted
        return list(self.get_roi_ids())

    def get_rejected_list(self) -> list:
        """Get a list of rejected ROI ids.

        .. deprecated::
            `get_rejected_list` is deprecated and will be removed in May 2026.
            Use `get_property()` instead to access format-specific acceptance data.

        Returns
        -------
        rejected_list: list
            List of rejected ROI ids.
        """
        warnings.warn(
            "get_rejected_list is deprecated and will be removed in May 2026. "
            "Use get_property() instead to access format-specific acceptance data.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Default: no ROIs rejected
        return []

    @abstractmethod
    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
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
        return None

    @abstractmethod
    def get_frame_shape(self) -> ArrayType:
        """Get frame size of movie (height, width).

        Returns
        -------
        frame_shape: array_like
            2-D array: image height x image width
        """
        pass

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

    def get_roi_locations(self, roi_ids=None) -> np.ndarray:
        """Get the locations of the Regions of Interest (ROIs).

        .. deprecated::
            `get_roi_locations` is deprecated and will be removed in or after September 2026.
            Use `get_property("roi_centroids", roi_ids)` instead for centroid data
            stored as a property.

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs requested.

        Returns
        -------
        roi_locs: numpy.ndarray
            2-D array: 2 X no_ROIs. The pixel ids (x,y) where the centroid of the ROI is.
        """
        warnings.warn(
            "get_roi_locations is deprecated and will be removed in or after September 2026. "
            "Use get_property('roi_centroids', roi_ids) instead.",
            FutureWarning,
            stacklevel=2,
        )
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
        if self._roi_ids is not None:
            return self._roi_ids

        # For backward compatibility, only return cell ROIs (exclude background components)
        if self._roi_masks is not None:
            all_roi_ids = self._roi_masks.get_roi_ids()
            cell_roi_ids = [rid for rid in all_roi_ids if not str(rid).startswith("background")]
            return cell_roi_ids

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
        if roi_ids is None:
            roi_ids = self.get_roi_ids()

        if self._roi_masks is None:
            # Fallback for extractors that haven't migrated yet
            raise NotImplementedError("This extractor has not been updated to use the new ROI representation system.")

        # Filter to only cell ROIs (exclude background)
        cell_roi_ids = [rid for rid in roi_ids if not str(rid).startswith("background")]

        if len(cell_roi_ids) == 0:
            frame_shape = self.get_frame_shape()
            return np.zeros((*frame_shape, 0))

        # Get masks from representations
        masks = []
        for roi_id in cell_roi_ids:
            mask = self._roi_masks.get_roi_image_mask(roi_id)
            masks.append(mask)

        return np.stack(masks, axis=2)

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
            roi_ids = self.get_roi_ids()

        if self._roi_masks is None:
            # Fallback for extractors that haven't migrated yet
            raise NotImplementedError("This extractor has not been updated to use the new ROI representation system.")

        # Filter to only cell ROIs (exclude background)
        cell_roi_ids = [rid for rid in roi_ids if not str(rid).startswith("background")]

        # Get pixel masks from representations
        pixel_masks = []
        for roi_id in cell_roi_ids:
            pixel_mask = self._roi_masks.get_roi_pixel_mask(roi_id)
            pixel_masks.append(pixel_mask)

        return pixel_masks

    def get_background_ids(self) -> list:
        """Get the list of background components ids.

        Returns
        -------
        background_components_ids: list
            List of background components ids.
        """
        if self._roi_masks is None:
            return list(range(self.get_num_background_components()))

        # Extract background IDs from roi_masks
        all_roi_ids = self._roi_masks.get_roi_ids()
        background_ids = [rid for rid in all_roi_ids if str(rid).startswith("background")]
        return background_ids

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
        if background_ids is None:
            background_ids = self.get_background_ids()

        if self._roi_masks is None:
            # Fallback for extractors that haven't migrated yet
            return np.zeros((*self.get_frame_shape(), 0))

        if len(background_ids) == 0:
            frame_shape = self.get_frame_shape()
            return np.zeros((*frame_shape, 0))

        # Get masks from representations
        masks = []
        for bg_id in background_ids:
            mask = self._roi_masks.get_roi_image_mask(bg_id)
            masks.append(mask)

        return np.stack(masks, axis=2)

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
            background_ids = self.get_background_ids()

        if self._roi_masks is None:
            # Fallback for extractors that haven't migrated yet
            return []

        # Get pixel masks from representations
        pixel_masks = []
        for bg_id in background_ids:
            pixel_mask = self._roi_masks.get_roi_pixel_mask(bg_id)
            pixel_masks.append(pixel_mask)

        return pixel_masks

    def slice_samples(self, start_sample: int | None = None, end_sample: int | None = None):
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

    def select_rois(self, roi_ids: list[str | int]):
        """Return a new SegmentationExtractor with only the specified ROIs.

        Parameters
        ----------
        roi_ids : list[str | int]
            List of ROI IDs to include. Can include both cell and background ROI IDs.
            The order of IDs is preserved in the returned extractor.

        Returns
        -------
        segmentation : RoiSlicedSegmentationExtractor
            The ROI-sliced SegmentationExtractor object.

        Raises
        ------
        ValueError
            If roi_ids is empty or contains IDs not present in the extractor.

        Notes
        -----
        This method creates a lazy view of the segmentation data with a subset of ROIs.
        The slicing is applied to ROI-related data while temporal and spatial properties
        are preserved.

        Examples
        --------
        >>> # Select specific ROIs
        >>> subset = extractor.select_rois([0, 1, 2])
        >>>
        >>> # Compose with temporal slicing
        >>> subset = extractor.select_rois([0, 1, 2]).slice_samples(100, 200)
        """
        if not roi_ids:
            raise ValueError("roi_ids cannot be empty")

        all_valid_ids = set(self.get_roi_ids()) | set(self.get_background_ids())
        invalid_ids = [rid for rid in roi_ids if rid not in all_valid_ids]
        if invalid_ids:
            raise ValueError(
                f"ROI ids {invalid_ids} not found in extractor. "
                f"Available cell ROI ids: {self.get_roi_ids()}, "
                f"Available background ids: {self.get_background_ids()}"
            )

        from .roislicedsegmentationextractor import _RoiSlicedSegmentationExtractor

        return _RoiSlicedSegmentationExtractor(parent_segmentation=self, roi_ids=roi_ids)

    def get_traces(
        self,
        roi_ids: list[int | str] = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
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
        traces_dict = self.get_traces_dict()
        if traces_dict.get(name) is None:
            return None

        response = next((r for r in self._roi_responses if r.response_type == name), None)
        if response is None:
            raise ValueError(
                f"Traces for {name} are registered in the trace dictionary but missing from the internal store."
            )

        data = np.asarray(response.data)
        sliced = data[start_frame:end_frame, :]

        input_roi_ids = roi_ids
        if input_roi_ids is None:
            return np.array(sliced)

        # Match ROI ids by value, allowing for differing orders between sources
        response_roi_ids = list(response.roi_ids)
        indices: list[int] = []
        missing_roi_ids: list = []
        for roi_id in input_roi_ids:
            try:
                indices.append(response_roi_ids.index(roi_id))
            except ValueError:
                missing_roi_ids.append(roi_id)

        if missing_roi_ids:
            raise ValueError(
                f"ROI ids {missing_roi_ids} not found for response '{name}'. Available ids: {response_roi_ids}"
            )

        return np.array(sliced[:, indices])

    def get_traces_dict(self) -> dict:
        """Get traces as a dictionary with key as the name of the ROiResponseSeries.

        Returns
        -------
        _roi_response_dict: dict
            dictionary with key, values representing different types of RoiResponseSeries:
                Raw Fluorescence, DeltaFOverF, Denoised, Neuropil, Deconvolved, Background, etc.
        """
        traces = {response.response_type: response.data for response in self._roi_responses}
        for expected_type in ("raw", "dff", "neuropil", "deconvolved", "denoised", "baseline", "background"):
            traces.setdefault(expected_type, None)
        return traces

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
        if self._roi_masks is not None:
            # Count only cell ROIs (exclude background)
            all_roi_ids = self._roi_masks.get_roi_ids()
            cell_roi_ids = [rid for rid in all_roi_ids if not str(rid).startswith("background")]
            return len(cell_roi_ids)

        # Fallback to trace-based counting
        for trace in self.get_traces_dict().values():
            if trace is not None and len(trace.shape) > 0:
                return trace.shape[1]

        return 0

    def get_num_background_components(self) -> int:
        """Get total number of background components in the acquired images.

        Returns
        -------
        num_background_components: int
            The number of background components extracted.
        """
        if self._roi_masks is not None:
            # Count background ROIs from representations
            all_roi_ids = self._roi_masks.get_roi_ids()
            background_ids = [rid for rid in all_roi_ids if str(rid).startswith("background")]
            return len(background_ids)

        # Fallback to response-based counting
        for response in self._roi_responses:
            if response.response_type in {"neuropil", "background"}:
                data = response.data
                if data is None:
                    continue
                if not hasattr(data, "shape"):
                    continue
                if len(data.shape) == 1:
                    return int(data.shape[0])
                return int(data.shape[1])

        return 0

    def get_channel_names(self) -> list[str]:
        """Get names of channels in the pipeline.

        Returns
        -------
        _channel_names: list
            names of channels (str)
        """
        warnings.warn(
            "get_channel_names is deprecated and will be removed in May 2026 or after.",
            category=FutureWarning,
            stacklevel=2,
        )
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
        property_array = np.empty(values.shape, dtype=values.dtype)
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
        start_sample: int | None = None,
        end_sample: int | None = None,
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

        self._start_sample = start_sample
        self._end_sample = end_sample
        self._num_samples = self._end_sample - self._start_sample

        super().__init__()

        # Share the parent's ROI representations (spatial data is same, only temporal is sliced)
        if hasattr(self._parent_segmentation, "_roi_masks"):
            self._roi_masks = self._parent_segmentation._roi_masks

        self._roi_ids = list(self._parent_segmentation.get_roi_ids())
        for roi_response in self._parent_segmentation._roi_responses:
            sliced_data = roi_response.data[start_sample:end_sample, :]
            self._roi_responses.append(
                _RoiResponse(roi_response.response_type, sliced_data, list(roi_response.roi_ids))
            )
        self._summary_images = dict(self._parent_segmentation.get_images_dict())
        # Preserve parent's channel names and other attributes (access attribute directly to avoid deprecation warning)
        self._channel_names = self._parent_segmentation._channel_names
        self._num_planes = self._parent_segmentation.get_num_planes()

        # The _times attribute of the sliced extractor acts like a view to the parent's _times,
        # which is memory efficient. However, it maintains copy semantics which are safer for the following reasons:
        # Currently, there are only two ways of setting the _times:
        #
        # 1. set_times() method - always overwrites the entire _times array
        # 2. get_timestamps() method - in some cases will cache get_native_timestamps() output
        #
        # Both methods overwrite the entire _times array of the instance, preventing aliasing
        # problems where the _times reference of a slice extractor could be modified by the parent
        # or vice versa. See issue 498 for more details about this design.
        if getattr(self._parent_segmentation, "_times") is not None:
            self._times = self._parent_segmentation._times[start_sample:end_sample]

        # Copy properties from parent (ROI properties are not affected by temporal slicing)
        self._properties = dict(self._parent_segmentation._properties)

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        # Adjust the sample indices to account for the slice offset
        start_sample = start_sample or 0
        end_sample = end_sample or self.get_num_samples()

        # Map slice-relative indices to parent indices
        parent_start = self._start_sample + start_sample
        parent_end = self._start_sample + end_sample

        return self._parent_segmentation.get_native_timestamps(start_sample=parent_start, end_sample=parent_end)

    def get_frame_shape(self) -> tuple[int, int]:
        return tuple(self._parent_segmentation.get_frame_shape())

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_roi_ids(self) -> list:
        return self._parent_segmentation.get_roi_ids()

    def get_roi_image_masks(self, roi_ids=None) -> np.ndarray:
        return self._parent_segmentation.get_roi_image_masks(roi_ids=roi_ids)

    def get_roi_pixel_masks(self, roi_ids: ArrayLike | None = None) -> list[np.ndarray]:
        return self._parent_segmentation.get_roi_pixel_masks(roi_ids=roi_ids)

    def get_background_ids(self) -> list:
        return self._parent_segmentation.get_background_ids()

    def get_background_image_masks(self, background_ids=None) -> np.ndarray:
        return self._parent_segmentation.get_background_image_masks(background_ids=background_ids)

    def get_background_pixel_masks(self, background_ids: ArrayLike | None = None) -> list[np.ndarray]:
        return self._parent_segmentation.get_background_pixel_masks(background_ids=background_ids)

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
        warnings.warn(
            "get_channel_names is deprecated and will be removed in May 2026 or after.",
            category=FutureWarning,
            stacklevel=2,
        )
        return self._parent_segmentation.get_channel_names()

    def get_num_channels(self) -> int:
        return self._parent_segmentation.get_num_channels()

    def get_num_planes(self) -> int:
        return self._parent_segmentation.get_num_planes()

    def has_time_vector(self) -> bool:
        # Override to check parent segmentation for time vector
        return self._parent_segmentation.has_time_vector()
