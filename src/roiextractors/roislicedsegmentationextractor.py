"""Module for _RoiSlicedSegmentationExtractor class."""

import numpy as np

from .segmentationextractor import (
    ArrayType,
    SegmentationExtractor,
    _ROIMasks,
    _RoiResponse,
)


class _RoiSlicedSegmentationExtractor(SegmentationExtractor):
    """Class to get a lazy ROI subset.

    Do not use this class directly but use `.select_rois(...)` on a SegmentationExtractor object.
    """

    extractor_name = "_RoiSlicedSegmentationExtractor"

    def __init__(
        self,
        parent_segmentation: SegmentationExtractor,
        roi_ids: list[str | int],
    ):
        """Initialize a SegmentationExtractor with a subset of ROIs.

        Parameters
        ----------
        parent_segmentation : SegmentationExtractor
            The SegmentationExtractor object to subset.
        roi_ids : list[str | int]
            List of ROI IDs to include. Order is preserved.
        """
        self._parent_segmentation = parent_segmentation

        # Separate cell and background IDs while preserving order
        parent_cell_ids = set(parent_segmentation.get_roi_ids())
        parent_background_ids = set(parent_segmentation.get_background_ids())

        self._selected_cell_ids = [rid for rid in roi_ids if rid in parent_cell_ids]
        self._selected_background_ids = [rid for rid in roi_ids if rid in parent_background_ids]
        self._all_selected_ids = list(roi_ids)  # Preserve original order

        super().__init__()

        # Set up filtered _roi_masks if parent has them
        if hasattr(self._parent_segmentation, "_roi_masks") and self._parent_segmentation._roi_masks is not None:
            parent_masks = self._parent_segmentation._roi_masks
            # Create filtered roi_id_map for all selected IDs (cells + background)
            filtered_roi_id_map = {
                roi_id: parent_masks.roi_id_map[roi_id]
                for roi_id in self._all_selected_ids
                if roi_id in parent_masks.roi_id_map
            }
            self._roi_masks = _ROIMasks(
                data=parent_masks.data,
                mask_tpe=parent_masks.mask_tpe,
                field_of_view_shape=parent_masks.field_of_view_shape,
                roi_id_map=filtered_roi_id_map,
            )

        # Store cell ROI IDs for get_roi_ids()
        self._roi_ids = self._selected_cell_ids

        # Create filtered _roi_responses for compatibility with slice_samples()
        # Each _RoiResponse stores traces with shape (num_samples, num_rois)
        # We need to slice the columns to only include selected cell ROIs
        for roi_response in self._parent_segmentation._roi_responses:
            # Build index mapping for selected cell ROIs
            parent_roi_ids = list(roi_response.roi_ids)
            selected_indices = []
            selected_ids = []
            for roi_id in self._selected_cell_ids:
                if roi_id in parent_roi_ids:
                    selected_indices.append(parent_roi_ids.index(roi_id))
                    selected_ids.append(roi_id)

            if selected_indices:
                # Slice the data to only include selected ROIs
                sliced_data = roi_response.data[:, selected_indices]
                self._roi_responses.append(_RoiResponse(roi_response.response_type, sliced_data, selected_ids))

        # Copy parent's summary images reference (not affected by ROI slicing)
        self._summary_images = self._parent_segmentation._summary_images

        # Copy parent's time settings if present
        if getattr(self._parent_segmentation, "_times", None) is not None:
            self._times = self._parent_segmentation._times

        # Inherit other attributes directly
        self._num_planes = self._parent_segmentation._num_planes
        self._sampling_frequency = self._parent_segmentation._sampling_frequency

        # Properties use the same copy-on-write pattern as _times above.
        # The shallow dict copy shares the underlying _PropertyInfo instances with the parent
        # (memory efficient), but set_property() always creates a new _PropertyInfo and rebinds
        # the dict key, so writes only affect this instance.
        self._properties = dict(self._parent_segmentation._properties)

    # --- Core ROI Methods ---

    def get_roi_ids(self) -> list:
        return list(self._selected_cell_ids)

    def get_num_rois(self) -> int:
        return len(self._selected_cell_ids)

    def get_background_ids(self) -> list:
        return list(self._selected_background_ids)

    def get_num_background_components(self) -> int:
        return len(self._selected_background_ids)

    # --- Spatial/Temporal Methods (Delegate) ---
    # Note: get_traces() and get_traces_dict() are inherited from base class
    # and use self._roi_responses which we populated with filtered data

    def get_frame_shape(self) -> tuple[int, int]:
        return tuple(self._parent_segmentation.get_frame_shape())

    def get_num_samples(self) -> int:
        return self._parent_segmentation.get_num_samples()

    def get_sampling_frequency(self) -> float:
        return self._parent_segmentation.get_sampling_frequency()

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        return self._parent_segmentation.get_native_timestamps(start_sample=start_sample, end_sample=end_sample)

    def has_time_vector(self) -> bool:
        return self._parent_segmentation.has_time_vector()

    def get_images_dict(self) -> dict:
        return self._parent_segmentation.get_images_dict()

    def get_image(self, name: str = "correlation") -> ArrayType:
        return self._parent_segmentation.get_image(name=name)

    def get_num_planes(self) -> int:
        return self._parent_segmentation.get_num_planes()

    # Note: get_property(), set_property(), get_property_info(), and get_property_keys()
    # are inherited from the base class and operate on the copied _properties and
    # _property_descriptions dicts. See the copy-on-write comment in __init__.
