"""Inscopix Segmentation Extractor."""

from typing import Optional, List, Union
import platform
import numpy as np

from ...extraction_tools import PathType, ArrayType
from ...segmentationextractor import SegmentationExtractor


class InscopixSegmentationExtractor(SegmentationExtractor):
    """A segmentation extractor for Inscopix."""

    extractor_name = "InscopixSegmentationExtractor"
    installed = True
    is_writable = False
    mode = "file"
    installation_mesg = ""

    def __init__(self, file_path: PathType):
        """Initialize a InscopixSegmentationExtractor instance.

        Main class for extracting segmentation data from Inscopix format.

        Parameters
        ----------
        file_path: str or PathType
            The location of the folder containing Inscopix *.mat output file.
        """
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            raise ImportError(
                "For macOS ARM64, please use a special conda environment setup. " "See README for instructions."
            )

        import isx

        SegmentationExtractor.__init__(self)
        self.file_path = file_path
        file_path_str = str(file_path)

        self.cell_set = isx.CellSet.read(file_path_str)

        # Create mappings between original IDs and integer IDs
        self._original_ids = [self.cell_set.get_cell_name(x) for x in range(self.cell_set.num_cells)]
        self._id_to_index = {id: i for i, id in enumerate(self._original_ids)}
        self._index_to_id = {i: id for id, i in self._id_to_index.items()}

    def get_num_rois(self) -> int:
        return self.cell_set.num_cells

    def _get_roi_indices(self, roi_ids=None) -> List[int]:
        """Convert ROI IDs to indices (positions in the original CellSet).

        Handle both string IDs (e.g., 'C0') and integer IDs (e.g., 0).
        """
        if roi_ids is None:
            return list(range(self.get_num_rois()))

        indices = []
        for roi_id in roi_ids:
            if isinstance(roi_id, int) and roi_id in self._index_to_id:
                # If it's an integer ID from our mapping
                original_id = self._index_to_id[roi_id]
                idx = self._original_ids.index(original_id)
                indices.append(idx)
            elif roi_id in self._original_ids:
                # If it's an original string ID
                idx = self._original_ids.index(roi_id)
                indices.append(idx)
            else:
                raise ValueError(f"ROI ID {roi_id} not found in segmentation data")

        return indices

    def get_roi_image_masks(self, roi_ids=None) -> np.ndarray:
        """Get image masks for the specified ROIs.

        Parameters
        ----------
        roi_ids : list or None
            List of ROI IDs (can be integers or original string IDs)

        Returns
        -------
        np.ndarray
            Image masks for the specified ROIs
        """
        roi_indices = self._get_roi_indices(roi_ids)

        masks = [self.cell_set.get_cell_image_data(roi_idx) for roi_idx in roi_indices]
        if len(masks) == 1:
            return masks[0]
        return np.stack(masks)

    def get_roi_pixel_masks(self, roi_ids=None) -> List[np.ndarray]:
        """Get pixel masks for the specified ROIs.

        This converts the image masks to pixel masks with the format expected by the NWB standard.

        Parameters
        ----------
        roi_ids : list or None
            List of ROI IDs (can be integers or original string IDs)

        Returns
        -------
        list
            List of pixel masks, each with shape (N, 3) where N is the number of pixels in the ROI.
            Each row is (x, y, weight).
        """
        # Get image masks
        image_masks = self.get_roi_image_masks(roi_ids=roi_ids)

        # Handle case when only one ROI ID is specified
        if roi_ids is not None and (not isinstance(roi_ids, list) or len(roi_ids) == 1):
            image_masks = [image_masks]

        # Convert image masks to pixel masks
        pixel_masks = []
        for mask in image_masks:
            # Find non-zero pixels in the mask
            y_indices, x_indices = np.where(mask > 0)

            if len(x_indices) > 0:
                # Use the mask values as weights
                weights = mask[y_indices, x_indices]
                # Create pixel mask with (x, y, weight) format
                pixel_mask = np.column_stack((x_indices, y_indices, weights))
            else:
                # For empty ROIs, create a dummy pixel mask with correct shape
                pixel_mask = np.array([[0, 0, 1.0]])

            pixel_masks.append(pixel_mask)

        return pixel_masks

    def get_roi_ids(self) -> list:
        """Get ROI IDs as integers (0, 1, 2, ...)."""
        return list(range(self.get_num_rois()))

    def get_original_roi_ids(self) -> list:
        """Get original ROI IDs from the CellSet."""
        return self._original_ids.copy()

    def get_image_size(self) -> ArrayType:
        if hasattr(self.cell_set, "spacing"):
            # Swap dimensions to return (width, height)
            pixels = self.cell_set.spacing.num_pixels
            return (pixels[1], pixels[0])
        else:
            if self.get_num_rois() > 0:
                shape = self.cell_set.get_cell_image_data(0).shape
                # Swap dimensions to return (width, height)
                return (shape[1], shape[0])
            raise ValueError("No ROIs found in the segmentation. Unable to determine image size.")

    def get_accepted_list(self) -> list:
        """Get list of accepted ROI IDs (as integers)."""
        accepted = []
        for i, original_id in enumerate(self._original_ids):
            idx = self._original_ids.index(original_id)
            if self.cell_set.get_cell_status(idx) == "accepted":
                accepted.append(i)  # Return integer IDs
        return accepted

    def get_rejected_list(self) -> list:
        """Get list of rejected ROI IDs (as integers)."""
        rejected = []
        for i, original_id in enumerate(self._original_ids):
            idx = self._original_ids.index(original_id)
            if self.cell_set.get_cell_status(idx) == "rejected":
                rejected.append(i)  # Return integer IDs
        return rejected

    def get_traces(self, roi_ids=None, start_frame=None, end_frame=None, name="raw") -> ArrayType:
        """Get traces for the specified ROIs.

        Parameters
        ----------
        roi_ids : list or None
            List of ROI IDs (can be integers or original string IDs)
        start_frame : int or None
            Start frame index
        end_frame : int or None
            End frame index
        name : str
            Name of the trace type

        Returns
        -------
        np.ndarray
            Traces for the specified ROIs
        """
        roi_indices = self._get_roi_indices(roi_ids)

        return np.vstack([self.cell_set.get_cell_trace_data(roi_idx)[start_frame:end_frame] for roi_idx in roi_indices])

    def get_num_frames(self) -> int:
        try:
            return self.cell_set.timing.num_samples
        except AttributeError:
            if self.get_num_rois() > 0:
                return len(self.cell_set.get_cell_trace_data(0))
            return 0

    def get_sampling_frequency(self) -> float:
        try:
            return 1 / self.cell_set.timing.period.secs_float
        except AttributeError:
            return None
