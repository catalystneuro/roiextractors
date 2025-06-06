"""Inscopix Segmentation Extractor."""

from typing import Optional
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
        file_path: str
            The location of the folder containing Inscopix *.mat output file.
        """
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            raise ImportError(
                "The isx package is currently not natively supported on macOS with Apple Silicon. "
                "Installation instructions can be found at: "
                "https://github.com/inscopix/pyisx?tab=readme-ov-file#install"
            )

        import isx

        SegmentationExtractor.__init__(self)
        self.file_path = file_path
        self.cell_set = isx.CellSet.read(file_path)

    def get_num_rois(self) -> int:
        return self.cell_set.num_cells

    def get_roi_image_masks(self, roi_ids=None) -> np.ndarray:
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            all_ids = self.get_roi_ids()
            roi_idx_ = [all_ids.index(i) for i in roi_ids]

        masks = [self.cell_set.get_cell_image_data(roi_id) for roi_id in roi_idx_]
        if len(masks) == 1:
            return masks[0]
        return np.stack(masks)

    def get_roi_ids(self) -> list:
        return [self.cell_set.get_cell_name(x) for x in range(self.get_num_rois())]

    def get_frame_shape(self) -> ArrayType:
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
        return [id for x, id in enumerate(self.get_roi_ids()) if self.cell_set.get_cell_status(x) == "accepted"]

    def get_rejected_list(self) -> list:
        return [id for x, id in enumerate(self.get_roi_ids()) if self.cell_set.get_cell_status(x) == "rejected"]

    def get_traces(self, roi_ids=None, start_frame=None, end_frame=None, name="raw") -> ArrayType:
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            all_ids = self.get_roi_ids()
            roi_idx_ = [all_ids.index(i) for i in roi_ids]
        return np.vstack([self.cell_set.get_cell_trace_data(roi_id)[start_frame:end_frame] for roi_id in roi_idx_])

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
