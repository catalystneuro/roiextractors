"""
Inscopix Segmentation Extractor
"""

import numpy as np

from ...extraction_tools import PathType, ArrayType
from ...segmentationextractor import SegmentationExtractor


class InscopixSegmentationExtractor(SegmentationExtractor):
    """A segmentation extractor for Inscopix."""

    extractor_name = "InscopixSegmentationExtractor"
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = "file"
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path: PathType):
        """Initialize a InscopixSegmentationExtractor instance.

        Parameters
        ----------
        file_path: str
            The location of the folder containing Inscopix *.mat output file.
        """
        import isx

        SegmentationExtractor.__init__(self)
        self.file_path = file_path
        self.cell_set = isx.CellSet.read(str(file_path))

    def get_num_rois(self):
        return self.cell_set.num_cells

    def get_roi_image_masks(self, roi_ids=None) -> np.ndarray:
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            all_ids = self.get_roi_ids()
            roi_idx_ = [all_ids.index(i) for i in roi_ids]
        return np.hstack([self.cell_set.get_cell_image_data(roi_id) for roi_id in roi_idx_])

    def get_roi_ids(self) -> list:
        return [self.cell_set.get_cell_name(x) for x in range(self.get_num_rois())]

    def get_image_size(self) -> ArrayType:
        num_pixels = self.cell_set.footer["spacingInfo"]["numPixels"]
        return num_pixels["x"], num_pixels["y"]

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
        return self.cell_set.footer["timingInfo"]["numTimes"]

    def get_sampling_frequency(self) -> float:
        return 1 / self.cell_set.timing.period.secs_float
