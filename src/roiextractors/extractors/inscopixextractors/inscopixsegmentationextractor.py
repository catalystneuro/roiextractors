"""Inscopix Segmentation Extractor."""

from typing import Optional
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

        Main class for extracting segmentation data from Inscopix format.

        Parameters
        ----------
        file_path: str
            The location of the folder containing Inscopix *.mat output file.
        """
        import isx

        SegmentationExtractor.__init__(self)
        self.file_path = file_path
        self.cell_set = isx.CellSet.read(file_path)

    @staticmethod
    def _get_roi_ids(cellset):
        try:
            return [cellset.get_cell_name(x) for x in range(cellset.num_cells)]
        except AttributeError:
            return np.arange(len(cellset.get_cell_indices()))

    def _load_properties(self):
        self._num_rois = self._cellset.num_cells
        self._roi_ids = self._get_roi_ids(self._cellset)
        try:
            num_pixels = self._cellset.footer["spacingInfo"]["numPixels"]
            self._image_size = (num_pixels["x"], num_pixels["y"])
        except (AttributeError, KeyError):
            first_mask = self._cellset.get_cell_mask(0)
            self._image_size = first_mask.shape if self._num_rois > 0 else None

    def get_num_rois(self):
        return self._num_rois

    def get_roi_ids(self):
        return self._roi_ids

    def get_image_size(self):
        return self._image_size

    def get_roi_image_masks(self, roi_ids=None):
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            all_ids = self.get_roi_ids()
            roi_idx_ = [all_ids.index(i) for i in roi_ids]
        return np.hstack([self._cellset.get_cell_image_data(roi_id) for roi_id in roi_idx_])

    def get_accepted_list(self):
        return [id for x, id in enumerate(self.get_roi_ids()) if self._cellset.get_cell_status(x) == "accepted"]

    def get_rejected_list(self):
        return [id for x, id in enumerate(self.get_roi_ids()) if self._cellset.get_cell_status(x) == "rejected"]

    def get_traces(self, roi_ids=None, start_frame=None, end_frame=None, name="raw"):
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            all_ids = self.get_roi_ids()
            roi_idx_ = [all_ids.index(i) for i in roi_ids]
        return np.vstack([self._cellset.get_cell_trace_data(roi_id)[start_frame:end_frame] for roi_id in roi_idx_])

    def get_num_frames(self):
        try:
            return self._cellset.footer["timingInfo"]["numTimes"]
        except (AttributeError, KeyError):
            first_trace = self._cellset.get_cell_trace_data(0)
            return len(first_trace) if first_trace is not None else 0

    def get_sampling_frequency(self):
        try:
            return 1 / self._cellset.timing.period.secs_float
        except AttributeError:
            return None

    @staticmethod
    def write_segmentation(segmentation_object, save_path):
        raise NotImplementedError("Writing to Inscopix format is not supported yet")
