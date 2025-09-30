"""A segmentation extractor for CNMF-E ROI segmentation method.

Classes
-------
CnmfeSegmentationExtractor
    A segmentation extractor for CNMF-E ROI segmentation method.
"""

from warnings import warn

import h5py
import numpy as np
from lazy_ops import DatasetView

from ...extraction_tools import PathType
from ...segmentationextractor import SegmentationExtractor


class CnmfeSegmentationExtractor(SegmentationExtractor):
    """A segmentation extractor for CNMF-E ROI segmentation method.

    This class inherits from the SegmentationExtractor class, having all
    its functionality specifically applied to the dataset output from
    the 'CNMF-E' ROI segmentation method.
    """

    extractor_name = "CnmfeSegmentation"
    mode = "file"

    def __init__(self, file_path: PathType):
        """Create a CnmfeSegmentationExtractor from a .mat file.

        Parameters
        ----------
        file_path: str
            The location of the folder containing dataset.mat file.
        """
        SegmentationExtractor.__init__(self)
        self.file_path = file_path
        self._dataset_file, self._group0 = self._file_extractor_read()
        self._image_masks = self._image_mask_extractor_read()
        self._roi_response_raw = self._trace_extractor_read()
        self._raw_movie_file_location = self._raw_datafile_read()
        self._sampling_frequency = self.get_num_frames() / self._tot_exptime_extractor_read()
        # self._sampling_frequency = self._dataset_file[self._group0[0]]['inputOptions']["Fs"][...][0][0]
        correlation_image = self._summary_image_read()
        if correlation_image is not None:
            self._summary_images["correlation"] = correlation_image

    def __del__(self):
        """Close the file when the object is deleted."""
        self._dataset_file.close()

    def _file_extractor_read(self):
        """Read the .mat file and return the file object and the group.

        Returns
        -------
        f: h5py.File
            The file object.
        _group0: list
            Group of relevant segmentation objects.
        """
        f = h5py.File(self.file_path, "r")
        _group0_temp = list(f.keys())
        _group0 = [a for a in _group0_temp if "#" not in a]
        return f, _group0

    def _image_mask_extractor_read(self):
        """Read the image masks from the .mat file and return the image masks.

        Returns
        -------
        DatasetView
            The image masks.
        """
        return DatasetView(self._dataset_file[self._group0[0]]["extractedImages"]).lazy_transpose([1, 2, 0])

    def _trace_extractor_read(self):
        """Read the traces from the .mat file and return the traces.

        Returns
        -------
        DatasetView
            The traces.
        """
        return self._dataset_file[self._group0[0]]["extractedSignals"]

    def _tot_exptime_extractor_read(self):
        """Read the total experiment time from the .mat file and return the total experiment time.

        Returns
        -------
        tot_exptime: float
            The total experiment time.
        """
        return self._dataset_file[self._group0[0]]["time"]["totalTime"][0][0]

    def _summary_image_read(self):
        """Read the summary image from the .mat file and return the summary image (Cn).

        Returns
        -------
        summary_image: np.ndarray
            The summary image (Cn).
        """
        summary_image = self._dataset_file[self._group0[0]]["Cn"]
        return np.array(summary_image)

    def _raw_datafile_read(self):
        """Read the raw data file location from the .mat file and return the raw data file location.

        Returns
        -------
        raw_datafile: str
            The raw data file location.
        """
        if self._dataset_file[self._group0[0]].get("movieList"):
            charlist = [chr(i) for i in np.squeeze(self._dataset_file[self._group0[0]]["movieList"][:])]
            return "".join(charlist)

    def get_accepted_list(self):
        return list(range(self.get_num_rois()))

    def get_rejected_list(self):
        ac_set = set(self.get_accepted_list())
        return [a for a in range(self.get_num_rois()) if a not in ac_set]

    def get_frame_shape(self):
        """Get the frame shape (height, width) of the movie.

        Returns
        -------
        tuple
            The frame shape as (height, width).
        """
        return self._image_masks.shape[0:2]

    def get_image_size(self):
        warn(
            "get_image_size is deprecated and will be removed on or after January 2026. "
            "Use get_frame_shape instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_frame_shape()

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        # CNMF-E segmentation data does not have native timestamps
        return None
