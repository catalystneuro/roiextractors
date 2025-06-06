"""A SegmentationExtractor for CaImAn.

Classes
-------
CaimanSegmentationExtractor
    A class for extracting segmentation from CaImAn output.
"""

from pathlib import Path
from warnings import warn
import warnings

import h5py

from scipy.sparse import csc_matrix
import numpy as np

from ...extraction_tools import PathType, get_package
from ...multisegmentationextractor import MultiSegmentationExtractor
from ...segmentationextractor import SegmentationExtractor


class CaimanSegmentationExtractor(SegmentationExtractor):
    """A SegmentationExtractor for CaImAn.

    This class inherits from the SegmentationExtractor class, having all
    its functionality specifically applied to the dataset output from
    the 'CaImAn' ROI segmentation method.
    """

    extractor_name = "CaimanSegmentation"
    mode = "file"

    def __init__(self, file_path: PathType):
        """Initialize a CaimanSegmentationExtractor instance.

        Parameters
        ----------
        file_path: str
            The location of the folder containing caiman .hdf5 output file.
        """
        SegmentationExtractor.__init__(self)
        self.file_path = file_path
        self._dataset_file = self._file_extractor_read()
        self._roi_response_raw = self._raw_trace_extractor_read()
        self._roi_response_dff = self._trace_extractor_read("F_dff")
        self._roi_response_denoised = self._trace_extractor_read("C")
        self._roi_response_neuropil = self._trace_extractor_read("f")
        self._roi_response_deconvolved = self._trace_extractor_read("S")
        self._image_correlation = self._correlation_image_read()
        self._image_mean = self._summary_image_read()
        self._sampling_frequency = self._dataset_file["params"]["data"]["fr"][()]
        self._image_masks = self._image_mask_sparse_read()
        self._background_image_masks = self._background_image_mask_read()

    def __del__(self):  # TODO: refactor segmentation extractors who use __del__ together into a base class
        """Close the h5py file when the object is deleted."""
        self._dataset_file.close()

    def _file_extractor_read(self):
        """Read the h5py file.

        Returns
        -------
        h5py.File
            The h5py file object specified by self.file_path.
        """
        return h5py.File(self.file_path, "r")

    def _image_mask_sparse_read(self):
        """Read the image masks from the h5py file.

        Returns
        -------
        image_masks: numpy.ndarray
            The image masks for each ROI.
        """
        roi_ids = self._dataset_file["estimates"]["A"]["indices"]
        masks = self._dataset_file["estimates"]["A"]["data"]
        ids = self._dataset_file["estimates"]["A"]["indptr"]
        image_mask_in = csc_matrix(
            (masks, roi_ids, ids),
            shape=(np.prod(self.get_frame_shape()), self.get_num_rois()),
        ).toarray()
        image_masks = np.reshape(image_mask_in, (*self.get_frame_shape(), -1), order="F")
        return image_masks

    def _background_image_mask_read(self):
        """Read the image masks from the h5py file.

        Returns
        -------
        image_masks: numpy.ndarray
            The image masks for each background components.
        """
        if self._dataset_file["estimates"].get("b"):
            background_image_mask_in = self._dataset_file["estimates"]["b"]
            background_image_masks = np.reshape(background_image_mask_in, (*self.get_frame_shape(), -1), order="F")
            return background_image_masks

    def _trace_extractor_read(self, field):
        """Read the traces specified by the field from the estimates dataset of the h5py file.

        Parameters
        ----------
        field: str
            The field to read from the estimates object.

        Returns
        -------
        lazy_ops.DatasetView
            The traces specified by the field.
        """
        lazy_ops = get_package(package_name="lazy_ops")

        if field in self._dataset_file["estimates"]:
            return lazy_ops.DatasetView(self._dataset_file["estimates"][field]).lazy_transpose()

    def _raw_trace_extractor_read(self):
        """Read the denoised trace and the residual trace from the h5py file and sum them to obtain the raw roi response trace.

        Returns
        -------
        roi_response_raw: numpy.ndarray
            The raw roi response trace.
        """
        roi_response_raw = self._dataset_file["estimates"]["C"][:] + self._dataset_file["estimates"]["YrA"][:]
        return np.array(roi_response_raw.T)

    def _correlation_image_read(self):
        """Read correlation image Cn."""
        if self._dataset_file["estimates"].get("Cn"):
            return np.array(self._dataset_file["estimates"]["Cn"])

    def _summary_image_read(self):
        """Read summary image mean."""
        if self._dataset_file["estimates"].get("b"):
            FOV_shape = self._dataset_file["params"]["data"]["dims"][()]
            b_sum = self._dataset_file["estimates"]["b"][:].sum(axis=1)
            return np.array(b_sum).reshape(FOV_shape, order="F")

    def get_accepted_list(self):
        accepted = self._dataset_file["estimates"]["idx_components"]
        if len(accepted.shape) == 0:
            accepted = list(range(self.get_num_rois()))
        else:
            accepted = list(accepted[:])
        return accepted

    def get_rejected_list(self):
        rejected = self._dataset_file["estimates"]["idx_components_bad"]
        if len(rejected.shape) == 0:
            rejected = list()
        else:
            rejected = list(rejected[:])
        return rejected

    def get_frame_shape(self):
        return self._dataset_file["params"]["data"]["dims"][()]

    def get_image_size(self):
        warnings.warn(
            "get_image_size is deprecated and will be removed on or after January 2026. "
            "Use get_frame_shape instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_frame_shape()
