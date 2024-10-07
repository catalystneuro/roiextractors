"""A SegmentationExtractor for Minian.

Classes
-------
MinianSegmentationExtractor
    A class for extracting segmentation from Minian output.
"""

from pathlib import Path

import zarr
import warnings
import numpy as np

from ...extraction_tools import PathType
from ...segmentationextractor import SegmentationExtractor


class MinianSegmentationExtractor(SegmentationExtractor):
    """A SegmentationExtractor for Minian.

    This class inherits from the SegmentationExtractor class, having all
    its functionality specifically applied to the dataset output from
    the 'Minian' ROI segmentation method.
    """

    extractor_name = "MinianSegmentation"
    is_writable = True
    mode = "file"

    def __init__(self, folder_path: PathType):
        """Initialize a MinianSegmentationExtractor instance.

        Parameters
        ----------
        folder_path: str
            The location of the folder containing minian .zarr output.
        """
        SegmentationExtractor.__init__(self)
        self.folder_path = folder_path
        self._roi_response_denoised = self._trace_extractor_read(field = "C")
        self._roi_response_baseline = self._trace_extractor_read(field = "b0")
        self._roi_response_neuropil = self._trace_extractor_read(field = "f")
        self._roi_response_deconvolved = self._trace_extractor_read(field = "S")
        self._image_maximum_projection = self._file_extractor_read("/max_proj.zarr/max_proj")
        self._image_masks = self._roi_image_mask_read()
        self._background_image_masks = self._background_image_mask_read()


    def _file_extractor_read(self, zarr_group = ""):
        """Read the zarr.

        Returns
        -------
        zarr.open
            The zarr object specified by self.folder_path.
        """
        if zarr_group not in zarr.open(self.folder_path, mode="r"):
            warnings.warn(f"Group '{zarr_group}' not found in the Zarr store.", UserWarning)
            return None
        else:
            return zarr.open(self.folder_path + f"/{zarr_group}", "r")

    def _roi_image_mask_read(self):
        """Read the image masks from the zarr output.

        Returns
        -------
        image_masks: numpy.ndarray
            The image masks for each ROI.
        """
        dataset = self._file_extractor_read("/A.zarr")
        if dataset is None or "A" not in dataset:
            return None
        else:
            return np.transpose(dataset["A"], (1,2,0))

    def _background_image_mask_read(self):
        """Read the image masks from the zarr output.

        Returns
        -------
        image_masks: numpy.ndarray
            The image masks for each background components.
        """
        dataset = self._file_extractor_read("/b.zarr")
        if dataset is None or "b" not in dataset:
            return None
        else:
            return np.expand_dims(dataset["b"], axis=2)

    def _trace_extractor_read(self, field):
        """Read the traces specified by the field from the zarr object.

        Parameters
        ----------
        field: str
            The field to read from the zarr object.

        Returns
        -------
        trace: numpy.ndarray
            The traces specified by the field.
        """

        dataset = self._file_extractor_read(f"/{field}.zarr")

        if dataset is None or field not in dataset:
            return None
        elif dataset[field].ndim == 2:
            return np.transpose(dataset[field])
        elif dataset[field].ndim == 1:
            return np.expand_dims(dataset[field], axis=1)

    def get_image_size(self):
        dataset = self._file_extractor_read("/A.zarr")
        height = dataset["height"].shape[0]
        width = dataset["width"].shape[0]
        return (height, width)

    def get_accepted_list(self) -> list:
        """Get a list of accepted ROI ids.

        Returns
        -------
        accepted_list: list
            List of accepted ROI ids.
        """
        return list(range(self.get_num_rois()))

    def get_rejected_list(self) -> list:
        """Get a list of rejected ROI ids.

        Returns
        -------
        rejected_list: list
            List of rejected ROI ids.
        """
        return list()

    def get_roi_ids(self) -> list:
        dataset = self._file_extractor_read("/A.zarr")
        return list(dataset["unit_id"])

    def get_traces_dict(self) -> dict:
        """Get traces as a dictionary with key as the name of the ROiResponseSeries.

        Returns
        -------
        _roi_response_dict: dict
            dictionary with key, values representing different types of RoiResponseSeries:
                Raw Fluorescence, DeltaFOverF, Denoised, Neuropil, Deconvolved, Background, etc.
        """
        return dict(
            denoised=self._roi_response_denoised,
            baseline=self._roi_response_baseline,
            neuropil=self._roi_response_neuropil,
            deconvolved=self._roi_response_deconvolved,
        )

    def get_images_dict(self) -> dict:
        """Get images as a dictionary with key as the name of the ROIResponseSeries.

        Returns
        -------
        _roi_image_dict: dict
            dictionary with key, values representing different types of Images used in segmentation:
                Mean, Correlation image
        """
        return dict(mean=self._image_mean, correlation=self._image_correlation, maximum_projection=self._image_maximum_projection)