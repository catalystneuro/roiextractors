"""Extractor for reading the segmentation data that results from calls to EXTRACT (Public)."""
from typing import Optional

import numpy as np

from ...extraction_tools import PathType
from ...segmentationextractor import SegmentationExtractor

try:
    import h5py  # 'output' will nearly always be large enough to require saving with -v7.3

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

try:
    from lazy_ops import DatasetView

    HAVE_LAZY_OPS = True
except ImportError:
    HAVE_LAZY_OPS = False


class ExtractPublicSegmentationExtractor(SegmentationExtractor):
    """Load a SegmentationExtractor from a .mat file containing the output and config structs from EXTRACT."""

    extractor_name = "ExtractPublicSegmentationExtractor"
    installed = HAVE_H5PY and HAVE_LAZY_OPS
    is_writable = False
    mode = "file"
    installation_mesg = """
    To use ExtractPublicSegmentationExtractor, install h5py and lazy_ops:
        \n\n pip install h5py \n\n pip install lazy_ops \n\n
    """

    def __init__(
        self, file_path: PathType, output_struct_name: Optional[str] = None, config_struct_name: Optional[str] = None
    ):
        """
        Load a SegmentationExtractor from a .mat file containing the output and config structs of the EXTRACT algorithm.

        Parameters
        ----------
        file_path: PathType
            Path to the .mat file containing the structs.
        output_struct_name: str, optional
            The user has control over the names of the variables that return from `extraction(images, config)`.
            The tutorials for EXTRACT follow the naming convention of 'output', which we assume as the default.
        config_struct_name: str, optional
            The user has control over the names of the variables passed into `extraction(images, config)`.
            The tutorials for EXTRACT follow the naming convention of 'config', which we assume as the default.
        """
        assert self.installed, self.installation_mesg

        output_struct_name = output_struct_name or "output"
        config_struct_name = config_struct_name or "config"

        self._dataset_file = h5py.File(name=file_path, mode="r")
        if self._dataset_file[self._group0[0]]["config"]["preprocess"][0, 0] == 1:
            self._roi_response_dff = self._roi_response_raw
            self._roi_response_raw = None
        self._sampling_frequency = None  # should be able to get this from config

    def _image_mask_extractor_read(self):
        return DatasetView(self._dataset_file[self._group0[0]]["spatial_weights"]).lazy_transpose()

    def _trace_extractor_read(self):
        return self._dataset_file[self._group0[0]]["temporal_weights"]

    def _tot_exptime_extractor_read(self):
        return np.nan

    def _raw_datafile_read(self):
        pass
