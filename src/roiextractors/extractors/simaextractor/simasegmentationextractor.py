"""A segmentation extractor for Sima.

Classes
-------
SimaSegmentationExtractor
    A segmentation extractor for Sima.
"""

import importlib
import os
import pickle
import re
import warnings
from shutil import copyfile
from typing import Optional

import numpy as np

from ...extraction_tools import PathType
from ...segmentationextractor import SegmentationExtractor


class SimaSegmentationExtractor(SegmentationExtractor):
    """A segmentation extractor for Sima.

    This class inherits from the SegmentationExtractor class, having all
    its functionality specifically applied to the dataset output from
    the 'SIMA' ROI segmentation method.
    """

    extractor_name = "SimaSegmentation"
    mode = "file"
    # error message when not installed
    installation_mesg = "To use the SimaSegmentationExtractor install sima and dill: \n\n pip install sima/dill\n\n"

    def __init__(self, file_path: PathType, sima_segmentation_label: str = "auto_ROIs"):
        """Create a SegmentationExtractor instance from a sima file.

        Parameters
        ----------
        file_path: str or Path
            The location of the folder containing dataset.sima file and the raw
            image file(s) (tiff, h5, .zip)
        sima_segmentation_label: str
            name of the ROIs in the dataset from which to extract all ROI info
        """
        sima_spec = importlib.util.find_spec("sima")
        dill_spec = importlib.util.find_spec("dill")
        if sima_spec is not None and dill_spec is not None:

            HAVE_SIMA = True
        else:
            HAVE_SIMA = False

        assert HAVE_SIMA, self.installation_mesg
        SegmentationExtractor.__init__(self)
        self.file_path = file_path
        self._convert_sima(file_path)
        self._dataset_file = self._file_extractor_read()
        self._channel_names = [str(i) for i in self._dataset_file.channel_names]
        self._num_of_channels = len(self._channel_names)
        self.sima_segmentation_label = sima_segmentation_label
        self._image_masks = self._image_mask_extractor_read()
        self._roi_response_raw = self._trace_extractor_read()
        self._image_mean = self._summary_image_read()

    @staticmethod
    def _convert_sima(old_pkl_loc):
        """Convert the sima file to python 3 pickle.

        This function is used to convert python 2 pickles to python 3 pickles.
        Forward compatibility of '*.sima' files containing .pkl dataset, rois,
        sequences, signals, time_averages.

        Replaces the pickle file with a python 3 version with the same name. Saves
        the old Py2 pickle as 'oldpicklename_p2.pkl'

        Parameters
        ----------
        old_pkl_loc: str
            Path of the pickle file to be converted
        """
        import dill

        # Make a name for the new pickle
        old_pkl_loc = old_pkl_loc + "/"
        for dirpath, dirnames, filenames in os.walk(old_pkl_loc):
            _exit = [True for file in filenames if "_p2.pkl" in file]
            if True in _exit:
                print("pickle already in Py3 format")
                continue
            for file in filenames:
                if ".pkl" in file:
                    old_pkl = os.path.join(dirpath, file)
                    print(old_pkl)
                    # Make a name for the new pickle
                    new_pkl_name = os.path.splitext(os.path.basename(old_pkl))[0] + "_p2.pkl"
                    base_directory = os.path.split(old_pkl)[0]
                    new_pkl = base_directory + "/" + new_pkl_name
                    # Convert Python 2 "ObjectType" to Python 3 object
                    dill._dill._reverse_typemap["ObjectType"] = object

                    # Open the pickle using latin1 encoding
                    with open(old_pkl, "rb") as f:
                        loaded = pickle.load(f, encoding="latin1")
                    copyfile(old_pkl, new_pkl)
                    os.remove(f.name)
                    # Re-save as Python 3 pickle
                    with open(old_pkl, "wb") as outfile:
                        pickle.dump(loaded, outfile)

    def _file_extractor_read(self):
        """Read the sima file and return the sima.ImagingDataset object."""
        import sima

        _img_dataset = sima.ImagingDataset.load(self.file_path)
        _img_dataset._savedir = self.file_path
        return _img_dataset

    def _image_mask_extractor_read(self):
        """Read the image mask from the sima.ImagingDataset object (self._dataset_file)."""
        _sima_rois = self._dataset_file.ROIs
        if len(_sima_rois) > 1:
            if self.sima_segmentation_label in list(_sima_rois.keys()):
                _sima_rois_data = _sima_rois[self.sima_segmentation_label]
            else:
                raise Exception("Enter a valid name of ROIs from: {}".format(",".join(list(_sima_rois.keys()))))
        elif len(_sima_rois) == 1:
            _sima_rois_data = list(_sima_rois.values())[0]
            self.sima_segmentation_label = list(_sima_rois.keys())[0]
        else:
            raise Exception("no ROIs found in the sima file")
        image_masks_ = [np.squeeze(np.array(roi_dat)).T for roi_dat in _sima_rois_data]
        return np.array(image_masks_).T

    def _trace_extractor_read(self):
        """Read the traces from the sima.ImagingDataset object (self._dataset_file)."""
        for channel_now in self._channel_names:
            for labels in self._dataset_file.signals(channel=channel_now):
                if labels:
                    _active_channel = channel_now
                    break
            print(
                "extracting signal from channel {} from {} no of channels".format(
                    _active_channel, self._num_of_channels
                )
            )
        # label for the extraction method in SIMA:
        for labels in self._dataset_file.signals(channel=_active_channel):
            _count = 0
            if not re.findall(r"[\d]{4}-[\d]{2}-[\d]{2}-", labels):
                _count = _count + 1
                _label = labels
                break
        if _count > 1:
            print("multiple labels found for extract method using {}".format(_label))
        elif _count == 0:
            print("no label found for extract method using {}".format(labels))
            _label = labels
        extracted_signals = np.array(self._dataset_file.signals(channel=_active_channel)[_label]["raw"][0])
        return extracted_signals

    def _summary_image_read(self):
        """Read the summary image from the sima.ImagingDataset object (self._dataset_file)."""
        summary_image = np.squeeze(self._dataset_file.time_averages[0]).T
        return np.array(summary_image).T

    def get_accepted_list(self):
        return list(range(self.get_num_rois()))

    def get_rejected_list(self):
        return [a for a in range(self.get_num_rois()) if a not in set(self.get_accepted_list())]

    def get_frame_shape(self):
        """Get the frame shape (height, width) of the movie.

        Returns
        -------
        tuple
            The frame shape as (height, width).
        """
        return self._image_masks.shape[0:2]

    def get_image_size(self):
        warnings.warn(
            "get_image_size is deprecated and will be removed on or after January 2026. "
            "Use get_frame_shape instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_frame_shape()

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Retrieve the original unaltered timestamps for the data in this interface.

        Returns
        -------
        timestamps: numpy.ndarray or None
            The timestamps for the data stream, or None if native timestamps are not available.
        """
        # SIMA segmentation data does not have native timestamps
        return None
