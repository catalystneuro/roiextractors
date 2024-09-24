"""Imaging and Segmenation Extractors for .npy files.

Classes
-------
NumpyImagingExtractor
    An ImagingExtractor specified by timeseries .npy file, sampling frequency, and channel names.
NumpySegmentationExtractor
    A Segmentation extractor specified by image masks and traces .npy files.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from ...extraction_tools import PathType, FloatType, ArrayType, IntType
from ...imagingextractor import ImagingExtractor
from ...segmentationextractor import SegmentationExtractor


class NumpyImagingExtractor(ImagingExtractor):
    """An ImagingExtractor specified by timeseries .npy file, sampling frequency, and channel names."""

    extractor_name = "NumpyImagingExtractor"
    installed = True
    is_writable = True
    installation_mesg = ""  # error message when not installed

    def __init__(self, timeseries: Union[PathType, np.ndarray], sampling_frequency: FloatType):
        """Create a NumpyImagingExtractor from a .npy file or a numpy.ndarray.

        Parameters
        ----------
        timeseries: PathType or numpy.ndarray
            Path to .npy file or numpy array containing the video.
        sampling_frequency: FloatType
            Sampling frequency of the video in Hz.
        """
        super().__init__()

        if isinstance(timeseries, (str, Path)):
            timeseries = Path(timeseries)
            if timeseries.is_file():
                assert timeseries.suffix == ".npy", "'timeseries' file is not a numpy file (.npy)"
                self.is_dumpable = True
                self._video = np.load(timeseries, mmap_mode="r")
                self._kwargs = {
                    "timeseries": str(Path(timeseries).absolute()),
                    "sampling_frequency": sampling_frequency,
                }
            else:
                raise ValueError("'timeseries' is does not exist")
        elif isinstance(timeseries, np.ndarray):
            self.is_dumpable = False
            self._video = timeseries
            self._kwargs = {
                "timeseries": timeseries,
                "sampling_frequency": sampling_frequency,
            }
        else:
            raise TypeError("'timeseries' can be a str or a numpy array")

        self._sampling_frequency = float(sampling_frequency)
        self._num_frames, self._num_rows, self._num_columns = self._video.shape
        self._dtype = self._video.dtype

    def get_video(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> np.ndarray:
        start_frame, end_frame = self._validate_get_video_arguments(start_frame=start_frame, end_frame=end_frame)
        return self._video[start_frame:end_frame, ...]

    def get_image_size(self) -> Tuple[int, int]:
        return (self._num_rows, self._num_columns)

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_dtype(self):
        return self._dtype


class NumpySegmentationExtractor(SegmentationExtractor):
    """A Segmentation extractor specified by image masks and traces .npy files.

    NumpySegmentationExtractor objects are built to contain all data coming from
    a file format for which there is currently no support. To construct this,
    all data must be entered manually as arguments.
    """

    extractor_name = "NumpySegmentationExtractor"
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = "file"
    installation_mesg = ""  # error message when not installed

    def __init__(
        self,
        image_masks,
        raw=None,
        dff=None,
        deconvolved=None,
        background=None,
        accepted_lst=None,
        mean_image=None,
        correlation_image=None,
        roi_ids=None,
        roi_locations=None,
        sampling_frequency=None,
        rejected_list=None,
        channel_names=None,
        movie_dims=None,
    ):
        """Create a NumpySegmentationExtractor from a .npy file.

        Parameters
        ----------
        image_masks: np.ndarray
            Binary image for each of the regions of interest
        raw: np.ndarray
            Fluorescence response of each of the ROI in time
        dff: np.ndarray
            DfOverF response of each of the ROI in time
        deconvolved: np.ndarray
            deconvolved response of each of the ROI in time
        background: np.ndarray
            background response of each of the ROI in time
        mean_image: np.ndarray
            Mean image
        correlation_image: np.ndarray
            correlation image
        roi_ids: int list
            Unique ids of the ROIs if any
        roi_locations: np.ndarray
            x and y location representative of ROI mask
        sampling_frequency: float
            Frame rate of the movie
        rejected_list: list
            list of ROI ids that are rejected manually or via automated rejection
        channel_names: list
            list of strings representing channel names
        movie_dims: tuple
            height x width of the movie
        """
        SegmentationExtractor.__init__(self)
        if isinstance(image_masks, (str, Path)):
            image_masks = Path(image_masks)
            if image_masks.is_file():
                assert image_masks.suffix == ".npy", "'image_masks' file is not a numpy file (.npy)"

                self.is_dumpable = True
                self._image_masks = np.load(image_masks, mmap_mode="r")

                if raw is not None:
                    raw = Path(raw)
                    assert raw.suffix == ".npy", "'raw' file is not a numpy file (.npy)"
                    self._roi_response_raw = np.load(raw, mmap_mode="r")
                if dff is not None:
                    dff = Path(dff)
                    assert dff.suffix == ".npy", "'dff' file is not a numpy file (.npy)"
                    self._roi_response_dff = np.load(dff, mmap_mode="r")
                    self._roi_response_background = np.load(background, mmap_mode="r")
                if deconvolved is not None:
                    deconvolved = Path(deconvolved)
                    assert deconvolved.suffix == ".npy", "'deconvolved' file is not a numpy file (.npy)"
                    self._roi_response_deconvolved = np.load(deconvolved, mmap_mode="r")
                if background is not None:
                    background = Path(background)
                    assert background.suffix == ".npy", "'background' file is not a numpy file (.npy)"
                    self._roi_response_background = np.load(background, mmap_mode="r")

                self._kwargs = {"image_masks": str(Path(image_masks).absolute())}
                if raw is not None:
                    self._kwargs.update({"raw": str(Path(raw).absolute())})
                if raw is not None:
                    self._kwargs.update({"dff": str(Path(dff).absolute())})
                if raw is not None:
                    self._kwargs.update({"background": str(Path(background).absolute())})
                if raw is not None:
                    self._kwargs.update({"deconvolved": str(Path(deconvolved).absolute())})

            else:
                raise ValueError("'timeeseries' is does not exist")
        elif isinstance(image_masks, np.ndarray):
            NoneType = type(None)
            assert isinstance(raw, (np.ndarray, NoneType))
            assert isinstance(dff, (np.ndarray, NoneType))
            assert isinstance(background, (np.ndarray, NoneType))
            assert isinstance(deconvolved, (np.ndarray, NoneType))
            self.is_dumpable = False
            self._image_masks = image_masks
            self._roi_response_raw = raw
            assert self._image_masks.shape[-1] == self._roi_response_raw.shape[-1], (
                "Inconsistency between image masks and raw traces. "
                "Image masks must be (px, py, num_rois), "
                "traces must be (num_frames, num_rois)"
            )
            self._roi_response_dff = dff
            if self._roi_response_dff is not None:
                assert self._image_masks.shape[-1] == self._roi_response_dff.shape[-1], (
                    "Inconsistency between image masks and raw traces. "
                    "Image masks must be (px, py, num_rois), "
                    "traces must be (num_frames, num_rois)"
                )
            self._roi_response_background = background
            if self._roi_response_background is not None:
                assert self._image_masks.shape[-1] == self._roi_response_background.shape[-1], (
                    "Inconsistency between image masks and raw traces. "
                    "Image masks must be (px, py, num_rois), "
                    "traces must be (num_frames, num_rois)"
                )
            self._roi_response_deconvolved = deconvolved
            if self._roi_response_deconvolved is not None:
                assert self._image_masks.shape[-1] == self._roi_response_deconvolved.shape[-1], (
                    "Inconsistency between image masks and raw traces. "
                    "Image masks must be (px, py, num_rois), "
                    "traces must be (num_frames, num_rois)"
                )
            self._kwargs = {
                "image_masks": image_masks,
                "signal": raw,
                "dff": dff,
                "background": background,
                "deconvolved": deconvolved,
            }
        else:
            raise TypeError("'image_masks' can be a str or a numpy array")
        self._movie_dims = movie_dims if movie_dims is not None else image_masks.shape
        self._image_mean = mean_image
        self._image_correlation = correlation_image
        if roi_ids is None:
            self._roi_ids = list(np.arange(image_masks.shape[2]))
        else:
            assert all([isinstance(roi_id, (int, np.integer)) for roi_id in roi_ids]), "'roi_ids' must be int!"
            self._roi_ids = roi_ids
        self._roi_locs = roi_locations
        self._sampling_frequency = sampling_frequency
        self._channel_names = channel_names
        self._rejected_list = rejected_list
        self._accepted_list = accepted_lst

    @property
    def image_dims(self):
        """Return the dimensions of the image.

        Returns
        -------
        image_dims: list
            The dimensions of the image (num_rois, num_rows, num_columns).
        """
        return list(self._image_masks.shape[0:2])

    def get_accepted_list(self):
        if self._accepted_list is None:
            return list(range(self.get_num_rois()))
        else:
            return self._accepted_list

    def get_rejected_list(self):
        if self._rejected_list is None:
            return [a for a in range(self.get_num_rois()) if a not in set(self.get_accepted_list())]
        else:
            return self._rejected_list

    @property
    def roi_locations(self):
        """Returns the center locations (x, y) of each ROI."""
        if self._roi_locs is None:
            num_ROIs = self.get_num_rois()
            raw_images = self._image_masks
            roi_location = np.ndarray([2, num_ROIs], dtype="int")
            for i in range(num_ROIs):
                temp = np.where(raw_images[:, :, i] == np.amax(raw_images[:, :, i]))
                roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
            return roi_location
        else:
            return self._roi_locs

    @staticmethod
    def write_segmentation(segmentation_object, save_path):
        """Write a NumpySegmentationExtractor to a .npy file.

        Parameters
        ----------
        segmentation_object: NumpySegmentationExtractor
            The segmentation extractor object to be written to file.
        save_path: str or PathType
            Path to .npy file.

        Notes
        -----
        This method is not implemented yet.
        """
        raise NotImplementedError

    # defining the abstract class informed methods:
    def get_roi_ids(self):
        if self._roi_ids is None:
            return list(range(self.get_num_rois()))
        else:
            return self._roi_ids

    def get_image_size(self):
        return self._movie_dims
