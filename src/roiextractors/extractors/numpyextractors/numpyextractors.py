"""Imaging and Segmenation Extractors for .npy files.

Classes
-------
NumpyImagingExtractor
    An ImagingExtractor specified by timeseries np.ndarray or .npy file and sampling frequency.
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
    """An ImagingExtractor specified by timeseries np.ndarray or .npy file and sampling frequency."""

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
        background_ids=None,
        sampling_frequency=None,
        rejected_list=None,
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
        """
        SegmentationExtractor.__init__(self)
        NoneType = type(None)
        assert not all(
            isinstance(response, NoneType) for response in [raw, dff, deconvolved, background]
        ), "At least one of 'raw', 'dff', 'deconvolved', 'background' must be provided."
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
                    self._num_frames = self._roi_response_raw.shape[0]
                if dff is not None:
                    dff = Path(dff)
                    assert dff.suffix == ".npy", "'dff' file is not a numpy file (.npy)"
                    self._roi_response_dff = np.load(dff, mmap_mode="r")
                    self._num_frames = self._roi_response_dff.shape[0]
                if deconvolved is not None:
                    deconvolved = Path(deconvolved)
                    assert deconvolved.suffix == ".npy", "'deconvolved' file is not a numpy file (.npy)"
                    self._roi_response_deconvolved = np.load(deconvolved, mmap_mode="r")
                    self._num_frames = self._roi_response_deconvolved.shape[0]
                if background is not None:
                    background = Path(background)
                    assert background.suffix == ".npy", "'background' file is not a numpy file (.npy)"
                    self._roi_response_background = np.load(background, mmap_mode="r")
                    self._num_frames = self._roi_response_background.shape[0]

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
            assert isinstance(raw, (np.ndarray, NoneType))
            assert isinstance(dff, (np.ndarray, NoneType))
            assert isinstance(background, (np.ndarray, NoneType))
            assert isinstance(deconvolved, (np.ndarray, NoneType))
            self.is_dumpable = False
            self._image_masks = image_masks
            self._roi_response_raw = raw
            if self._roi_response_raw is not None:
                assert self._image_masks.shape[-1] == self._roi_response_raw.shape[-1], (
                    "Inconsistency between image masks and raw traces. "
                    "Image masks must be (px, py, num_rois), "
                    "traces must be (num_frames, num_rois)"
                )
                self._num_frames = self._roi_response_raw.shape[0]
            self._roi_response_dff = dff
            if self._roi_response_dff is not None:
                assert self._image_masks.shape[-1] == self._roi_response_dff.shape[-1], (
                    "Inconsistency between image masks and raw traces. "
                    "Image masks must be (px, py, num_rois), "
                    "traces must be (num_frames, num_rois)"
                )
                self._num_frames = self._roi_response_dff.shape[0]
            self._roi_response_background = background
            if self._roi_response_background is not None:
                assert self._image_masks.shape[-1] == self._roi_response_background.shape[-1], (
                    "Inconsistency between image masks and raw traces. "
                    "Image masks must be (px, py, num_rois), "
                    "traces must be (num_frames, num_rois)"
                )
                self._num_frames = self._roi_response_background.shape[0]
            self._roi_response_deconvolved = deconvolved
            if self._roi_response_deconvolved is not None:
                assert self._image_masks.shape[-1] == self._roi_response_deconvolved.shape[-1], (
                    "Inconsistency between image masks and raw traces. "
                    "Image masks must be (px, py, num_rois), "
                    "traces must be (num_frames, num_rois)"
                )
                self._num_frames = self._roi_response_deconvolved.shape[0]
            self._kwargs = {
                "image_masks": image_masks,
                "signal": raw,
                "dff": dff,
                "background": background,
                "deconvolved": deconvolved,
            }
        else:
            raise TypeError("'image_masks' can be a str or a numpy array")
        self._image_size = image_masks.shape[:2]
        self._num_rois = image_masks.shape[2]
        self._image_mean = mean_image
        self._image_correlation = correlation_image
        if roi_ids is None:
            self._roi_ids = list(np.arange(image_masks.shape[2]))
        else:
            assert all([isinstance(roi_id, (int, np.integer)) for roi_id in roi_ids]), "'roi_ids' must be int!"
            self._roi_ids = roi_ids
        if background_ids is not None:
            assert all(
                [isinstance(background_id, (int, np.integer)) for background_id in background_ids]
            ), "'background_ids' must be int!"
            self._background_ids = background_ids
            self._num_background_components = len(background_ids)
            assert background is not None, "'background' must be provided if 'background_ids' is provided!"
        elif background is not None:
            self._num_background_components = self._roi_response_background.shape[1]
            self._background_ids = list(np.arange(self._num_background_components))
        else:
            self._background_ids = None
            self._num_background_components = None

        if roi_locations is not None:
            self._roi_locations = roi_locations
        else:
            roi_location = np.zeros([2, len(self._roi_ids)], dtype="int")
            for i, _ in enumerate(roi_ids):
                image_mask = self._image_masks[:, :, i]
                temp = np.where(image_mask == np.amax(image_mask))
                roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T

        self._sampling_frequency = sampling_frequency
        self._rejected_list = rejected_list
        self._accepted_list = accepted_lst
        self._num_frames = self._roi_response_raw.shape[0]

    def get_roi_image_masks(self, roi_ids=None) -> np.ndarray:
        if roi_ids is None:
            return self._image_masks
        all_ids = self.get_roi_ids()
        roi_indices = [all_ids.index(i) for i in roi_ids]
        return self._image_masks[:, :, roi_indices]

    def get_roi_ids(self):
        return self._roi_ids

    def get_background_ids(self) -> list:
        return self._background_ids

    def get_image_size(self):
        return self._image_size

    def get_background_image_masks(self, background_ids=None) -> np.ndarray:
        if background_ids is None:
            return self._background_image_masks
        all_ids = self.get_background_ids()
        background_indices = [all_ids.index(i) for i in background_ids]
        return self._image_masks[:, :, background_indices]

    def get_roi_response_traces(
        self,
        names: Optional[list[str]] = None,
        roi_ids: Optional[ArrayType] = None,
        start_frame: Optional[IntType] = None,
        end_frame: Optional[IntType] = None,
    ) -> dict:
        all_roi_response_traces = dict(
            raw=self._roi_response_raw,
            dff=self._roi_response_dff,
            deconvolved=self._roi_response_deconvolved,
            # denoised=self._roi_response_denoised,
        )
        names = names if names is not None else list(all_roi_response_traces.keys())
        all_ids = self.get_roi_ids()
        roi_ids = roi_ids if roi_ids is not None else all_ids
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else self.get_num_frames()

        roi_indices = [all_ids.index(i) for i in roi_ids]
        roi_response_traces = {
            name: all_roi_response_traces[name][start_frame:end_frame, roi_indices] for name in names
        }
        return roi_response_traces

    def get_background_response_traces(
        self,
        names: Optional[list[str]] = None,
        background_ids: Optional[ArrayType] = None,
        start_frame: Optional[IntType] = None,
        end_frame: Optional[IntType] = None,
    ) -> dict:
        all_background_response_traces = dict(background=self._roi_response_background)
        names = names if names is not None else list(all_background_response_traces.keys())
        all_ids = self.get_background_ids()
        background_ids = background_ids if background_ids is not None else all_ids
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else self.get_num_frames()

        roi_indices = [all_ids.index(i) for i in background_ids]
        background_response_traces = {
            name: all_background_response_traces[name][start_frame:end_frame, roi_indices] for name in names
        }
        return background_response_traces

    def get_summary_images(self, names: Optional[list[str]] = None) -> dict:
        names = names if names is not None else ["mean", "correlation"]
        all_summary_images = dict(
            mean=self._image_mean,
            correlation=self._image_correlation,
        )
        summary_images = {name: all_summary_images[name] for name in names}
        return summary_images

    def get_num_frames(self):
        return self._num_frames

    def get_roi_locations(self, roi_ids=None):
        all_roi_ids = self.get_roi_ids()
        roi_ids = roi_ids if roi_ids is not None else all_roi_ids
        roi_indices = [all_roi_ids.index(i) for i in roi_ids]
        return self._roi_locations[:, roi_indices]

    def get_num_rois(self):
        return self._num_rois

    def get_num_background_components(self) -> int:
        return self._num_background_components

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_accepted_roi_ids(self) -> list:
        return self._accepted_list

    def get_rejected_roi_ids(self) -> list:
        return self._rejected_list
