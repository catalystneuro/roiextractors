"""Imaging and Segmenation Extractors for .npy files.

Classes
-------
NumpyImagingExtractor
    An ImagingExtractor specified by timeseries np.ndarray or .npy file and sampling frequency.
NumpySegmentationExtractor
    A Segmentation extractor specified by image masks and traces .npy files.
"""

from pathlib import Path
from typing import Optional, Tuple, Union, get_args

import numpy as np

from ...extraction_tools import (
    PathType,
    FloatType,
    ArrayType,
    IntType,
    NoneType,
    get_default_roi_locations_from_image_masks,
)
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
        # python 3.9 doesn't support get_instance on a Union of types, so we use get_args
        if isinstance(timeseries, get_args(PathType)):
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
        image_masks: Union[PathType, np.ndarray],
        roi_response_traces: dict[str, Union[PathType, np.ndarray]],
        sampling_frequency: FloatType,
        roi_ids: Optional[list] = None,
        accepted_roi_ids: Optional[list] = None,
        rejected_roi_ids: Optional[list] = None,
        roi_locations: Optional[ArrayType] = None,
        summary_images: Optional[dict[str, Union[PathType, np.ndarray]]] = None,
        background_image_masks: Optional[Union[PathType, np.ndarray]] = None,
        background_response_traces: Optional[dict[str, Union[PathType, np.ndarray]]] = None,
        background_ids: Optional[list] = None,
    ):
        """Create a NumpySegmentationExtractor from a set of .npy files or a set of np.ndarrays.

        Parameters
        ----------
        image_masks: Union[PathType, np.ndarray]
            Binary image for each of the regions of interest.
        roi_response_traces: dict[str, Union[PathType, np.ndarray]]
            Dictionary containing the fluorescence response of each ROI in time.
        sampling_frequency: FloatType
            Frame rate of the movie.
        roi_ids: Optional[list]
            Unique ids of the ROIs. If None, then the indices are used.
        accepted_roi_ids: Optional[list]
            List of ROI ids that are accepted. If None, then all ROI ids are accepted.
        rejected_roi_ids: Optional[list]
            List of ROI ids that are rejected manually or via automated rejection. If None, then no ROI ids are rejected.
        roi_locations: Optional[ArrayType]
            x and y location representative of ROI mask. If None, then the maximum location is used.
        summary_images: Optional[dict[str, Union[PathType, np.ndarray]]]
            Dictionary containing summary images like mean image, correlation image, etc.
        background_image_masks: Optional[Union[PathType, np.ndarray]]
            Binary image for each of the background components.
        background_response_traces: Optional[dict[str, Union[PathType, np.ndarray]]]
            Dictionary containing the background response of each component in time.
        background_ids: Optional[list]
            Unique ids of the background components. If None, then the indices are used.

        Notes
        -----
        If any of image_masks, roi_response_traces, summary_images, background_image_masks, or background_response_traces
        are .npy files, then the rest of them must be .npy files as well.
        """
        super().__init__()
        self._sampling_frequency = float(sampling_frequency)
        # python 3.9 doesn't support get_instance on a Union of types, so we use get_args
        if isinstance(image_masks, get_args(PathType)):
            self._init_from_npy(
                image_masks=image_masks,
                roi_response_traces=roi_response_traces,
                summary_images=summary_images,
                background_image_masks=background_image_masks,
                background_response_traces=background_response_traces,
            )

        elif isinstance(image_masks, np.ndarray):
            self._init_from_ndarray(
                image_masks=image_masks,
                roi_response_traces=roi_response_traces,
                summary_images=summary_images,
                background_image_masks=background_image_masks,
                background_response_traces=background_response_traces,
            )
        else:
            raise TypeError(
                f"'image_masks' must be a PathType (str, pathlib.Path) or a numpy array but got {type(image_masks)}"
            )

        self._image_size = self._image_masks.shape[:2]
        self._num_rois = self._image_masks.shape[2]
        self._num_frames = list(self._roi_response_traces.values())[0].shape[0]
        self._roi_ids = roi_ids if roi_ids is not None else list(np.arange(self._num_rois))
        self._accepted_roi_ids = accepted_roi_ids if accepted_roi_ids is not None else self._roi_ids
        self._rejected_roi_ids = (
            rejected_roi_ids if rejected_roi_ids is not None else list(set(self._roi_ids) - set(self._accepted_roi_ids))
        )

        if roi_locations is not None:
            self._roi_locations = roi_locations
        else:
            self._roi_locations = get_default_roi_locations_from_image_masks(self._image_masks)
        if background_image_masks is not None:
            self._num_background_components = self._background_image_masks.shape[2]
            self._background_ids = (
                background_ids if background_ids is not None else list(np.arange(self._num_background_components))
            )

    def _init_from_npy(
        self,
        image_masks: PathType,
        roi_response_traces: dict[str, PathType],
        summary_images: Optional[dict[str, PathType]],
        background_image_masks: Optional[PathType],
        background_response_traces: Optional[dict[str, PathType]],
    ):
        image_masks = Path(image_masks)
        assert image_masks.is_file(), "'image_masks' file does not exist"
        assert image_masks.suffix == ".npy", "'image_masks' file is not a numpy file (.npy)"

        self.is_dumpable = True
        self._image_masks = np.load(image_masks, mmap_mode="r")

        self._roi_response_traces = {}
        for name, trace in roi_response_traces.items():
            assert isinstance(
                trace,
                get_args(PathType),  # python 3.9 doesn't support get_instance on a Union of types, so we use get_args
            ), f"Since image_masks is a .npy file, roi response '{name}' must also be an .npy file but got {type(trace)}."
            trace = Path(trace)
            assert trace.is_file(), f"'{name}' file does not exist"
            assert trace.suffix == ".npy", f"'{name}' file is not a numpy file (.npy)"
            self._roi_response_traces[name] = np.load(trace, mmap_mode="r")

        if summary_images is not None:
            self._summary_images = {}
            for name, image in summary_images.items():
                assert isinstance(
                    image,
                    get_args(
                        PathType
                    ),  # python 3.9 doesn't support get_instance on a Union of types, so we use get_args
                ), f"Since image_masks is a .npy file, summary image '{name}' must also be an .npy file but got {type(image)}."
                image = Path(image)
                assert image.is_file(), f"'{name}' file does not exist"
                assert image.suffix == ".npy", f"'{name}' file is not a numpy file (.npy)"
                self._summary_images[name] = np.load(image, mmap_mode="r")

        if background_image_masks is not None:
            assert isinstance(
                background_image_masks,
                get_args(PathType),  # python 3.9 doesn't support get_instance on a Union of types, so we use get_args
            ), f"Since image_masks is a .npy file, background image masks must also be a .npy file but got {type(background_image_masks)}."
            background_image_masks = Path(background_image_masks)
            assert background_image_masks.is_file(), "'background_image_masks' file does not exist"
            assert background_image_masks.suffix == ".npy", "'background_image_masks' file is not a numpy file (.npy)"
            self._background_image_masks = np.load(background_image_masks, mmap_mode="r")

        if background_response_traces is not None:
            self._background_response_traces = {}
            for name, trace in background_response_traces.items():
                assert isinstance(
                    trace,
                    get_args(
                        PathType
                    ),  # python 3.9 doesn't support get_instance on a Union of types, so we use get_args
                ), f"Since image_masks is a .npy file, background response '{name}' must also be a .npy file but got {type(trace)}."
                trace = Path(trace)
                assert trace.is_file(), f"'{name}' file does not exist"
                assert trace.suffix == ".npy", f"'{name}' file is not a numpy file (.npy)"
                self._background_response_traces[name] = np.load(trace, mmap_mode="r")

    def _init_from_ndarray(
        self, image_masks, roi_response_traces, summary_images, background_image_masks, background_response_traces
    ):
        self.is_dumpable = False
        self._image_masks = image_masks

        self._roi_response_traces = roi_response_traces
        for name, trace in self._roi_response_traces.items():
            assert isinstance(
                trace, np.ndarray
            ), f"Since image_masks is a numpy array, roi response '{name}' must also be a numpy array but got {type(trace)}."
            assert trace.shape[-1] == self._image_masks.shape[-1], (
                f"Inconsistency between image masks and {name} traces. "
                f"Image masks must be (num_rows, num_columns, num_rois), "
                f"traces must be (num_frames, num_rois)"
            )
        if summary_images is not None:
            self._summary_images = summary_images
            for name, image in self._summary_images.items():
                assert image.shape[:2] == self._image_masks.shape[:2], (
                    f"Inconsistency between image masks and {name} images. "
                    f"Image masks must be (num_rows, num_columns, num_rois), "
                    f"images must be (num_rows, num_columns)"
                )

        if background_image_masks is not None:
            assert isinstance(
                background_image_masks, np.ndarray
            ), f"Since image_masks is a numpy array, background image masks must also be a numpy array but got {type(background_image_masks)}."
            self._background_image_masks = background_image_masks

        if background_response_traces is not None:
            assert (
                background_image_masks is not None
            ), "Background image masks must be provided if background response traces are provided."
            self._background_response_traces = background_response_traces
            for name, trace in self._background_response_traces.items():
                assert trace.shape[-1] == self._background_image_masks.shape[-1], (
                    "Inconsistency between background image masks and background response traces. "
                    "Background image masks must be (num_rows, num_columns, num_background_components), "
                    "background response traces must be (num_frames, num_background_components)"
                )

    def get_image_size(self):
        return self._image_size

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_roi_ids(self):
        return self._roi_ids

    def get_num_rois(self):
        return self._num_rois

    def get_accepted_roi_ids(self) -> list:
        return self._accepted_roi_ids

    def get_rejected_roi_ids(self) -> list:
        return self._rejected_roi_ids

    def get_roi_locations(self, roi_ids=None):
        roi_indices = self.get_roi_indices(roi_ids=roi_ids)
        return self._roi_locations[:, roi_indices]

    def get_roi_image_masks(self, roi_ids=None) -> np.ndarray:
        if roi_ids is None:
            return self._image_masks
        roi_indices = self.get_roi_indices(roi_ids=roi_ids)
        return self._image_masks[:, :, roi_indices]

    def get_roi_response_traces(
        self,
        names: Optional[list[str]] = None,
        roi_ids: Optional[ArrayType] = None,
        start_frame: Optional[IntType] = None,
        end_frame: Optional[IntType] = None,
    ) -> dict:
        names = names if names is not None else list(self._roi_response_traces.keys())
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else self.get_num_frames()

        roi_indices = self.get_roi_indices(roi_ids=roi_ids)
        roi_response_traces = {
            name: self._roi_response_traces[name][start_frame:end_frame, roi_indices] for name in names
        }
        return roi_response_traces

    def get_background_ids(self) -> list:
        return self._background_ids

    def get_num_background_components(self) -> int:
        return self._num_background_components

    def get_background_image_masks(self, background_ids=None) -> np.ndarray:
        if background_ids is None:
            return self._background_image_masks
        all_ids = self.get_background_ids()
        background_indices = [all_ids.index(i) for i in background_ids]
        return self._background_image_masks[:, :, background_indices]

    def get_background_response_traces(
        self,
        names: Optional[list[str]] = None,
        background_ids: Optional[ArrayType] = None,
        start_frame: Optional[IntType] = None,
        end_frame: Optional[IntType] = None,
    ) -> dict:
        names = names if names is not None else list(self._background_response_traces.keys())
        all_ids = self.get_background_ids()
        background_ids = background_ids if background_ids is not None else all_ids
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else self.get_num_frames()

        background_indices = [all_ids.index(i) for i in background_ids]
        background_response_traces = {
            name: self._background_response_traces[name][start_frame:end_frame, background_indices] for name in names
        }
        return background_response_traces

    def get_summary_images(self, names: Optional[list[str]] = None) -> dict:
        names = names if names is not None else list(self._summary_images.keys())
        summary_images = {name: self._summary_images[name] for name in names}
        return summary_images
