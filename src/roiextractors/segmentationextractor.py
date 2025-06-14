"""Base segmentation extractors.

Classes
-------
SegmentationExtractor
    Abstract class that contains all the meta-data and output data from the ROI segmentation operation when applied to
    the pre-processed data. It also contains methods to read from and write to various data formats output from the
    processing pipelines like SIMA, CaImAn, Suite2p, CNMF-E.
FrameSliceSegmentationExtractor
    Class to get a lazy frame slice.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, Iterable, List
import warnings

import numpy as np
from numpy.typing import ArrayLike

from .extraction_tools import ArrayType, IntType, FloatType
from .extraction_tools import _pixel_mask_extractor


class SegmentationExtractor(ABC):
    """Abstract segmentation extractor class.

    An abstract class that contains all the meta-data and output data from
    the ROI segmentation operation when applied to the pre-processed data.
    It also contains methods to read from and write to various data formats
    output from the processing pipelines like SIMA, CaImAn, Suite2p, CNMF-E.
    All the methods with @abstract decorator have to be defined by the
    format specific classes that inherit from this.
    """

    def __init__(self):
        """Create a new SegmentationExtractor for a specific data format (unique to each child SegmentationExtractor)."""
        self._sampling_frequency = None
        self._times = None
        self._channel_names = ["OpticalChannel"]
        self._num_planes = 1
        self._roi_response_raw = None
        self._roi_response_dff = None
        self._roi_response_neuropil = None
        self._roi_response_denoised = None
        self._roi_response_deconvolved = None
        self._image_correlation = None
        self._image_mean = None
        self._image_mask = None

    @abstractmethod
    def get_accepted_list(self) -> list:
        """Get a list of accepted ROI ids.

        Returns
        -------
        accepted_list: list
            List of accepted ROI ids.
        """
        pass

    @abstractmethod
    def get_rejected_list(self) -> list:
        """Get a list of rejected ROI ids.

        Returns
        -------
        rejected_list: list
            List of rejected ROI ids.
        """
        pass

    @abstractmethod
    def get_frame_shape(self) -> ArrayType:
        """Get frame size of movie (height, width).

        Returns
        -------
        frame_shape: array_like
            2-D array: image height x image width
        """
        pass

    def get_image_size(self) -> ArrayType:
        """Get frame size of movie (height, width).

        .. deprecated:: on or after January 2026
           Use :meth:`get_frame_shape` instead.

        Returns
        -------
        no_rois: array_like
            2-D array: image height x image width
        """
        warnings.warn(
            "get_image_size is deprecated and will be removed on or after January 2026. "
            "Use get_frame_shape instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_frame_shape()

    def get_num_samples(self) -> int:
        """Get the number of samples in the recording (duration of recording).

        Returns
        -------
        num_samples: int
            Number of samples in the recording.
        """
        for trace in self.get_traces_dict().values():
            if trace is not None and len(trace.shape) > 0:
                return trace.shape[0]

    def get_num_frames(self) -> int:
        """Get the number of frames in the recording (duration of recording).

        .. deprecated:: on or after January 2026
           Use :meth:`get_num_samples` instead.

        Returns
        -------
        num_frames: int
            Number of frames in the recording.
        """
        warnings.warn(
            "get_num_frames is deprecated and will be removed on or after January 2026. "
            "Use get_num_samples instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_num_samples()

    def get_roi_locations(self, roi_ids=None) -> np.ndarray:
        """Get the locations of the Regions of Interest (ROIs).

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs requested.

        Returns
        -------
        roi_locs: numpy.ndarray
            2-D array: 2 X no_ROIs. The pixel ids (x,y) where the centroid of the ROI is.
        """
        if roi_ids is None:
            roi_ids = self.get_roi_ids()

        roi_location = np.zeros([2, len(roi_ids)], dtype="int")
        for roi_index, roi_id in enumerate(roi_ids):
            image_mask = self.get_roi_image_masks(roi_ids=[roi_id])
            temp = np.where(image_mask == np.amax(image_mask))
            roi_location[:, roi_index] = np.array([np.median(temp[0]), np.median(temp[1])]).T
        return roi_location

    def get_roi_ids(self) -> list:
        """Get the list of ROI ids.

        Returns
        -------
        roi_ids: list
            List of roi ids.
        """
        return list(range(self.get_num_rois()))

    def get_roi_image_masks(self, roi_ids=None) -> np.ndarray:
        """Get the image masks extracted from segmentation algorithm.

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs requested.

        Returns
        -------
        image_masks: numpy.ndarray
            3-D array(val 0 or 1): image_height X image_width X length(roi_ids)
        """
        if roi_ids is None:
            roi_indices = range(self.get_num_rois())
        else:
            all_roi_ids = self.get_roi_ids()
            roi_indices = [all_roi_ids.index(roi_id) for roi_id in roi_ids]

        return np.stack([self._image_masks[:, :, k] for k in roi_indices], 2)

    def get_roi_pixel_masks(self, roi_ids=None) -> np.array:
        """Get the weights applied to each of the pixels of the mask.

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs requested.

        Returns
        -------
        pixel_masks: list
            List of length number of rois, each element is a 2-D array with shape (number_of_non_zero_pixels, 3).
            Columns 1 and 2 are the x and y coordinates of the pixel, while the third column represents the weight of
            the pixel.
        """
        if roi_ids is None:
            roi_ids = range(self.get_num_rois())

        return _pixel_mask_extractor(self.get_roi_image_masks(roi_ids=roi_ids), roi_ids)

    def get_background_ids(self) -> list:
        """Get the list of background components ids.

        Returns
        -------
        background_components_ids: list
            List of background components ids.
        """
        return list(range(self.get_num_background_components()))

    def get_background_image_masks(self, background_ids=None) -> np.ndarray:
        """Get the background image masks extracted from segmentation algorithm.

        Parameters
        ----------
        background_ids: array_like
            A list or 1D array of ids of the background components. Length is the number of background components requested.

        Returns
        -------
        background_image_masks: numpy.ndarray
            3-D array(val 0 or 1): image_height X image_width X length(background_ids)
        """
        if background_ids is None:
            background_ids_ = range(self.get_num_background_components())
        else:
            all_ids = self.get_background_ids()
            background_ids_ = [all_ids.index(i) for i in background_ids]
        return np.stack([self._background_image_masks[:, :, k] for k in background_ids_], 2)

    def get_background_pixel_masks(self, background_ids=None) -> np.array:
        """Get the weights applied to each of the pixels of the mask.

        Parameters
        ----------
        background_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs requested.

        Returns
        -------
        pixel_masks: list
            List of length number of rois, each element is a 2-D array with shape (number_of_non_zero_pixels, 3).
            Columns 1 and 2 are the x and y coordinates of the pixel, while the third column represents the weight of
            the pixel.
        """
        if background_ids is None:
            background_ids = range(self.get_num_background_components())

        return _pixel_mask_extractor(self.get_background_image_masks(background_ids=background_ids), background_ids)

    def frame_slice(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None):
        """Return a new SegmentationExtractor ranging from the start_frame to the end_frame.

        Parameters
        ----------
        start_frame: int
            The starting frame of the new SegmentationExtractor.
        end_frame: int
            The ending frame of the new SegmentationExtractor.

        Returns
        -------
        frame_slice_segmentation_extractor: FrameSliceSegmentationExtractor
            The frame slice segmentation extractor object.
        """
        return FrameSliceSegmentationExtractor(parent_segmentation=self, start_frame=start_frame, end_frame=end_frame)

    def get_traces(
        self,
        roi_ids: ArrayType = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        name: str = "raw",
    ) -> ArrayType:
        """Get the traces of each ROI specified by roi_ids.

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs requested.
        start_frame: int
            The starting frame of the trace.
        end_frame: int
            The ending frame of the trace.
        name: str
            The name of the trace to retrieve ex. 'raw', 'dff', 'neuropil', 'deconvolved'

        Returns
        -------
        traces: array_like
            2-D array (ROI x timepoints)
        """
        if name not in self.get_traces_dict():
            raise ValueError(f"traces for {name} not found, enter one of {list(self.get_traces_dict().keys())}")
        if roi_ids is not None:
            all_ids = self.get_roi_ids()
            roi_idxs = [all_ids.index(i) for i in roi_ids]
        traces = self.get_traces_dict().get(name)
        if traces is not None and len(traces.shape) != 0:
            idxs = slice(None) if roi_ids is None else roi_idxs
            return np.array(traces[start_frame:end_frame, :])[:, idxs]  # numpy fancy indexing is quickest

    def get_traces_dict(self) -> dict:
        """Get traces as a dictionary with key as the name of the ROiResponseSeries.

        Returns
        -------
        _roi_response_dict: dict
            dictionary with key, values representing different types of RoiResponseSeries:
                Raw Fluorescence, DeltaFOverF, Denoised, Neuropil, Deconvolved, Background, etc.
        """
        return dict(
            raw=self._roi_response_raw,
            dff=self._roi_response_dff,
            neuropil=self._roi_response_neuropil,
            deconvolved=self._roi_response_deconvolved,
            denoised=self._roi_response_denoised,
        )

    def get_images_dict(self) -> dict:
        """Get images as a dictionary with key as the name of the ROIResponseSeries.

        Returns
        -------
        _roi_image_dict: dict
            dictionary with key, values representing different types of Images used in segmentation:
                Mean, Correlation image
        """
        return dict(mean=self._image_mean, correlation=self._image_correlation)

    def get_image(self, name: str = "correlation") -> ArrayType:
        """Get specific images: mean or correlation.

        Parameters
        ----------
        name:str
            name of the type of image to retrieve

        Returns
        -------
        images: numpy.ndarray
        """
        if name not in self.get_images_dict():
            raise ValueError(f"could not find {name} image, enter one of {list(self.get_images_dict().keys())}")
        return self.get_images_dict().get(name)

    def get_sampling_frequency(self) -> float:
        """Get the sampling frequency in Hz.

        Returns
        -------
        sampling_frequency: float
            Sampling frequency of the recording in Hz.
        """
        if self._sampling_frequency is not None:
            return float(self._sampling_frequency)

        return self._sampling_frequency

    def get_num_rois(self) -> int:
        """Get total number of Regions of Interest (ROIs) in the acquired images.

        Returns
        -------
        num_rois: int
            The number of ROIs extracted.
        """
        for trace in self.get_traces_dict().values():
            if trace is not None and len(trace.shape) > 0:
                return trace.shape[1]

    def get_num_background_components(self) -> int:
        """Get total number of background components in the acquired images.

        Returns
        -------
        num_background_components: int
            The number of background components extracted.
        """
        if self._roi_response_neuropil is not None and len(self._roi_response_neuropil.shape) > 0:
            return self._roi_response_neuropil.shape[1]

    def get_channel_names(self) -> List[str]:
        """Get names of channels in the pipeline.

        Returns
        -------
        _channel_names: list
            names of channels (str)
        """
        return self._channel_names

    def get_num_channels(self) -> int:
        """Get number of channels in the pipeline.

        Returns
        -------
        num_of_channels: int
            number of channels
        """
        return len(self._channel_names)

    def get_num_planes(self) -> int:
        """Get the default number of planes of imaging for the segmentation extractor.

        Notes
        -----
        Defaults to 1 for all but the MultiSegmentationExtractor.

        Returns
        -------
        self._num_planes: int
            number of planes
        """
        return self._num_planes

    def set_times(self, times: ArrayType):
        """Set the recording times in seconds for each frame.

        Parameters
        ----------
        times: array-like
            The times in seconds for each frame

        Notes
        -----
        Operates on _times attribute of the SegmentationExtractor object.
        """
        assert len(times) == self.get_num_samples(), "'times' should have the same length of the number of samples!"
        self._times = np.array(times, dtype=np.float64)

    def has_time_vector(self) -> bool:
        """Detect if the SegmentationExtractor has a time vector set or not.

        Returns
        -------
        has_time_vector: bool
            True if the SegmentationExtractor has a time vector set, otherwise False.
        """
        return self._times is not None

    def frame_to_time(self, frames: Union[IntType, ArrayType]) -> Union[FloatType, ArrayType]:
        """Get the timing of frames in unit of seconds.

        Parameters
        ----------
        frames: int or array-like
            The frame or frames to be converted to times

        Returns
        -------
        times: float or array-like
            The corresponding times in seconds

        Deprecated
        ----------
        This method will be removed on or after January 2026.
        Use sample_indices_to_time() instead.
        """
        warnings.warn(
            "frame_to_time() is deprecated and will be removed on or after January 2026. "
            "Use sample_indices_to_time() instead.",
            FutureWarning,
            stacklevel=2,
        )
        if self._times is None:
            return frames / self.get_sampling_frequency()
        else:
            return self._times[frames]

    def sample_indices_to_time(self, sample_indices: Union[FloatType, np.ndarray]) -> Union[FloatType, np.ndarray]:
        """Convert user-inputted sample indices to times with units of seconds.

        Parameters
        ----------
        sample_indices: int or array-like
            The sample indices to be converted to times.

        Returns
        -------
        times: float or array-like
            The corresponding times in seconds.
        """
        # Default implementation
        if self._times is None:
            return sample_indices / self.get_sampling_frequency()
        else:
            return self._times[sample_indices]


class FrameSliceSegmentationExtractor(SegmentationExtractor):
    """Class to get a lazy frame slice.

    Do not use this class directly but use `.frame_slice(...)`
    """

    extractor_name = "FrameSliceSegmentationExtractor"

    def __init__(
        self,
        parent_segmentation: SegmentationExtractor,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ):
        """Create a new FrameSliceSegmentationExtractor from parent SegmentationExtractor.

        Parameters
        ----------
        parent_segmentation: SegmentationExtractor
            The parent SegmentationExtractor object.
        start_frame: int
            The starting frame of the new SegmentationExtractor.
        end_frame: int
            The ending frame of the new SegmentationExtractor.
        """
        self._parent_segmentation = parent_segmentation
        self._start_frame = start_frame or 0
        self._end_frame = end_frame or self._parent_segmentation.get_num_samples()
        self._num_frames = self._end_frame - self._start_frame

        if hasattr(self._parent_segmentation, "_image_masks"):  # otherwise, do not set attribute at all
            self._image_masks = self._parent_segmentation._image_masks

        if hasattr(self._parent_segmentation, "_background_image_masks"):  # otherwise, do not set attribute at all
            self._background_image_masks = self._parent_segmentation._background_image_masks

        parent_size = self._parent_segmentation.get_num_samples()
        if start_frame is None:
            start_frame = 0
        else:
            assert 0 <= start_frame < parent_size
        if end_frame is None:
            end_frame = parent_size
        else:
            assert 0 < end_frame <= parent_size
        assert end_frame > start_frame, "'start_frame' must be smaller than 'end_frame'!"

        super().__init__()
        if getattr(self._parent_segmentation, "_times") is not None:
            self._times = self._parent_segmentation._times[start_frame:end_frame]

    def get_accepted_list(self) -> list:
        return self._parent_segmentation.get_accepted_list()

    def get_rejected_list(self) -> list:
        return self._parent_segmentation.get_rejected_list()

    def get_traces(
        self,
        roi_ids: Optional[Iterable[int]] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        name: str = "raw",
    ) -> np.ndarray:
        start_frame = min(start_frame or 0, self._num_frames)
        end_frame = min(end_frame or self._num_frames, self._num_frames)
        return self._parent_segmentation.get_traces(
            roi_ids=roi_ids,
            start_frame=start_frame + self._start_frame,
            end_frame=end_frame + self._start_frame,
            name=name,
        )

    def get_traces_dict(self) -> dict:
        return {
            trace_name: self._parent_segmentation.get_traces(
                start_frame=self._start_frame, end_frame=self._end_frame, name=trace_name
            )
            for trace_name, trace in self._parent_segmentation.get_traces_dict().items()
        }

    def get_frame_shape(self) -> Tuple[int, int]:
        return tuple(self._parent_segmentation.get_frame_shape())

    def get_image_size(self) -> Tuple[int, int]:
        return tuple(self._parent_segmentation.get_frame_shape())

    def get_num_samples(self) -> int:
        return self._num_frames

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_num_rois(self) -> int:
        return self._parent_segmentation.get_num_rois()

    def get_num_background_components(self) -> int:
        return self._parent_segmentation.get_num_background_components()

    def get_images_dict(self) -> dict:
        return self._parent_segmentation.get_images_dict()

    def get_image(self, name="correlation"):
        return self._parent_segmentation.get_image(name=name)

    def get_sampling_frequency(self) -> float:
        return self._parent_segmentation.get_sampling_frequency()

    def get_channel_names(self) -> list:
        return self._parent_segmentation.get_channel_names()

    def get_num_channels(self) -> int:
        return self._parent_segmentation.get_num_channels()

    def get_num_planes(self) -> int:
        return self._parent_segmentation.get_num_planes()

    def get_roi_pixel_masks(self, roi_ids: Optional[ArrayLike] = None) -> List[np.ndarray]:
        return self._parent_segmentation.get_roi_pixel_masks(roi_ids=roi_ids)

    def get_background_pixel_masks(self, background_ids: Optional[ArrayLike] = None) -> List[np.ndarray]:
        return self._parent_segmentation.get_background_pixel_masks(background_ids=background_ids)
