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
        self._times = None

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
    def get_image_size(self) -> ArrayType:
        """Get frame size of movie (height, width).

        Returns
        -------
        no_rois: array_like
            2-D array: image height x image width
        """
        pass

    @abstractmethod
    def get_num_frames(self) -> int:
        """Get the number of frames in the recording (duration of recording).

        Returns
        -------
        num_frames: int
            Number of frames in the recording.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_roi_ids(self) -> list:
        """Get the list of ROI ids.

        Returns
        -------
        roi_ids: list
            List of roi ids.
        """
        pass

    @abstractmethod
    def get_roi_image_masks(self, roi_ids=None) -> np.ndarray:
        """Get the image masks extracted from segmentation algorithm.

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs requested. If None, image masks for all
            ROIs are returned.

        Returns
        -------
        image_masks: numpy.ndarray
            3-D array(val 0 or 1): image_height X image_width X length(roi_ids)
        """
        pass

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
            roi_ids = self.get_roi_ids()

        return _pixel_mask_extractor(self.get_roi_image_masks(roi_ids=roi_ids), roi_ids)

    @abstractmethod
    def get_background_ids(self) -> list:
        """Get the list of background components ids.

        Returns
        -------
        background_components_ids: list
            List of background components ids.
        """
        return list(range(self.get_num_background_components()))

    @abstractmethod
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
        pass

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

    @abstractmethod
    def get_roi_response_traces(
        self,
        names: Optional[list[str]] = None,
        roi_ids: Optional[ArrayType] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> dict:
        """Get the roi response traces.

        Parameters
        ----------
        names: list
            List of names of the traces to retrieve. Must be one of {'raw', 'dff', 'background', 'deconvolved', 'denoised'}. If None, all traces are returned.
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs requested. If None, all ROIs are returned.
        start_frame: int
            The starting frame of the trace. If None, the trace starts from the beginning.
        end_frame: int
            The ending frame of the trace. If None, the trace ends at the last frame.

        Returns
        -------
        traces: dict
            Dictionary of traces with key as the name of the trace and value as the trace.
        """
        pass

    @abstractmethod
    def get_summary_images(self, names: Optional[list[str]] = None) -> dict:
        """Get summary images.

        Parameters
        ----------
        names: list
            List of names of the images to retrieve. Must be one of {'mean', 'correlation'}. If None, all images are returned.

        Returns
        -------
        summary_images: dict
            Dictionary of summary images with key as the name of the image and value as the image.
        """
        pass

    @abstractmethod
    def get_sampling_frequency(self) -> float:
        """Get the sampling frequency in Hz.

        Returns
        -------
        sampling_frequency: float
            Sampling frequency of the recording in Hz.
        """
        pass

    @abstractmethod
    def get_num_rois(self) -> int:
        """Get total number of Regions of Interest (ROIs) in the acquired images.

        Returns
        -------
        num_rois: int
            The number of ROIs extracted.
        """
        pass

    @abstractmethod
    def get_num_background_components(self) -> int:
        """Get total number of background components in the acquired images.

        Returns
        -------
        num_background_components: int
            The number of background components extracted.
        """
        pass

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
        assert len(times) == self.get_num_frames(), "'times' should have the same length of the number of frames!"
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
        """
        if self._times is None:
            return frames / self.get_sampling_frequency()
        else:
            return self._times[frames]


class FrameSliceSegmentationExtractor(SegmentationExtractor):
    """Class to get a lazy frame slice.

    Do not use this class directly but use `.frame_slice(...)`
    """

    extractor_name = "FrameSliceSegmentationExtractor"
    is_writable = True

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
        self._end_frame = end_frame or self._parent_segmentation.get_num_frames()
        self._num_frames = self._end_frame - self._start_frame

        if hasattr(self._parent_segmentation, "_image_masks"):  # otherwise, do not set attribute at all
            self._image_masks = self._parent_segmentation._image_masks

        parent_size = self._parent_segmentation.get_num_frames()
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

    def get_image_size(self) -> Tuple[int, int]:
        return tuple(self._parent_segmentation.get_image_size())

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_num_rois(self) -> int:
        return self._parent_segmentation.get_num_rois()

    def get_images_dict(self) -> dict:
        return self._parent_segmentation.get_images_dict()

    def get_image(self, name="correlation"):
        return self._parent_segmentation.get_image(name=name)

    def get_sampling_frequency(self) -> float:
        return self._parent_segmentation.get_sampling_frequency()

    def get_roi_pixel_masks(self, roi_ids: Optional[ArrayLike] = None) -> List[np.ndarray]:
        return self._parent_segmentation.get_roi_pixel_masks(roi_ids=roi_ids)
