from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, Iterable, List

import numpy as np
from numpy.typing import ArrayLike

from .extraction_tools import ArrayType, IntType, FloatType
from .extraction_tools import _pixel_mask_extractor


class SegmentationExtractor(ABC):
    """
    An abstract class that contains all the meta-data and output data from
    the ROI segmentation operation when applied to the pre-processed data.
    It also contains methods to read from and write to various data formats
    output from the processing pipelines like SIMA, CaImAn, Suite2p, CNNM-E.
    All the methods with @abstract decorator have to be defined by the
    format specific classes that inherit from this.
    """

    def __init__(self):
        self._sampling_frequency = None
        self._times = None
        self._channel_names = ["OpticalChannel"]
        self._num_planes = 1
        self._roi_response_raw = None
        self._roi_response_dff = None
        self._roi_response_neuropil = None
        self._roi_response_deconvolved = None
        self._image_correlation = None
        self._image_mean = None

    @abstractmethod
    def get_accepted_list(self) -> list:
        """
        The ids of the ROIs which are accepted after manual verification of
        ROIs.

        Returns
        -------
        accepted_list: list
            List of accepted ROIs
        """
        pass

    @abstractmethod
    def get_rejected_list(self) -> list:
        """
        The ids of the ROIs which are rejected after manual verification of
        ROIs.

        Returns
        -------
        accepted_list: list
            List of rejected ROIs
        """
        pass

    def get_num_frames(self) -> int:
        """This function returns the number of frames in the recording.

        Returns
        -------
        num_of_frames: int
            Number of frames in the recording (duration of recording).
        """
        for trace in self.get_traces_dict().values():
            if trace is not None and len(trace.shape) > 0:
                return trace.shape[0]

    def get_roi_locations(self, roi_ids=None) -> np.ndarray:
        """
        Returns the locations of the Regions of Interest

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

        Returns
        ------
        roi_locs: numpy.ndarray
            2-D array: 2 X no_ROIs. The pixel ids (x,y) where the centroid of the ROI is.
        """
        if roi_ids is None:
            roi_idx_ = list(range(self.get_num_rois()))
        else:
            all_ids = self.get_roi_ids()
            roi_idx_ = [all_ids.index(i) for i in roi_ids]
        roi_location = np.zeros([2, len(roi_idx_)], dtype="int")
        for c, i in enumerate(roi_idx_):
            image_mask = self.get_roi_image_masks(roi_ids=[i])
            temp = np.where(image_mask == np.amax(image_mask))
            roi_location[:, c] = np.array([np.median(temp[0]), np.median(temp[1])]).T
        return roi_location

    def get_roi_ids(self) -> list:
        """Returns the list of ROI ids.
        Returns
        -------
        roi_ids: list
            List of roi ids.
        """
        return list(range(self.get_num_rois()))

    def get_roi_image_masks(self, roi_ids=None) -> np.ndarray:
        """Returns the image masks extracted from segmentation algorithm.

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

        Returns
        -------
        image_masks: numpy.ndarray
            3-D array(val 0 or 1): image_height X image_width X length(roi_ids)
        """
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            all_ids = self.get_roi_ids()
            roi_idx_ = [all_ids.index(i) for i in roi_ids]
        return np.stack([self._image_masks[:, :, k] for k in roi_idx_], 2)

    def get_roi_pixel_masks(self, roi_ids=None) -> np.array:
        """
        Returns the weights applied to each of the pixels of the mask.

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

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

    @abstractmethod
    def get_image_size(self) -> ArrayType:
        """
        Frame size of movie ( x and y size of image).

        Returns
        -------
        no_rois: array_like
            2-D array: image y x image x
        """
        pass

    def frame_slice(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None):
        """Return a new SegmentationExtractor ranging from the start_frame to the end_frame."""
        return FrameSliceSegmentationExtractor(parent_segmentation=self, start_frame=start_frame, end_frame=end_frame)

    def get_traces(self, roi_ids=None, start_frame=None, end_frame=None, name="raw"):
        """
        Return RoiResponseSeries
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

    def get_traces_dict(self):
        """
        Returns traces as a dictionary with key as the name of the ROiResponseSeries
        Returns
        -------
        _roi_response_dict: dict
            dictionary with key, values representing different types of RoiResponseSeries
            Flourescence, Neuropil, Deconvolved, Background etc
        """
        return dict(
            raw=self._roi_response_raw,
            dff=self._roi_response_dff,
            neuropil=self._roi_response_neuropil,
            deconvolved=self._roi_response_deconvolved,
        )

    def get_images_dict(self):
        """
        Returns traces as a dictionary with key as the name of the ROiResponseSeries
        Returns
        -------
        _roi_response_dict: dict
            dictionary with key, values representing different types of Images used in segmentation:
            Mean, Correlation image
        """
        return dict(mean=self._image_mean, correlation=self._image_correlation)

    def get_image(self, name="correlation"):
        """
        Return specific images: mean or correlation
        Parameters
        ----------
        name:str
            name of the type of image to retrieve
        Returns
        -------
        images: np.ndarray
        """
        if name not in self.get_images_dict():
            raise ValueError(f"could not find {name} image, enter one of {list(self.get_images_dict().keys())}")
        return self.get_images_dict().get(name)

    def get_sampling_frequency(self):
        """This function returns the sampling frequency in units of Hz.

        Returns
        -------
        sampling_frequency: float
            Sampling frequency of the recording in Hz.
        """
        if self._sampling_frequency is not None:
            return float(self._sampling_frequency)

        return self._sampling_frequency

    def get_num_rois(self):
        """Returns total number of Regions of Interest in the acquired images.

        Returns
        -------
        num_rois: int
            The number of ROIs extracted.
        """
        for trace in self.get_traces_dict().values():
            if trace is not None and len(trace.shape) > 0:
                return trace.shape[1]

    def get_channel_names(self):
        """
        Names of channels in the pipeline
        Returns
        -------
        _channel_names: list
            names of channels (str)
        """
        return self._channel_names

    def get_num_channels(self):
        """
        Number of channels in the pipeline
        Returns
        -------
        num_of_channels: int
        """
        return len(self._channel_names)

    def get_num_planes(self):
        """
        Returns the default number of planes of imaging for the segmentation extractor.
        Defaults to 1 for all but the MultiSegmentationExtractor
        Returns
        -------
        self._num_planes: int
        """
        return self._num_planes

    def set_times(self, times: ArrayType):
        """Sets the recording times in seconds for each frame.

        Parameters
        ----------
        times: array-like
            The times in seconds for each frame
        """
        assert len(times) == self.get_num_frames(), "'times' should have the same length of the number of frames!"
        self._times = np.array(times, dtype=np.float64)

    def has_time_vector(self) -> bool:
        """Detect if the SegmentationExtractor has a time vector set or not."""
        return self._times is not None

    def frame_to_time(self, frames: Union[IntType, ArrayType]) -> Union[FloatType, ArrayType]:
        """Returns the timing of frames in unit of seconds.

        Parameters
        ----------
        frame_indices: int or array-like
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

    @staticmethod
    def write_segmentation(segmentation_extractor, save_path, overwrite=False):
        """
        Static method to write recording back to the native format.

        Parameters
        ----------
        segmentation_extractor: [SegmentationExtractor, MultiSegmentationExtractor]
            The EXTRACT segmentation object from which an EXTRACT native format
            file has to be generated.
        save_path: str
            path to save the native format.
        overwrite: bool
            If True, the file is overwritten if existing (default False)
        """
        raise NotImplementedError


class FrameSliceSegmentationExtractor(SegmentationExtractor):
    """
    Class to get a lazy frame slice.

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

    def get_traces_dict(self):
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

    def get_num_rois(self):
        return self._parent_segmentation.get_num_rois()

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

    def get_num_planes(self):
        return self._parent_segmentation.get_num_planes()

    def get_roi_pixel_masks(self, roi_ids: Optional[ArrayLike] = None) -> List[np.ndarray]:
        return self._parent_segmentation.get_roi_pixel_masks(roi_ids=roi_ids)
