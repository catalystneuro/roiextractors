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

from abc import abstractmethod
from typing import Union, Optional, Tuple, Iterable, List, get_args

import numpy as np
from numpy.typing import ArrayLike

from .baseextractor import BaseExtractor
from .tools.typing import ArrayType, IntType, FloatType


class SegmentationExtractor(BaseExtractor):
    """Abstract segmentation extractor class.

    An abstract class that contains all the meta-data and output data from
    the ROI segmentation operation when applied to the pre-processed data.
    It also contains methods to read from and write to various data formats
    output from the processing pipelines like SIMA, CaImAn, Suite2p, CNMF-E.
    All the methods with @abstract decorator have to be defined by the
    format specific classes that inherit from this.
    """

    @abstractmethod
    def get_roi_ids(self) -> list:
        """Get the list of ROI ids.

        Returns
        -------
        roi_ids: list
            List of roi ids.
        """
        pass

    def get_roi_indices(self, roi_ids: Optional[list] = None) -> list:
        """Get the list of ROI indices corresponding to the ROI ids.

        Parameters
        ----------
        roi_ids: list
            List of roi ids. If None, all roi indices are returned.

        Returns
        -------
        roi_indices: list
            List of roi indices.
        """
        if roi_ids is None:
            return list(range(self.get_num_rois()))
        all_roi_ids = self.get_roi_ids()
        roi_indices = [all_roi_ids.index(roi_id) for roi_id in roi_ids]
        return roi_indices

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
    def get_accepted_roi_ids(self) -> list:
        """Get a list of accepted ROI ids.

        Returns
        -------
        accepted_roi_ids: list
            List of accepted ROI ids.
        """
        pass

    @abstractmethod
    def get_rejected_roi_ids(self) -> list:
        """Get a list of rejected ROI ids.

        Returns
        -------
        rejected_roi_ids: list
            List of rejected ROI ids.
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
            Columns 1 and 2 are the row and column coordinates of the pixel, while the third column represents the weight of
            the pixel.
        """
        return convert_image_masks_to_pixel_masks(image_masks=self.get_roi_image_masks(roi_ids=roi_ids))

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
            List of names of the traces to retrieve. Must be one of {'raw', 'dff', 'deconvolved', 'denoised'}. If None, all traces are returned.
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
    def get_background_ids(self) -> list:
        """Get the list of background components ids.

        Returns
        -------
        background_components_ids: list
            List of background components ids.
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
        return convert_image_masks_to_pixel_masks(self.get_background_image_masks(background_ids=background_ids))

    @abstractmethod
    def get_background_response_traces(
        self,
        names: Optional[list[str]] = None,
        background_ids: Optional[ArrayType] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> dict:
        """Get the background response traces.

        Parameters
        ----------
        names: list
            List of names of the traces to retrieve. Must be one of {'background'}. If None, all traces are returned.
        background_ids: array_like
            A list or 1D array of ids of the background components. Length is the number of background components requested. If None, all background components are returned.
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

    def frame_slice(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None):
        """Return a new ImagingExtractor ranging from the start_frame to the end_frame.

        Parameters
        ----------
        start_frame: int, optional
            Start frame index (inclusive).
        end_frame: int, optional
            End frame index (exclusive).

        Returns
        -------
        imaging: FrameSliceImagingExtractor
            The sliced ImagingExtractor object.
        """
        num_frames = self.get_num_frames()
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else num_frames
        assert 0 <= start_frame < num_frames, f"'start_frame' must be in [0, {num_frames}) but got {start_frame}"
        assert 0 < end_frame <= num_frames, f"'end_frame' must be in (0, {num_frames}] but got {end_frame}"
        assert (
            start_frame <= end_frame
        ), f"'start_frame' ({start_frame}) must be less than or equal to 'end_frame' ({end_frame})"
        assert isinstance(start_frame, get_args(IntType)), "'start_frame' must be an integer"
        assert isinstance(end_frame, get_args(IntType)), "'end_frame' must be an integer"

        return FrameSliceSegmentationExtractor(parent_segmentation=self, start_frame=start_frame, end_frame=end_frame)


class FrameSliceSegmentationExtractor(SegmentationExtractor):
    """Class to get a lazy frame slice.

    Do not use this class directly but use `.frame_slice(...)`
    """

    extractor_name = "FrameSliceSegmentationExtractor"
    is_writable = True

    def __init__(
        self,
        parent_segmentation: SegmentationExtractor,
        start_frame: int,
        end_frame: int,
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
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._num_frames = self._end_frame - self._start_frame

        super().__init__()
        if getattr(self._parent_segmentation, "_times") is not None:
            self._times = self._parent_segmentation._times[start_frame:end_frame]

    def get_image_size(self) -> Tuple[int, int]:
        return self._parent_segmentation.get_image_size()

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return self._parent_segmentation.get_sampling_frequency()

    def get_roi_ids(self) -> list:
        return self._parent_segmentation.get_roi_ids()

    def get_num_rois(self) -> int:
        return self._parent_segmentation.get_num_rois()

    def get_accepted_roi_ids(self) -> list:
        return self._parent_segmentation.get_accepted_roi_ids()

    def get_rejected_roi_ids(self) -> list:
        return self._parent_segmentation.get_rejected_roi_ids()

    def get_roi_locations(self, roi_ids=None) -> np.ndarray:
        return self._parent_segmentation.get_roi_locations(roi_ids=roi_ids)

    def get_roi_image_masks(self, roi_ids=None) -> np.ndarray:
        return self._parent_segmentation.get_roi_image_masks(roi_ids=roi_ids)

    def get_roi_response_traces(
        self,
        names: Optional[list[str]] = None,
        roi_ids: Optional[ArrayType] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> dict:
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else self.get_num_frames()
        start_frame_shifted = start_frame + self._start_frame
        end_frame_shifted = end_frame + self._start_frame
        return self._parent_segmentation.get_roi_response_traces(
            names=names,
            roi_ids=roi_ids,
            start_frame=start_frame_shifted,
            end_frame=end_frame_shifted,
        )

    def get_background_ids(self) -> list:
        return self._parent_segmentation.get_background_ids()

    def get_num_background_components(self) -> int:
        return self._parent_segmentation.get_num_background_components()

    def get_background_image_masks(self, background_ids=None) -> np.ndarray:
        return self._parent_segmentation.get_background_image_masks(background_ids=background_ids)

    def get_background_response_traces(
        self,
        names: Optional[list[str]] = None,
        background_ids: Optional[ArrayType] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> dict:
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else self.get_num_frames()
        start_frame_shifted = start_frame + self._start_frame
        end_frame_shifted = end_frame + self._start_frame
        return self._parent_segmentation.get_background_response_traces(
            names=names,
            background_ids=background_ids,
            start_frame=start_frame_shifted,
            end_frame=end_frame_shifted,
        )

    def get_summary_images(self, names: Optional[list[str]] = None) -> dict:
        return self._parent_segmentation.get_summary_images(names=names)


def convert_image_masks_to_pixel_masks(image_masks: np.ndarray) -> list:
    """Convert image masks to pixel masks.

    Pixel masks are an alternative data format for storage of image masks which relies on the sparsity of the images.
    The location and weight of each non-zero pixel is stored for each mask.

    Parameters
    ----------
    image_masks: numpy.ndarray
        Dense representation of the ROIs with shape (number_of_rows, number_of_columns, number_of_rois).

    Returns
    -------
    pixel_masks: list
        List of length number of rois, each element is a 2-D array with shape (number_of_non_zero_pixels, 3).
        Columns 1 and 2 are the row and column coordinates of the pixel, while the third column represents the weight of
        the pixel.
    """
    pixel_masks = []
    for i in range(image_masks.shape[2]):
        image_mask = image_masks[:, :, i]
        locs = np.where(image_mask > 0)
        pix_values = image_mask[image_mask > 0]
        pixel_masks.append(np.vstack((locs[0], locs[1], pix_values)).T)
    return pixel_masks


def convert_pixel_masks_to_image_masks(pixel_masks: list[np.ndarray], image_shape: tuple) -> np.ndarray:
    """Convert pixel masks to image masks.

    Parameters
    ----------
    pixel_masks: list[np.ndarray]
        List of pixel mask arrays (number_of_non_zero_pixels X 3) for each ROI.
    image_shape: tuple
        Shape of the image (number_of_rows, number_of_columns).

    Returns
    -------
    image_masks: np.ndarray
        Dense representation of the ROIs with shape (number_of_rows, number_of_columns, number_of_rois).
    """
    shape = (*image_shape, len(pixel_masks))
    image_masks = np.zeros(shape=shape)
    for i, pixel_mask in enumerate(pixel_masks):
        for row, column, wt in pixel_mask:
            image_masks[int(row), int(column), i] = wt
    return image_masks


def get_default_roi_locations_from_image_masks(image_masks: np.ndarray) -> np.ndarray:
    """Calculate the default ROI locations from given image masks.

    This function takes a 3D numpy array of image masks and computes the coordinates (row, column)
    of the maximum values in each 2D mask. In the case of a tie, the integer median of the coordinates is used.
    The result is a 2D numpy array where each column represents the (row, column) coordinates of the ROI for
    each mask.

    Parameters
    ----------
    image_masks : np.ndarray
        A 3D numpy array of shape (height, width, num_rois) containing the image masks.

    Returns
    -------
    np.ndarray
        A 2D numpy array of shape (2, num_rois) where each column contains the
        (row, column) coordinates of the ROI for each mask.
    """
    num_rois = image_masks.shape[2]
    roi_locations = np.zeros([2, num_rois], dtype="int")
    for i in range(num_rois):
        image_mask = image_masks[:, :, i]
        max_value_indices = np.where(image_mask == np.amax(image_mask))
        roi_locations[:, i] = np.array([int(np.median(max_value_indices[0])), int(np.median(max_value_indices[1]))]).T
    return roi_locations
