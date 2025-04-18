"""Base class definition for volumetric imaging extractors."""

from typing import Tuple, List, Iterable, Optional
import warnings
import numpy as np

from .extraction_tools import ArrayType, DtypeType
from .imagingextractor import ImagingExtractor


class VolumetricImagingExtractor(ImagingExtractor):
    """Class to combine multiple ImagingExtractor objects by depth plane."""

    extractor_name = "VolumetricImaging"
    installatiuon_mesage = ""

    def __init__(self, imaging_extractors: List[ImagingExtractor]):
        """Initialize a VolumetricImagingExtractor object from a list of ImagingExtractors.

        Parameters
        ----------
        imaging_extractors: list of ImagingExtractor
            list of imaging extractor objects
        """
        super().__init__()
        assert isinstance(imaging_extractors, list), "Enter a list of ImagingExtractor objects as argument"
        assert all(isinstance(imaging_extractor, ImagingExtractor) for imaging_extractor in imaging_extractors)
        self._check_consistency_between_imaging_extractors(imaging_extractors)
        self._imaging_extractors = imaging_extractors
        self._num_planes = len(imaging_extractors)

    @staticmethod
    def _check_consistency_between_imaging_extractors(imaging_extractors: List[ImagingExtractor]):
        """Check that essential properties are consistent between extractors so that they can be combined appropriately.

        Parameters
        ----------
        imaging_extractors: list of ImagingExtractor
            list of imaging extractor objects

        Raises
        ------
        AssertionError
            If any of the properties are not consistent between extractors.

        Notes
        -----
        This method checks the following properties:
            - sampling frequency
            - image size
            - number of channels
            - channel names
            - data type
            - num_frames
        """
        properties_to_check = dict(
            get_sampling_frequency="The sampling frequency",
            get_image_shape="The shape of a frame",
            get_image_size="The size of a frame",
            get_num_channels="The number of channels",
            get_channel_names="The name of the channels",
            get_dtype="The data type",
            get_num_samples="The number of samples",
        )
        for method, property_message in properties_to_check.items():
            values = [getattr(extractor, method)() for extractor in imaging_extractors]
            unique_values = set(tuple(v) if isinstance(v, Iterable) else v for v in values)
            assert (
                len(unique_values) == 1
            ), f"{property_message} is not consistent over the files (found {unique_values})."

    def get_video(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> np.ndarray:
        """Get the video frames.

        Parameters
        ----------
        start_frame: int, optional
            Start frame index (inclusive).
        end_frame: int, optional
            End frame index (exclusive).

        Returns
        -------
        video: numpy.ndarray
            The 3D video frames (num_frames, num_rows, num_columns, num_planes).
        """
        if start_frame is None:
            start_frame = 0
        elif start_frame < 0:
            start_frame = self.get_num_samples() + start_frame
        elif start_frame >= self.get_num_samples():
            raise ValueError(
                f"start_frame {start_frame} is greater than or equal to the number of samples {self.get_num_samples()}"
            )
        if end_frame is None:
            end_frame = self.get_num_samples()
        elif end_frame < 0:
            end_frame = self.get_num_samples() + end_frame
        elif end_frame > self.get_num_samples():
            raise ValueError(f"end_frame {end_frame} is greater than the number of samples {self.get_num_samples()}")
        if end_frame <= start_frame:
            raise ValueError(f"end_frame {end_frame} is less than or equal to start_frame {start_frame}")

        video = np.zeros((end_frame - start_frame, *self.get_image_size()), self.get_dtype())
        for i, imaging_extractor in enumerate(self._imaging_extractors):
            video[..., i] = imaging_extractor.get_video(start_frame, end_frame)
        return video

    def get_frames(self, frame_idxs: ArrayType) -> np.ndarray:
        """Get specific video frames from indices (not necessarily continuous).

        Parameters
        ----------
        frame_idxs: array-like
            Indices of frames to return.

        Returns
        -------
        frames: numpy.ndarray
            The 3D video frames (num_rows, num_columns, num_planes).
        """
        if isinstance(frame_idxs, int):
            frame_idxs = [frame_idxs]
        for frame_idx in frame_idxs:
            if frame_idx < -1 * self.get_num_samples() or frame_idx >= self.get_num_samples():
                raise ValueError(f"frame_idx {frame_idx} is out of bounds")

        # Note np.all([]) returns True so not all(np.diff(frame_idxs) == 1) returns False if frame_idxs is a single int
        if not all(np.diff(frame_idxs) == 1):
            frames = np.zeros((len(frame_idxs), *self.get_image_size()), self.get_dtype())
            for i, imaging_extractor in enumerate(self._imaging_extractors):
                frames[..., i] = imaging_extractor.get_frames(frame_idxs)
            return frames
        else:
            return self.get_video(start_frame=frame_idxs[0], end_frame=frame_idxs[-1] + 1)

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._imaging_extractors[0].get_image_shape()

    def get_image_size(self) -> Tuple:
        """Get the size of a single frame.

        Returns
        -------
        image_size: tuple
            The size of a single frame (num_rows, num_columns, num_planes).

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_image_shape() instead for consistent behavior across all extractors.
        """
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        image_size = (*self._imaging_extractors[0].get_image_size(), self.get_num_planes())
        return image_size

    def get_num_planes(self) -> int:
        """Get the number of depth planes.

        Returns
        -------
        _num_planes: int
            The number of depth planes.
        """
        return self._num_planes

    def get_num_samples(self) -> int:
        return self._imaging_extractors[0].get_num_samples()

    def get_num_frames(self) -> int:
        """Get the number of frames in the video.

        Returns
        -------
        num_frames: int
            Number of frames in the video.

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_num_samples() instead.
        """
        warnings.warn(
            "get_num_frames() is deprecated and will be removed in or after September 2025. "
            "Use get_num_samples() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_num_samples()

    def get_sampling_frequency(self) -> float:
        return self._imaging_extractors[0].get_sampling_frequency()

    def get_channel_names(self) -> list:
        return self._imaging_extractors[0].get_channel_names()

    def get_num_channels(self) -> int:
        warnings.warn(
            "get_num_channels() is deprecated and will be removed in or after August 2025.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._imaging_extractors[0].get_num_channels()

    def get_dtype(self) -> DtypeType:
        return self._imaging_extractors[0].get_dtype()

    def depth_slice(self, start_plane: Optional[int] = None, end_plane: Optional[int] = None):
        """Return a new VolumetricImagingExtractor ranging from the start_plane to the end_plane."""
        start_plane = start_plane if start_plane is not None else 0
        end_plane = end_plane if end_plane is not None else self._num_planes
        assert (
            0 <= start_plane < self._num_planes
        ), f"'start_plane' ({start_plane}) must be greater than 0 and smaller than the number of planes ({self._num_planes})."
        assert (
            start_plane < end_plane <= self._num_planes
        ), f"'end_plane' ({end_plane}) must be greater than 'start_plane' ({start_plane}) and smaller than or equal to the number of planes ({self._num_planes})."

        return DepthSliceVolumetricImagingExtractor(parent_extractor=self, start_plane=start_plane, end_plane=end_plane)

    def frame_slice(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None):
        """Return a new VolumetricImagingExtractor with a subset of frames."""
        raise NotImplementedError(
            "frame_slice is not implemented for VolumetricImagingExtractor due to conflicts with get_video()."
        )


class DepthSliceVolumetricImagingExtractor(VolumetricImagingExtractor):
    """Class to get a lazy depth slice.

    This class can only be used for volumetric imaging data.
    Do not use this class directly but use `.depth_slice(...)` on a VolumetricImagingExtractor object.
    """

    extractor_name = "DepthSliceVolumetricImagingExtractor"
    is_writable = True
    installation_mesg = ""

    def __init__(
        self,
        parent_extractor: VolumetricImagingExtractor,
        start_plane: Optional[int] = None,
        end_plane: Optional[int] = None,
    ):
        """Initialize a VolumetricImagingExtractor whose plane(s) subset the parent.

        Subset is exclusive on the right bound, that is, the plane indices of this VolumetricImagingExtractor range over
        [0, ..., end_plane-start_plane-1].

        Parameters
        ----------
        parent_extractor : VolumetricImagingExtractor
            The VolumetricImagingExtractor object to subset the planes of.
        start_plane : int, optional
            The left bound of the depth to subset.
            The default is the first plane of the parent.
        end_plane : int, optional
            The right bound of the depth, exclusively, to subset.
            The default is the last plane of the parent.
        """
        super().__init__(imaging_extractors=parent_extractor._imaging_extractors[start_plane:end_plane])
