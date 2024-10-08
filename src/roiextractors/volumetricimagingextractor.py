"""Base class definition for volumetric imaging extractors."""

from typing import Tuple, List, Iterable, Optional
import numpy as np

from .tools.typing import ArrayType, DtypeType
from .imagingextractor import ImagingExtractor


class VolumetricImagingExtractor(ImagingExtractor):
    """Class to combine multiple ImagingExtractor objects by depth plane."""

    extractor_name = "VolumetricImaging"
    installed = True
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
            - data type
            - num_frames
        """
        properties_to_check = dict(
            get_sampling_frequency="The sampling frequency",
            get_image_size="The size of a frame",
            get_dtype="The data type",
            get_num_frames="The number of frames",
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
        start_frame, end_frame = self._validate_get_video_arguments(start_frame, end_frame)
        shape = (end_frame - start_frame, *self.get_image_size())
        video = np.empty(shape=shape, dtype=self.get_dtype())
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
        super().get_frames(frame_idxs=frame_idxs)

    def get_image_size(self) -> Tuple:
        """Get the size of a single frame.

        Returns
        -------
        image_size: tuple
            The size of a single frame (num_rows, num_columns, num_planes).
        """
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

    def get_num_frames(self) -> int:
        return self._imaging_extractors[0].get_num_frames()

    def get_sampling_frequency(self) -> float:
        return self._imaging_extractors[0].get_sampling_frequency()

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
    installed = True
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
