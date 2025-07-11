"""Base class definition for volumetric imaging extractors."""

import warnings
from typing import Iterable, List, Optional, Tuple

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
        self.is_volumetric = True

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

    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
        if start_sample is None:
            start_sample = 0
        elif start_sample < 0:
            start_sample = self.get_num_samples() + start_sample
        elif start_sample >= self.get_num_samples():
            raise ValueError(
                f"start_sample {start_sample} is greater than or equal to the number of samples {self.get_num_samples()}"
            )
        if end_sample is None:
            end_sample = self.get_num_samples()
        elif end_sample < 0:
            end_sample = self.get_num_samples() + end_sample
        elif end_sample > self.get_num_samples():
            raise ValueError(f"end_sample {end_sample} is greater than the number of samples {self.get_num_samples()}")
        if end_sample <= start_sample:
            raise ValueError(f"end_sample {end_sample} is less than or equal to start_sample {start_sample}")

        series = np.zeros((end_sample - start_sample, *self.get_sample_shape()), self.get_dtype())
        for i, imaging_extractor in enumerate(self._imaging_extractors):
            series[..., i] = imaging_extractor.get_series(start_sample, end_sample)
        return series

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

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_series() instead.
        """
        warnings.warn(
            "get_video() is deprecated and will be removed in or after September 2025. " "Use get_series() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_series(start_sample=start_frame, end_sample=end_frame)

    def get_frames(self, frame_idxs: ArrayType, channel: Optional[int] = 0) -> np.ndarray:
        """Get specific video frames from indices (not necessarily continuous).

        Parameters
        ----------
        frame_idxs: array-like
            Indices of frames to return.
        channel: int, optional
            Channel index. Deprecated: This parameter will be removed in August 2025.

        Returns
        -------
        frames: numpy.ndarray
            The 3D video frames (num_rows, num_columns, num_planes).
        """
        if channel != 0:
            warnings.warn(
                "The 'channel' parameter in get_frames() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
        if isinstance(frame_idxs, int):
            frame_idxs = [frame_idxs]
        for frame_idx in frame_idxs:
            if frame_idx < -1 * self.get_num_samples() or frame_idx >= self.get_num_samples():
                raise ValueError(f"frame_idx {frame_idx} is out of bounds")

        # Note np.all([]) returns True so not all(np.diff(frame_idxs) == 1) returns False if frame_idxs is a single int
        if not all(np.diff(frame_idxs) == 1):
            frames = np.zeros((len(frame_idxs), *self.get_sample_shape()), self.get_dtype())
            for i, imaging_extractor in enumerate(self._imaging_extractors):
                frames[..., i] = imaging_extractor.get_frames(frame_idxs, channel=channel)
            return frames
        else:
            return self.get_series(start_sample=frame_idxs[0], end_sample=frame_idxs[-1] + 1)

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
        image_size = (*self._imaging_extractors[0].get_frame_shape(), self.get_num_planes())
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

    def get_volume_shape(self) -> Tuple[int, int, int]:
        """Get the shape of the volumetric video (num_rows, num_columns, num_planes).

        Returns
        -------
        video_shape: tuple
            Shape of the volumetric video (num_rows, num_columns, num_planes).
        """
        image_shape = self.get_image_shape()
        return (image_shape[0], image_shape[1], self.get_num_planes())

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

    def slice_samples(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None):
        """Return a new VolumetricImagingExtractor with a subset of samples."""
        raise NotImplementedError(
            "slice_samples is not implemented for VolumetricImagingExtractor due to conflicts with get_series()."
        )

    def frame_slice(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None):
        """Return a new VolumetricImagingExtractor with a subset of frames.

        Deprecated
        ----------
        This method will be removed in or after October 2025.
        Use slice_samples() instead.
        """
        warnings.warn(
            "frame_slice() is deprecated and will be removed in or after October 2025. " "Use slice_samples() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.slice_samples(start_sample=start_frame, end_sample=end_frame)

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # Delegate to the first imaging extractor
        return self._imaging_extractors[0].get_native_timestamps(start_sample, end_sample)


class DepthSliceVolumetricImagingExtractor(VolumetricImagingExtractor):
    """Class to get a lazy depth slice.

    This class can only be used for volumetric imaging data.
    Do not use this class directly but use `.depth_slice(...)` on a VolumetricImagingExtractor object.
    """

    extractor_name = "DepthSliceVolumetricImagingExtractor"
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
