"""NumpyMemmapImagingExtractor class.

Classes
-------
NumpyMemmapImagingExtractor
    The class for reading optical imaging data stored in a binary format with numpy.memmap.
"""

from pathlib import Path
from typing import Tuple
from dataclasses import dataclass
import lazy_ops
import numpy as np
import h5py
from .memmapextractors import MemmapImagingExtractor


@dataclass
class VideoStructure:
    """A data class for specifying the structure of a video.

    The role of the data class is to ensure consistency in naming and provide some initial
    consistency checks to ensure the validity of the sturcture.

    Attributes
    ----------
    num_rows : int
        The number of rows of each frame as a matrix.
    num_columns : int
        The number of columns of each frame as a matrix.
    num_channels : int
        The number of channels (1 for grayscale, 3 for color).
    rows_axis : int
        The axis or dimension corresponding to the rows.
    columns_axis : int
        The axis or dimension corresponding to the columns.
    channels_axis : int
        The axis or dimension corresponding to the channels.
    frame_axis : int
        The axis or dimension corresponding to the frames in the video.

    As an example if you wanted to build the structure for a video with gray (n_channels=1) frames of 10 x 5
    where the video is to have the following shape (num_frames, num_rows, num_columns, num_channels) you
    could define the class this way:

    >>> from roiextractors.extraction_tools import VideoStructure
    >>> num_rows = 10
    >>> num_columns = 5
    >>> num_channels = 1
    >>> frame_axis = 0
    >>> rows_axis = 1
    >>> columns_axis = 2
    >>> channels_axis = 3
    >>> video_structure = VideoStructure(
        num_rows=num_rows,
        num_columns=num_columns,
        num_channels=num_channels,
        rows_axis=rows_axis,
        columns_axis=columns_axis,
        channels_axis=channels_axis,
        frame_axis=frame_axis,
    )
    """

    num_rows: int
    num_columns: int
    num_channels: int
    rows_axis: int
    columns_axis: int
    channels_axis: int
    frame_axis: int

    def __post_init__(self) -> None:
        """Validate the structure of the video and initialize the shape of the frame."""
        self._validate_video_structure()
        self._initialize_frame_shape()
        self.number_of_pixels_per_frame = np.prod(self.frame_shape)

    def _initialize_frame_shape(self) -> None:
        """Initialize the shape of the frame."""
        self.frame_shape = [None, None, None, None]
        self.frame_shape[self.rows_axis] = self.num_rows
        self.frame_shape[self.columns_axis] = self.num_columns
        self.frame_shape[self.channels_axis] = self.num_channels
        self.frame_shape.pop(self.frame_axis)
        self.frame_shape = tuple(self.frame_shape)

    def _validate_video_structure(self) -> None:
        """Validate the structure of the video."""
        exception_message = (
            "Invalid structure: "
            f"{self.__repr__()}, "
            "each property axis should be unique value between 0 and 3 (inclusive)"
        )

        axis_values = {self.rows_axis, self.columns_axis, self.channels_axis, self.frame_axis}
        axis_values_are_not_unique = len(axis_values) != 4
        if axis_values_are_not_unique:
            raise ValueError(exception_message)

        values_out_of_range = any([axis < 0 or axis > 4 for axis in axis_values])
        if values_out_of_range:
            raise ValueError(exception_message)

    def build_video_shape(self, n_frames: int) -> Tuple[int, int, int, int]:
        """Build the shape of the video from class attributes.

        Parameters
        ----------
        n_frames : int
            The number of frames in the video.

        Returns
        -------
        Tuple[int, int, int, int]
            The shape of the video.

        Notes
        -----
        The class attributes frame_axis, rows_axis, columns_axis and channels_axis are used to determine the order of the
        dimensions in the returned tuple.
        """
        video_shape = [None] * 4
        video_shape[self.frame_axis] = n_frames
        video_shape[self.rows_axis] = self.num_rows
        video_shape[self.columns_axis] = self.num_columns
        video_shape[self.channels_axis] = self.num_channels

        return tuple(video_shape)

    def transform_video_to_canonical_form(self, video: np.ndarray) -> np.ndarray:
        """Transform a video to the canonical internal format of roiextractors (num_frames, num_rows, num_columns, num_channels).

        Parameters
        ----------
        video : numpy.ndarray
            The video to be transformed
        Returns
        -------
        numpy.ndarray
            The reshaped video

        Raises
        ------
        KeyError
            If the video is not in a format that can be transformed.
        """
        canonical_shape = (self.frame_axis, self.rows_axis, self.columns_axis, self.channels_axis)
        if isinstance(video, (h5py.Dataset, zarr.core.Array)):
            re_mapped_video = lazy_ops.DatasetView(video).lazy_transpose(canonical_shape)
        elif isinstance(video, np.ndarray):
            re_mapped_video = video.transpose(canonical_shape)
        else:
            raise KeyError(f"Function not implemented for specific format {type(video)}")

        return re_mapped_video


def read_numpy_memmap_video(
    file_path: PathType, video_structure: VideoStructure, dtype: DtypeType, offset: int = 0
) -> np.array:
    """Auxiliary function to read videos from binary files.

    Parameters
    ----------
        file_path : PathType
            the file_path where the data resides.
        video_structure : VideoStructure
            A VideoStructure instance describing the structure of the video to read. This includes parameters
            such as the number of rows, columns and channels plus which axis (i.e. dimension) of the
            image corresponds to each of them.

            As an example you create one of these structures in the following way:

            from roiextractors.extraction_tools import VideoStructure

            num_rows = 10
            num_columns = 5
            num_channels = 3
            frame_axis = 0
            rows_axis = 1
            columns_axis = 2
            channels_axis = 3

            video_structure = VideoStructure(
                num_rows=num_rows,
                num_columns=num_columns,
                num_channels=num_channels,
                rows_axis=rows_axis,
                columns_axis=columns_axis,
                channels_axis=channels_axis,
                frame_axis=frame_axis,
            )

        dtype : DtypeType
            The type of the data to be loaded (int, float, etc.)
        offset : int, optional
            The offset in bytes. Usually corresponds to the number of bytes occupied by the header. 0 by default.

    Returns
    -------
    video_memap: np.array
        A numpy memmap pointing to the video.
    """
    file_size_bytes = Path(file_path).stat().st_size

    pixels_per_frame = video_structure.number_of_pixels_per_frame
    type_size = np.dtype(dtype).itemsize
    frame_size_bytes = pixels_per_frame * type_size

    bytes_available = file_size_bytes - offset
    number_of_frames = bytes_available // frame_size_bytes

    memmap_shape = video_structure.build_video_shape(n_frames=number_of_frames)
    video_memap = np.memmap(file_path, offset=offset, dtype=dtype, mode="r", shape=memmap_shape)

    return video_memap


class NumpyMemmapImagingExtractor(MemmapImagingExtractor):
    """An ImagingExtractor class for reading optical imaging data stored in a binary format with numpy.memmap."""

    extractor_name = "NumpyMemmapImagingExtractor"

    def __init__(
        self,
        file_path: PathType,
        video_structure: VideoStructure,
        sampling_frequency: float,
        dtype: DtypeType,
        offset: int = 0,
    ):
        """Create an instance of NumpyMemmapImagingExtractor.

        Parameters
        ----------
        file_path : PathType
            the file_path where the data resides.
        video_structure : VideoStructure
            A VideoStructure instance describing the structure of the image to read. This includes parameters
            such as the number of rows, columns and channels plus which axis (i.e. dimension) of the
            image corresponds to each of them.

            As an example you create one of these structures in the following way:

            from roiextractors.extraction_tools import VideoStructure

            num_rows = 10
            num_columns = 5
            num_channels = 3
            frame_axis = 0
            rows_axis = 1
            columns_axis = 2
            channel_axis = 3

            video_structure = VideoStructure(
                num_rows=num_rows,
                columns=columns,
                num_channels=num_channels,
                rows_axis=rows_axis,
                columns_axis=columns_axis,
                channel_axis=channel_axis,
                frame_axis=frame_axis,
            )

        sampling_frequency : float, optional
            The sampling frequency.
        dtype : DtypeType
            The type of the data to be loaded (int, float, etc.)
        offset : int, optional
            The offset in bytes. Usually corresponds to the number of bytes occupied by the header. 0 by default.
        """
        self.installed = True

        self.file_path = Path(file_path)
        self.video_structure = video_structure
        self._sampling_frequency = float(sampling_frequency)
        self.offset = offset
        self.dtype = dtype

        # Extract video
        self._video = read_numpy_memmap_video(
            file_path=file_path, video_structure=video_structure, dtype=dtype, offset=offset
        )
        self._video = video_structure.transform_video_to_canonical_form(self._video)
        self._num_frames, self._num_rows, self._num_columns, self._num_channels = self._video.shape

        super().__init__(video=self._video)
