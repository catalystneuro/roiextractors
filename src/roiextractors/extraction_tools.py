"""Various tools for extraction of ROIs from imaging data.

Classes
-------
VideoStructure
    A data class for specifying the structure of a video.
"""

import sys
import importlib.util
from functools import wraps
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List
from types import ModuleType
from dataclasses import dataclass
from platform import python_version

import lazy_ops
import scipy
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from tqdm import tqdm
from packaging import version


try:
    import h5py

    HAVE_H5 = True
except ImportError:
    HAVE_H5 = False
try:
    if hasattr(scipy.io.matlab, "mat_struct"):
        from scipy.io.matlab import mat_struct
    else:
        from scipy.io.matlab.mio5_params import mat_struct

    HAVE_Scipy = True
except AttributeError:
    if hasattr(scipy, "io") and hasattr(scipy.io.matlab, "mat_struct"):
        from scipy.io import mat_struct
    else:
        from scipy.io.matlab.mio5_params import mat_struct

    HAVE_Scipy = True
except ImportError:
    HAVE_Scipy = False

try:
    import zarr

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


ArrayType = ArrayLike
PathType = Union[str, Path]
NumpyArray = np.ndarray
DtypeType = DTypeLike
IntType = Union[int, np.integer]
FloatType = float


def raise_multi_channel_or_depth_not_implemented(extractor_name: str):
    """Raise a NotImplementedError for an extractor that does not support multiple channels or depth (z-axis)."""
    raise NotImplementedError(
        f"The {extractor_name}Extractor does not currently support multiple color channels or 3-dimensional depth."
        "If you with to request either of these features, please do so by raising an issue at "
        "https://github.com/catalystneuro/roiextractors/issues"
    )


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

        axis_values = set((self.rows_axis, self.columns_axis, self.channels_axis, self.frame_axis))
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


def _pixel_mask_extractor(image_mask_, _roi_ids):
    """Convert image mask to pixel mask.

    Pixel masks are an alternative data format for storage of image masks which relies on the sparsity of the images.
    The location and weight of each non-zero pixel is stored for each mask.

    Parameters
    ----------
    image_mask_: numpy.ndarray
        Dense representation of the ROIs with shape (number_of_rows, number_of_columns, number_of_rois).
    _roi_ids: list
        List of roi ids with length number_of_rois.

    Returns
    -------
    pixel_masks: list
        List of length number of rois, each element is a 2-D array with shape (number_of_non_zero_pixels, 3).
        Columns 1 and 2 are the x and y coordinates of the pixel, while the third column represents the weight of
        the pixel.
    """
    pixel_mask_list = []
    for i, roiid in enumerate(_roi_ids):
        image_mask = np.array(image_mask_[:, :, i])
        _locs = np.where(image_mask > 0)
        _pix_values = image_mask[image_mask > 0]
        pixel_mask_list.append(np.vstack((_locs[0], _locs[1], _pix_values)).T)
    return pixel_mask_list


def _image_mask_extractor(pixel_mask, _roi_ids, image_shape):
    """Convert a pixel mask to image mask.

    Parameters
    ----------
    pixel_mask: list
        list of pixel masks (no pixels X 3)
    _roi_ids: list
        list of roi ids with length number_of_rois
    image_shape: array_like
        shape of the image (number_of_rows, number_of_columns)

    Returns
    -------
    image_mask: np.ndarray
        Dense representation of the ROIs with shape (number_of_rows, number_of_columns, number_of_rois).
    """
    image_mask = np.zeros(list(image_shape) + [len(_roi_ids)])
    for no, rois in enumerate(_roi_ids):
        for y, x, wt in pixel_mask[rois]:
            image_mask[int(y), int(x), no] = wt
    return image_mask


def get_video_shape(video):
    """Get the shape of a video (num_channels, num_frames, size_x, size_y).

    Parameters
    ----------
    video: numpy.ndarray
        The video to get the shape of.

    Returns
    -------
    video_shape: tuple
        The shape of the video (num_channels, num_frames, size_x, size_y).
    """
    if len(video.shape) == 3:
        # 1 channel
        num_channels = 1
        num_frames, size_x, size_y = video.shape
    else:
        num_channels, num_frames, size_x, size_y = video.shape
    return num_channels, num_frames, size_x, size_y


def check_get_frames_args(func):
    """Check the arguments of the get_frames function.

    This decorator allows the get_frames function to be queried with either
    an integer, slice or an array and handles a common return. [I think that np.take can be used instead of this]

    Parameters
    ----------
    func: function
        The get_frames function.

    Returns
    -------
    corrected_args: function
        The get_frames function with corrected arguments.

    Raises
    ------
    AssertionError
        If 'frame_idxs' exceed the number of frames.
    """

    @wraps(func)
    def corrected_args(imaging, frame_idxs, channel=0):
        channel = int(channel)
        if isinstance(frame_idxs, (int, np.integer)):
            frame_idxs = [frame_idxs]
        if not isinstance(frame_idxs, slice):
            frame_idxs = np.array(frame_idxs)
            assert np.all(frame_idxs < imaging.get_num_frames()), "'frame_idxs' exceed number of frames"
        get_frames_correct_arg = func(imaging, frame_idxs, channel)

        if len(frame_idxs) == 1:
            return get_frames_correct_arg[0]
        else:
            return get_frames_correct_arg

    return corrected_args


def _cast_start_end_frame(start_frame, end_frame):
    """Cast start and end frame to int or None.

    Parameters
    ----------
    start_frame: int, float, None
        The start frame.
    end_frame: int, float, None
        The end frame.

    Returns
    -------
    start_frame: int, None
        The start frame.
    end_frame: int, None
        The end frame.

    Raises
    ------
    ValueError
        If start_frame is not an int, float or None.
    ValueError
        If end_frame is not an int, float or None.
    """
    if isinstance(start_frame, float):
        start_frame = int(start_frame)
    elif isinstance(start_frame, (int, np.integer, type(None))):
        start_frame = start_frame
    else:
        raise ValueError("start_frame must be an int, float (not infinity), or None")
    if isinstance(end_frame, float) and np.isfinite(end_frame):
        end_frame = int(end_frame)
    elif isinstance(end_frame, (int, np.integer, type(None))):
        end_frame = end_frame
    # else end_frame is infinity (accepted for get_unit_spike_train)
    if start_frame is not None:
        start_frame = int(start_frame)
    if end_frame is not None and np.isfinite(end_frame):
        end_frame = int(end_frame)
    return start_frame, end_frame


def check_get_videos_args(func):
    """Check the arguments of the get_videos function.

    This decorator allows the get_videos function to be queried with either
    an integer or slice and handles a common return.

    Parameters
    ----------
    func: function
        The get_videos function.

    Returns
    -------
    corrected_args: function
        The get_videos function with corrected arguments.

    Raises
    ------
    AssertionError
        If 'start_frame' exceeds the number of frames.
    AssertionError
        If 'end_frame' exceeds the number of frames.
    AssertionError
        If 'start_frame' is greater than 'end_frame'.
    """

    @wraps(func)
    def corrected_args(imaging, start_frame=None, end_frame=None, channel=0):
        if start_frame is not None:
            if start_frame > imaging.get_num_frames():
                raise Exception(f"'start_frame' exceeds number of frames {imaging.get_num_frames()}!")
            elif start_frame < 0:
                start_frame = imaging.get_num_frames() + start_frame
        else:
            start_frame = 0
        if end_frame is not None:
            if end_frame > imaging.get_num_frames():
                raise Exception(f"'end_frame' exceeds number of frames {imaging.get_num_frames()}!")
            elif end_frame < 0:
                end_frame = imaging.get_num_frames() + end_frame
        else:
            end_frame = imaging.get_num_frames()
        assert end_frame - start_frame > 0, "'start_frame' must be less than 'end_frame'!"

        start_frame, end_frame = _cast_start_end_frame(start_frame, end_frame)
        channel = int(channel)
        get_videos_correct_arg = func(imaging, start_frame=start_frame, end_frame=end_frame, channel=channel)

        return get_videos_correct_arg

    return corrected_args


def write_to_h5_dataset_format(
    imaging,
    dataset_path,
    save_path=None,
    file_handle=None,
    dtype=None,
    chunk_size=None,
    chunk_mb=1000,
    verbose=False,
):
    """Save the video of an imaging extractor in an h5 dataset.

    Parameters
    ----------
    imaging: ImagingExtractor
        The imaging extractor object to be saved in the .h5 file
    dataset_path: str
        Path to dataset in h5 file (e.g. '/dataset')
    save_path: str
        The path to the file.
    file_handle: file handle
        The file handle to dump data. This can be used to append data to an header. In case file_handle is given,
        the file is NOT closed after writing the binary data.
    dtype: dtype
        Type of the saved data. Default float32.
    chunk_size: None or int
        Number of chunks to save the file in. This avoid to much memory consumption for big files.
        If None and 'chunk_mb' is given, the file is saved in chunks of 'chunk_mb' Mb (default 500Mb)
    chunk_mb: None or int
        Chunk size in Mb (default 1000Mb)
    verbose: bool
        If True, output is verbose (when chunks are used)

    Returns
    -------
    save_path: str
        The path to the file.

    Raises
    ------
    AssertionError
        If h5py is not installed.
    AssertionError
        If neither 'save_path' nor 'file_handle' are given.
    """
    assert HAVE_H5, "To write to h5 you need to install h5py: pip install h5py"
    assert save_path is not None or file_handle is not None, "Provide 'save_path' or 'file handle'"

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix == "":
            # when suffix is already raw/bin/dat do not change it.
            save_path = save_path.parent / (save_path.name + ".h5")
    num_channels = imaging.get_num_channels()
    num_frames = imaging.get_num_frames()
    size_x, size_y = imaging.get_image_size()

    if file_handle is not None:
        assert isinstance(file_handle, h5py.File)
    else:
        file_handle = h5py.File(save_path, "w")
    if dtype is None:
        dtype_file = imaging.get_dtype()
    else:
        dtype_file = dtype
    dset = file_handle.create_dataset(dataset_path, shape=(num_channels, num_frames, size_x, size_y), dtype=dtype_file)

    # set chunk size
    if chunk_size is not None:
        chunk_size = int(chunk_size)
    elif chunk_mb is not None:
        n_bytes = np.dtype(imaging.get_dtype()).itemsize
        max_size = int(chunk_mb * 1e6)  # set Mb per chunk
        chunk_size = max_size // (size_x * size_y * n_bytes)
    # writ one channel at a time
    for ch in range(num_channels):
        if chunk_size is None:
            video = imaging.get_video(channel=ch)
            if dtype is not None:
                video = video.astype(dtype_file)
            dset[ch, ...] = np.squeeze(video)
        else:
            chunk_start = 0
            # chunk size is not None
            n_chunk = num_frames // chunk_size
            if num_frames % chunk_size > 0:
                n_chunk += 1
            if verbose:
                chunks = tqdm(range(n_chunk), ascii=True, desc="Writing to .h5 file")
            else:
                chunks = range(n_chunk)
            for i in chunks:
                video = imaging.get_video(
                    start_frame=i * chunk_size,
                    end_frame=min((i + 1) * chunk_size, num_frames),
                    channel=ch,
                )
                chunk_frames = np.squeeze(video).shape[0]
                if dtype is not None:
                    video = video.astype(dtype_file)
                dset[ch, chunk_start : chunk_start + chunk_frames, ...] = np.squeeze(video)
                chunk_start += chunk_frames
    if save_path is not None:
        file_handle.close()
    return save_path


# TODO will be moved eventually, but for now it's very handy :)
def show_video(imaging, ax=None):
    """Show video as animation.

    Parameters
    ----------
    imaging: ImagingExtractor
        The imaging extractor object to be saved in the .h5 file
    ax: matplotlib axis
        Axis to plot the video. If None, a new axis is created.

    Returns
    -------
    anim: matplotlib.animation.FuncAnimation
        Animation of the video.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    def animate_func(i, imaging, im, ax):
        ax.set_title(f"{i}")
        im.set_array(imaging.get_frames(i))
        return [im]

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
    im0 = imaging.get_frames(0)
    im = ax.imshow(im0, interpolation="none", aspect="auto", vmin=0, vmax=1)
    interval = 1 / imaging.get_sampling_frequency() * 1000
    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=imaging.get_num_frames(),
        fargs=(imaging, im, ax),
        interval=interval,
        blit=False,
    )
    return anim


def check_keys(dict):
    """Check keys of dictionary for mat-objects.

    Checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries.

    Parameters
    ----------
    dict: dict
        Dictionary to check.

    Returns
    -------
    dict: dict
        Dictionary with mat-objects converted to nested dictionaries.

    Raises
    ------
    AssertionError
        If scipy is not installed.
    """
    assert HAVE_Scipy, "To write to h5 you need to install scipy: pip install scipy"
    for key in dict:
        if isinstance(dict[key], mat_struct):
            dict[key] = todict(dict[key])
    return dict


def todict(matobj):
    """Recursively construct nested dictionaries from matobjects.

    Parameters
    ----------
    matobj: mat_struct
        Matlab object to convert to nested dictionary.

    Returns
    -------
    dict: dict
        Dictionary with mat-objects converted to nested dictionaries.
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, mat_struct):
            dict[strg] = todict(elem)
        else:
            dict[strg] = elem
    return dict


def get_package(
    package_name: str,
    installation_instructions: Optional[str] = None,
    excluded_platforms_and_python_versions: Optional[Dict[str, List[str]]] = None,
) -> ModuleType:
    """Check if package is installed and return module if so.

    Otherwise, raise informative error describing how to perform the installation.
    Inspired by https://docs.python.org/3/library/importlib.html#checking-if-a-module-can-be-imported.

    Parameters
    ----------
    package_name : str
        Name of the package to be imported.
    installation_instructions : str, optional
        String describing the source, options, and alias of package name (if needed) for installation.
        For example,
            >>> installation_source = "conda install -c conda-forge my-package-name"
        Defaults to f"pip install {package_name}".
    excluded_platforms_and_python_versions : dict mapping string platform names to a list of string versions, optional
        In case some combinations of platforms or Python versions are not allowed for the given package, specify
        this dictionary to raise a more specific error to that issue.
        For example, `excluded_platforms_and_python_versions = dict(darwin=["3.7"])` will raise an informative error
        when running on MacOS with Python version 3.7.
        Allows all platforms and Python versions used by default.

    Raises
    ------
    ModuleNotFoundError
        If the package is not installed.
    """
    installation_instructions = installation_instructions or f"pip install {package_name}"
    excluded_platforms_and_python_versions = excluded_platforms_and_python_versions or dict()

    if package_name in sys.modules:
        return sys.modules[package_name]

    if importlib.util.find_spec(package_name) is not None:
        return importlib.import_module(name=package_name)

    for excluded_version in excluded_platforms_and_python_versions.get(sys.platform, list()):
        if version.parse(python_version()).minor == version.parse(excluded_version).minor:
            raise ModuleNotFoundError(
                f"\nThe package '{package_name}' is not available on the {sys.platform} platform for "
                f"Python version {excluded_version}!"
            )

    raise ModuleNotFoundError(
        f"\nThe required package'{package_name}' is not installed!\n"
        f"To install this package, please run\n\n\t{installation_instructions}\n"
    )
