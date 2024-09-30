"""An imaging extractor for HDF5.

Classes
-------
Hdf5ImagingExtractor
    An imaging extractor for HDF5.
"""

from pathlib import Path
from typing import Optional, Tuple
from warnings import warn

import numpy as np

from ...tools.typing import PathType, FloatType, ArrayType
from ...imagingextractor import ImagingExtractor
from lazy_ops import DatasetView


import h5py


class Hdf5ImagingExtractor(ImagingExtractor):
    """An imaging extractor for HDF5."""

    extractor_name = "Hdf5Imaging"
    is_writable = True
    mode = "file"

    def __init__(
        self,
        file_path: PathType,
        mov_field: str = "mov",
        sampling_frequency: FloatType = None,
        start_time: FloatType = None,
        metadata: dict = None,
        channel_names: ArrayType = None,
    ):
        """Create an ImagingExtractor from an HDF5 file.

        Parameters
        ----------
        file_path : str or Path
            Path to the HDF5 file.
        mov_field : str, optional
            Name of the dataset in the HDF5 file that contains the imaging data. The default is "mov".
        sampling_frequency : float, optional
            Sampling frequency of the video. The default is None.
        start_time : float, optional
            Start time of the video. The default is None.
        metadata : dict, optional
            Metadata dictionary. The default is None.
        channel_names : array-like, optional
            List of channel names. The default is None.
        """
        ImagingExtractor.__init__(self)

        self.filepath = Path(file_path)
        self._sampling_frequency = sampling_frequency
        self._mov_field = mov_field
        if self.filepath.suffix not in [".h5", ".hdf5"]:
            warn("'file_path' file is not an .hdf5 or .h5 file")
        self._channel_names = channel_names

        self._file = h5py.File(file_path, "r")
        if mov_field in self._file.keys():
            self._video = DatasetView(self._file[self._mov_field])
            if sampling_frequency is None:
                assert "fr" in self._video.attrs, (
                    "Sampling frequency is unavailable as a dataset attribute! "
                    "Please set the keyword argument 'sampling_frequency'"
                )
                self._sampling_frequency = float(self._video.attrs["fr"])
            else:
                self._sampling_frequency = sampling_frequency
        else:
            raise Exception(f"{file_path} does not contain the 'mov' dataset")

        if start_time is None:
            if "start_time" in self._video.attrs.keys():
                self._start_time = self._video.attrs["start_time"]
        else:
            self._start_time = start_time

        if metadata is None:
            if "metadata" in self._video.attrs:
                self.metadata = self._video.attrs["metadata"]
        else:
            self.metadata = metadata

        # The test data has four dimensions and the first axis is channels
        self._num_channels, self._num_frames, self._num_rows, self._num_cols = self._video.shape
        self._video = self._video.lazy_transpose([1, 2, 3, 0])

        if self._channel_names is not None:
            assert len(self._channel_names) == self._num_channels, (
                "'channel_names' length is different than number " "of channels"
            )
        else:
            self._channel_names = [f"channel_{ch}" for ch in range(self._num_channels)]

        self._kwargs = {
            "file_path": str(Path(file_path).absolute()),
            "mov_field": mov_field,
            "sampling_frequency": sampling_frequency,
            "channel_names": channel_names,
        }

    def __del__(self):
        """Close the HDF5 file."""
        self._file.close()

    def get_frames(self, frame_idxs: ArrayType, channel: Optional[int] = 0):
        squeeze_data = False
        if isinstance(frame_idxs, int):
            squeeze_data = True
            frame_idxs = [frame_idxs]
        elif isinstance(frame_idxs, np.ndarray):
            frame_idxs = frame_idxs.tolist()
        frames = self._video.lazy_slice[frame_idxs, :, :, channel].dsetread()
        if squeeze_data:
            frames = frames.squeeze()
        return frames

    def get_video(self, start_frame=None, end_frame=None, channel: Optional[int] = 0) -> np.ndarray:
        return self._video.lazy_slice[start_frame:end_frame, :, :, channel].dsetread()

    def get_image_size(self) -> Tuple[int, int]:
        return self._num_rows, self._num_cols

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_channel_names(self):
        return self._channel_names

    def get_num_channels(self):
        return self._num_channels

    @staticmethod
    def write_imaging(
        imaging: ImagingExtractor,
        save_path,
        overwrite: bool = False,
        mov_field="mov",
        **kwargs,
    ):
        """Write an imaging extractor to an HDF5 file.

        Parameters
        ----------
        imaging : ImagingExtractor
            The imaging extractor object to be saved.
        save_path : str or Path
            Path to save the file.
        overwrite : bool, optional
            If True, overwrite the file if it already exists. The default is False.
        mov_field : str, optional
            Name of the dataset in the HDF5 file that contains the imaging data. The default is "mov".
        **kwargs : dict
            Keyword arguments to be passed to the HDF5 file writer.

        Raises
        ------
        AssertionError
            If the file extension is not .h5 or .hdf5.
        FileExistsError
            If the file already exists and overwrite is False.
        """
        save_path = Path(save_path)
        assert save_path.suffix in [
            ".h5",
            ".hdf5",
        ], "'save_path' file is not an .hdf5 or .h5 file"

        if save_path.is_file():
            if not overwrite:
                raise FileExistsError("The specified path exists! Use overwrite=True to overwrite it.")
            else:
                save_path.unlink()

        with h5py.File(save_path, "w") as f:
            write_to_h5_dataset_format(imaging=imaging, dataset_path=mov_field, file_handle=f, **kwargs)
            dset = f[mov_field]
            dset.attrs["fr"] = imaging.get_sampling_frequency()


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
        If neither 'save_path' nor 'file_handle' are given.
    """
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
