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

from ...extraction_tools import PathType, FloatType, ArrayType
from ...extraction_tools import (
    get_video_shape,
    write_to_h5_dataset_format,
)
from ...imagingextractor import ImagingExtractor
from lazy_ops import DatasetView


try:
    import h5py

    HAVE_H5 = True
except ImportError:
    HAVE_H5 = False


class Hdf5ImagingExtractor(ImagingExtractor):
    """An imaging extractor for HDF5."""

    extractor_name = "Hdf5Imaging"
    installed = HAVE_H5  # check at class level if installed or not
    is_writable = True
    mode = "file"
    installation_mesg = "To use the Hdf5 Extractor run:\n\n pip install h5py\n\n"  # error message when not installed

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
