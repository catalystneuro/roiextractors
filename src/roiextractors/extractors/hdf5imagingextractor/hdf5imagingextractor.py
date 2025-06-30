"""An imaging extractor for HDF5.

Classes
-------
Hdf5ImagingExtractor
    An imaging extractor for HDF5.
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple
from warnings import warn

import h5py
import numpy as np
from lazy_ops import DatasetView

from ...extraction_tools import (
    ArrayType,
    FloatType,
    PathType,
    write_to_h5_dataset_format,
)
from ...imagingextractor import ImagingExtractor


class Hdf5ImagingExtractor(ImagingExtractor):
    """An imaging extractor for HDF5."""

    extractor_name = "Hdf5Imaging"
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
        self._num_channels, self._num_samples, self._num_rows, self._num_cols = self._video.shape
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
        """Get specific video frames from indices.

        Parameters
        ----------
        frame_idxs: array-like
            Indices of frames to return.
        channel: int, optional
            Channel index. Deprecated: This parameter will be removed in August 2025.

        Returns
        -------
        frames: numpy.ndarray
            The video frames.
        """
        if channel != 0:
            warn(
                "The 'channel' parameter in get_frames() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
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

    def get_series(self, start_sample=None, end_sample=None) -> np.ndarray:
        return self._video.lazy_slice[start_sample:end_sample, :, :, 0].dsetread()

    def get_video(self, start_frame=None, end_frame=None, channel: Optional[int] = 0) -> np.ndarray:
        """Get the video frames.

        Parameters
        ----------
        start_frame: int, optional
            Start frame index (inclusive).
        end_frame: int, optional
            End frame index (exclusive).
        channel: int, optional
            Channel index. Deprecated: This parameter will be removed in August 2025.

        Returns
        -------
        video: numpy.ndarray
            The video frames.

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
        if channel != 0:
            warn(
                "The 'channel' parameter in get_video() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.get_series(start_sample=start_frame, end_sample=end_frame)

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._num_rows, self._num_cols

    def get_image_size(self) -> Tuple[int, int]:
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._num_rows, self._num_cols

    def get_num_samples(self):
        return self._num_samples

    def get_num_frames(self):
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

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_channel_names(self):
        return self._channel_names

    def get_num_channels(self):
        return self._num_channels

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Retrieve the original unaltered timestamps for the data in this interface.

        Returns
        -------
        timestamps: numpy.ndarray or None
            The timestamps for the data stream, or None if native timestamps are not available.
        """
        # HDF5 imaging data does not have native timestamps
        return None

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
        warn(
            "The write_imaging function is deprecated and will be removed on or after September 2025. ROIExtractors is no longer supporting write operations.",
            DeprecationWarning,
            stacklevel=2,
        )
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
