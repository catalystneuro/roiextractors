"""Imaging Extractors for Scanbox files.

Classes
-------
SbxImagingExtractor
    An ImagingExtractor for Scanbox Image files.
"""

from multiprocessing.sharedctypes import Value
import os
from pathlib import Path
from warnings import warn
from typing import Tuple, Optional

import numpy as np

from ...extraction_tools import PathType, ArrayType, raise_multi_channel_or_depth_not_implemented, check_keys
from ...imagingextractor import ImagingExtractor

try:
    import scipy.io as spio

    HAVE_Scipy = True
except ImportError:
    HAVE_Scipy = False


class SbxImagingExtractor(ImagingExtractor):
    """Imaging extractor for the Scanbox image format."""

    extractor_name = "SbxImaging"
    installed = HAVE_Scipy  # check at class level if installed or not
    is_writable = True
    mode = "folder"
    installation_mesg = "To use the Sbx Extractor run:\n\n pip install scipy\n\n"  # error message when not installed

    def __init__(self, file_path: PathType, sampling_frequency: Optional[float] = None):
        """Create a SbxImagingExtractor from .mat or .sbx files.

        Parameters
        ----------
        file_path : str or python Path objects
            The file path pointing to a file in either `.mat` or `.sbx` format.
        sampling_frequency : float, optional
            The sampling frequeny of the imaging device.
        """
        super().__init__()
        self._memmapped = True
        self.mat_file_path, self.sbx_file_path = self._return_mat_and_sbx_filepaths(file_path)
        self._info = self._loadmat()
        self._data = self._sbx_read()
        self._sampling_frequency = self._info.get("frame_rate", sampling_frequency)
        if self._sampling_frequency is None:
            raise ValueError(
                "sampling rate not found in in the `.mat` file, provide it with the sampling_frequency argument"
            )
        self._sampling_frequency = float(self._sampling_frequency)

        # channel names:
        self._channel_names = self._info.get("channel_names", None)
        if self._channel_names is None:
            self._channel_names = [f"channel_{ch}" for ch in range(self._info["nChan"])]

        self._kwargs = {
            "file_path": str(Path(file_path).absolute()),
            "sampling_frequency": self._sampling_frequency,
            "channel_names": self._channel_names,
        }

    @staticmethod
    def _return_mat_and_sbx_filepaths(file_path):
        """Return the `.mat` and `.sbx` file paths from a given file path pointing to either of them.

        Parameters
        ----------
        file_path : str or python Path objects
            The file path pointing to a file in either `.mat` or `.sbx` format.

        Returns
        -------
        mat_file_path : str or python Path object
            The file path pointing to the `.mat` file.
        sbx_file_path : str or python Path object
            The file path pointing to the `.sbx` file.
        """
        file_path = Path(file_path)
        if file_path.suffix not in [".mat", ".sbx"]:
            assertion_msg = "File path not pointing to a `.sbx` or `.mat` file"
            raise ValueError(assertion_msg)

        mat_file_path = file_path.with_suffix(".mat")
        sbx_file_path = file_path.with_suffix(".sbx")
        return mat_file_path, sbx_file_path

    def _loadmat(self):
        """Load matlab .mat file.

        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects.
        Based off of implementations @:
        https://github.com/GiocomoLab/TwoPUtils/blob/main/TwoPUtils/scanner_tools/sbx_utils.py
        https://github.com/losonczylab/sima/blob/0b16818d9ba47fe4aae6d4aad1a9735d16da00dc/sima/imaging_parameters.py
        https://scanbox.org/2016/09/02/reading-scanbox-files-in-python/
        """
        data = spio.loadmat(self.mat_file_path, struct_as_record=False, squeeze_me=True)
        info = check_keys(data)["info"]
        # Defining number of channels/size factor
        if info["channels"] == 1:
            info["nChan"] = 2
            factor = 1
        elif info["channels"] == 2:
            info["nChan"] = 1
            factor = 2
        elif info["channels"] == 3:
            info["nChan"] = 1
            factor = 2
        else:
            raise UserWarning("wrong 'channels' argument")

        if info["scanmode"] == 0:
            info["recordsPerBuffer"] *= 2

        if "fold_lines" in info:
            if info["fold_lines"] > 0:
                info["fov_repeats"] = int(info["config"]["lines"] / info["fold_lines"])
            else:
                info["fov_repeats"] = 1
        else:
            info["fold_lines"] = 0
            info["fov_repeats"] = 1

        info["frame_rate"] = int(
            info["resfreq"] / info["config"]["lines"] * (2 - info["scanmode"]) * info["fov_repeats"]
        )
        # SIMA:
        info["nsamples"] = info["sz"][1] * info["recordsPerBuffer"] * info["nChan"] * 2
        # SIMA:
        if ("volscan" in info and info["volscan"] > 0) or ("volscan" not in info and len(info.get("otwave", []))):
            info["nplanes"] = len(info["otwave"])
        else:
            info["nplanes"] = 1
        # SIMA:
        if info.get("scanbox_version", -1) >= 2:
            info["max_idx"] = os.path.getsize(self.sbx_file_path) // info["nsamples"] - 1
        else:
            info["max_idx"] = os.path.getsize(self.sbx_file_path) // info["bytesPerBuffer"] * factor - 1
        # SIMA: Fix for old scanbox versions
        if "sz" not in info:
            info["sz"] = np.array([512, 796])
        return info

    def _sbx_read(self):
        """Read the `.sbx` file and return a numpy array.

        Returns
        -------
        np_data : np.ndarray
            The numpy array containing the data from the `.sbx` file.
        """
        nrows = self._info["recordsPerBuffer"]
        ncols = self._info["sz"][1]
        nchannels = self._info["nChan"]
        nplanes = self._info["nplanes"]
        nframes = (self._info["max_idx"] + 1) // nplanes
        shape = (nchannels, ncols, nrows, nplanes, nframes)

        if nchannels != 1:
            raise_multi_channel_or_depth_not_implemented(extractor_name=self.extractor_name)

        np_data = np.memmap(self.sbx_file_path, dtype="uint16", mode="r", shape=shape, order="F")
        return np_data

    def get_frames(self, frame_idxs: ArrayType, channel: int = 0) -> np.array:
        frame_out = np.iinfo("uint16").max - self._data.transpose(4, 2, 1, 0, 3)[frame_idxs, :, :, channel, 0]

        return frame_out

    def get_video(self, start_frame=None, end_frame=None, channel: Optional[int] = 0) -> np.ndarray:
        frame_out = np.iinfo("uint16").max - self._data[channel, :, :, 0, start_frame:end_frame]
        return frame_out.transpose(2, 1, 0)

    def get_image_size(self) -> Tuple[int, int]:
        return tuple(self._info["sz"])

    def get_num_frames(self) -> int:
        return (self._info["max_idx"] + 1) // self._info["nplanes"]

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_channel_names(self) -> list:
        return self._channel_names

    def get_num_channels(self) -> int:
        return self._info["nChan"]

    @staticmethod
    def write_imaging(imaging, save_path: PathType, overwrite: bool = False):
        """Write a SbxImagingExtractor to a `.mat` file.

        Parameters
        ----------
        imaging : SbxImagingExtractor
            The imaging extractor object to be written to a `.mat` file.
        save_path : str or python Path object
            The path to the `.mat` file to be written.
        overwrite : bool, optional
            If True, the `.mat` file will be overwritten if it already exists.

        Notes
        -----
        This function is not implemented yet.
        """
        raise NotImplementedError
