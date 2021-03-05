import os
from pathlib import Path

import numpy as np

from ...extraction_tools import PathType, ArrayType
from ...extraction_tools import check_keys
from ...imagingextractor import ImagingExtractor

try:
    import scipy.io as spio

    HAVE_Scipy = True
except ImportError:
    HAVE_Scipy = False


class SbxImagingExtractor(ImagingExtractor):
    extractor_name = "SbxImaging"
    installed = HAVE_Scipy  # check at class level if installed or not
    is_writable = True
    mode = "folder"
    installation_mesg = "To use the Sbx Extractor run:\n\n pip install scipy\n\n"  # error message when not installed

    def __init__(self, file_path: PathType, sampling_frequency: float = None):
        super().__init__()
        self._memmapped = True
        self.mat_file_path, self.sbx_file_path = self._check_file_path(file_path)
        self._info = self._loadmat()
        self._data = self._sbx_read()
        self._sampling_frequency = sampling_frequency or float(self._info["frame_rate"])
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
    def _check_file_path(file_path):
        file_path = Path(file_path)
        assertion_msg = "for file_path arg, provide a path to one .sbx /  .mat file"
        file_type = file_path.suffix
        assert file_type in [".mat", ".sbx"], assertion_msg
        if file_type == ".mat":
            mat_file_path = file_path
            sbx_file_path = file_path.with_suffix(".sbx")
            assert sbx_file_path.is_file(), assertion_msg
        else:
            sbx_file_path = file_path
            mat_file_path = file_path.with_suffix(".mat")
            assert mat_file_path.is_file(), assertion_msg
        return mat_file_path, sbx_file_path

    def _loadmat(self):
        """
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects.
        Based off of implementations @ :
        https://github.com/GiocomoLab/TwoPUtils/blob/main/scanner_tools/sbx_utils.py
        https://github.com/losonczylab/sima/blob/0b16818d9ba47fe4aae6d4aad1a9735d16da00dc/sima/imaging_parameters.py
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

        info["frame_rate"] = np.int(
            info["resfreq"]
            / info["config"]["lines"]
            * (2 - info["scanmode"])
            * info["fov_repeats"]
        )
        # SIMA:
        info["nsamples"] = info["sz"][1] * info["recordsPerBuffer"] * info["nChan"] * 2
        # SIMA:
        if ("volscan" in info and info["volscan"] > 0) or (
            "volscan" not in info and len(info.get("otwave", []))
        ):
            info["nplanes"] = len(info["otwave"])
        else:
            info["nplanes"] = 1
        # SIMA:
        if info.get("scanbox_version", -1) >= 2:
            info["max_idx"] = (
                os.path.getsize(self.sbx_file_path) // info["nsamples"] - 1
            )
        else:
            info["max_idx"] = (
                os.path.getsize(self.sbx_file_path) // info["bytesPerBuffer"] * factor
                - 1
            )
        # SIMA: Fix for old scanbox versions
        if "sz" not in info:
            info["sz"] = np.array([512, 796])
        return info

    def _sbx_read(self):
        nrows = self._info["recordsPerBuffer"]
        ncols = self._info["sz"][1]
        nchannels = self._info["nChan"]
        nplanes = self._info["nplanes"]
        nframes = (self._info["max_idx"] + 1) // nplanes
        shape = (nchannels, ncols, nrows, nplanes, nframes)
        np_data = np.memmap(
            self.sbx_file_path, dtype="uint16", mode="r", shape=shape, order="F"
        )
        return np_data

    def get_frames(self, frame_idxs: ArrayType, channel: int = 0) -> np.array:
        frames_list = []
        for frame_no in frame_idxs:
            frames_list.append(self._data[channel, :, :, 0, frame_no])
        frame_out = np.stack(frames_list, axis=2).T.squeeze()
        return np.iinfo("uint16").max - frame_out

    def get_image_size(self) -> ArrayType:
        return self._info["sz"]

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
        raise NotImplementedError
