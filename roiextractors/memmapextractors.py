import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .extraction_tools import PathType, check_get_frames_args
from .imagingextractor import ImagingExtractor


class MemmapImagingExtractor(ImagingExtractor):
    def __init__(self, imaging_extractor: ImagingExtractor, save_path: PathType = None,
                 verbose: bool = False):
        ImagingExtractor.__init__(self)
        self.imaging = imaging_extractor
        tmp_folder = self.get_tmp_folder()

        if save_path is None:
            self._is_tmp = True
            self._tmp_file = tempfile.NamedTemporaryFile(suffix=".dat", dir=tmp_folder).name
        else:
            save_path = Path(save_path)
            if save_path.suffix != '.dat' and save_path.suffix != '.bin':
                save_path = save_path.with_suffix('.dat')
            if not save_path.parent.is_dir():
                os.makedirs(save_path.parent)
            self._is_tmp = False
            self._tmp_file = save_path

        self._save_memmap_video(verbose)
        self._verbose = verbose

    def __del__(self):
        if self._is_tmp:
            try:
                os.remove(self._tmp_file)
            except Exception as e:
                print("Unable to remove temporary file", e)

    def _save_memmap_video(self, verbose=False):
        save_path = Path(self._tmp_file)
        self._video = np.memmap(save_path, shape=(self.imaging.get_num_channels(),
                                                  self.imaging.get_num_frames(),
                                                  self.imaging.get_image_size()[0],
                                                  self.imaging.get_image_size()[1]),
                                dtype=self.imaging.get_dtype(), mode='w+')

        if verbose:
            for ch in range(self.imaging.get_num_channels()):
                print(f"Saving channel {ch}")
                for i in tqdm(range(self.imaging.get_num_frames())):
                    plane = self.imaging.get_frames(i, channel=ch)
                    self._video[ch, i] = plane
        else:
            for ch in range(self.imaging.get_num_channels()):
                for i in range(self.imaging.get_num_frames()):
                    plane = self.imaging.get_frames(i, channel=ch)
                    self._video[ch, i] = plane

    @property
    def filename(self):
        return str(self._tmp_file)

    def move_to(self, save_path):
        save_path = Path(save_path)
        if save_path.suffix != '.dat' and save_path.suffix != '.bin':
            save_path = save_path.with_suffix('.dat')
        if not save_path.parent.is_dir():
            os.makedirs(save_path.parent)
        shutil.move(self._tmp_file, str(save_path))
        self._tmp_file = str(save_path)
        self._video = np.memmap(save_path, shape=(self.imaging.get_num_channels(),
                                                  self.imaging.get_num_frames(),
                                                  self.imaging.get_image_size()[0],
                                                  self.imaging.get_image_size()[0]),
                                dtype=self.imaging.get_dtype(), mode='r')

    @check_get_frames_args
    def get_frames(self, frame_idxs, channel=0):
        return self._video[channel, frame_idxs]

    def get_image_size(self):
        return self.imaging.get_image_size()

    def get_num_frames(self):
        return self.imaging.get_num_frames()

    def get_sampling_frequency(self):
        return self.imaging.get_sampling_frequency()

    def get_channel_names(self):
        """List of  channels in the recoding.

        Returns
        -------
        channel_names: list
            List of strings of channel names
        """
        return self.imaging.get_channel_names()

    def get_num_channels(self):
        """Total number of active channels in the recording

        Returns
        -------
        no_of_channels: int
            integer count of number of channels
        """
        return self.imaging.get_channel_names()
