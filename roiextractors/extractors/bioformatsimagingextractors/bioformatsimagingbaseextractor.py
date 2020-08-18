from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod

from ...imagingextractor import ImagingExtractor
from ...extraction_tools import ArrayType, PathType, check_get_frames_args, check_get_videos_args, get_video_shape

try:
    import javabridge
    import bioformats

    HAVE_BIOFORMATS = True
except:
    HAVE_BIOFORMATS = False


class BioformatsImagingExtractor(ImagingExtractor, ABC):
    def __init__(self, file_path):
        ImagingExtractor.__init__(self)
        self.file_path = Path(file_path)
        self._start_javabridge_vm()
        self._reader = bioformats.ImageReader(str(self.file_path))

        # read metadata
        self._read_metadata()
        self._validate_metadata()

    def __del__(self):
        self._reader.close()
        self._kill_javabridge_vm()

    @abstractmethod
    def _read_metadata(self):
        """
        This abstract method needs to be overridden to load the following fields from the metadata:

        self._size_x
        self._size_y
        self._size_z
        self._num_channels
        self._num_frames
        self._channel_names
        self._sampling_frequency
        self._dtype

        """
        pass

    def _validate_metadata(self):
        assert self._size_x is not None
        assert self._size_y is not None
        assert self._num_channels is not None
        assert self._num_frames is not None
        assert self._sampling_frequency is not None

    def _start_javabridge_vm(self):
        javabridge.start_vm(class_path=bioformats.JARS)

    def _kill_javabridge_vm(self):
        javabridge.kill_vm()

    @check_get_frames_args
    def get_frames(self, frame_idxs, channel=0):
        planes = np.zeros((len(frame_idxs), self._size_x, self._size_y))
        for i, frame_idx in enumerate(frame_idxs):
            plane = self._reader.read(t=frame_idx).T
            planes[i] = plane
        return planes

    # TODO make this memmap if too large
    @check_get_videos_args
    def get_video(self, start_frame=None, end_frame=None, channel=0):
        video = np.zeros((end_frame - start_frame, self._size_x, self._size_y))
        for i, frame_idx in enumerate(np.arange(start_frame, end_frame)):
            video[i] = self._reader.read(t=frame_idx).T
        return video

    def get_image_size(self):
        return np.array([self._size_x, self._size_y])

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_channel_names(self):
        return self._channel_names

    def get_num_channels(self):
        return self._num_channels
