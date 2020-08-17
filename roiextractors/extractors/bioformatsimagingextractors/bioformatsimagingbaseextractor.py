from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
from ...imagingextractor import ImagingExtractor

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

    def get_frame(self, frame_idx, channel=0):
        assert frame_idx < self.get_num_frames()
        plane = self._reader.read(t=frame_idx).T
        return plane

    def get_frames(self, frame_idxs, channel=0):
        frame_idxs = np.array(frame_idxs)
        assert np.all(frame_idxs < self.get_num_frames())
        planes = np.zeros((len(frame_idxs), self._size_x, self._size_y))
        for i, frame_idx in enumerate(frame_idxs):
            plane = self._reader.read(t=frame_idx).T
            planes[i] = plane
        return planes

    def get_video(self, start_frame=None, end_frame=None, channel=0):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        end_frame = min(end_frame, self.get_num_frames())

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
