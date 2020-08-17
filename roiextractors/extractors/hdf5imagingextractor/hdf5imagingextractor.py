import numpy as np
from pathlib import Path
from ...imagingextractor import ImagingExtractor
from ...extraction_tools import get_video_shape

try:
    import h5py
    HAVE_H5 = True
except ImportError:
    HAVE_H5 = False


class Hdf5ImagingExtractor(ImagingExtractor):
    extractor_name = 'Hdf5Imaging'
    installed = HAVE_H5  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the Hdf5 Extractor run:\n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self, file_path, mov_field='mov', sampling_frequency=None, start_time=None, metadata=None,
                 channel_names=None):
        assert HAVE_H5, self.installation_mesg
        ImagingExtractor.__init__(self)
        self.filepath = Path(file_path)
        self._sampling_frequency = sampling_frequency
        self._mov_field = mov_field
        assert self.filepath.suffix in ['.h5', '.hdf5'], ""
        self._channel_names = channel_names

        with h5py.File(file_path, "r") as f:
            if 'mov' in f.keys():
                self._video = f[self._mov_field]
            else:
                raise Exception(f"{file_path} does not contain the 'mov' dataset")

        if sampling_frequency is None:
            assert 'fr' in f.keys(), "sampling frequency information is unavailable!"
            self._sampling_frequency = self._video.attrs["fr"]
        else:
            self._sampling_frequency = sampling_frequency

        if start_time is None:
            if 'start_time' in f.keys():
                self._start_time = self._video.attrs["start_time"]
        else:
            self._start_time = start_time

        if metadata is None:
            if 'metadata' in f.keys():
                self.metadata = self._video.attrs["metadata"]
        else:
            self.metadata = metadata

        self._num_channels, self._num_frames, self._size_x, self._size_y = get_video_shape(self._video)

        if len(self._video.shape) == 3:
            # check if this converts to np.ndarray
            self._video = self._video[np.newaxis, :]

        if self._channel_names is not None:
            assert len(self._channel_names) == self._num_channels, "'channel_names' length is different than number " \
                                                                   "of channels"
        else:
            self._channel_names = [f'channel_{ch}' for ch in range(self._num_channels)]

        self._kwargs = {'file_path': str(Path(file_path).absolute()), 'mov_field': mov_field,
                        'sampling_frequency': sampling_frequency, 'channel_names': channel_names}

    def get_frame(self, frame_idx, channel=0):
        assert frame_idx < self.get_num_frames()
        return self._video[channel, frame_idx]

    def get_frames(self, frame_idxs, channel=0):
        frame_idxs = np.array(frame_idxs)
        assert np.all(frame_idxs < self.get_num_frames())
        sorted_frame_idxs, sorting_inverse = np.sort(frame_idxs, return_inverse=True)
        return self._video[channel, sorted_frame_idxs][:, sorting_inverse]

    # TODO make decorator to check and correct inputs
    def get_video(self, start_frame=None, end_frame=None, channel=0):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        end_frame = min(end_frame, self.get_num_frames())

        video = self._video[channel, start_frame: end_frame]

        return video

    def get_image_size(self):
        return [self._size_x, self._size_y]

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_channel_names(self):
        return self._channel_names

    def get_num_channels(self):
        return self._num_channels

    @staticmethod
    def write_imaging(imaging, savepath):
        pass
