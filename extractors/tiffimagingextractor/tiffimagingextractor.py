import numpy as np
from pathlib import Path
from ..imagingextractor import ImagingExtractor
from ..extraction_tools import ArrayType, PathType


class TiffImagingExtractor(ImagingExtractor):
    def __init__(self, filepath: PathType, sampling_frequency=None,
                 channel_names=None):

        ImagingExtractor.__init__(self)
        self.filepath = Path(filepath)
        self._sampling_frequency = sampling_frequency
        assert self.filepath.suffix in ['.tiff', '.tif']
        # TODO placeholder
        self._video = np.load(self.filepath)
        self._channel_names = channel_names

        if len(self._video.shape) == 3:
            # 1 channel
            self._num_channels = 1
            self._num_frames, self._size_x, self._size_y = self._video.shape
            if channel_names is not None:
                if isinstance(channel_names, str):
                    self._channel_names = [channel_names]
            self._video = self._video[np.newaxis, :]
        else:
            # more channels
            # TODO deal with multiple channels properly
            self._num_channels, self._num_frames, self._size_x, self._size_y = self._video.shape
        if self._channel_names is not None:
            assert len(self._channel_names) == self._num_channels, "'channel_names' length is different than number " \
                                                                   "of channels"
        else:
            self._channel_names = [f'channel_{ch}' for ch in range(self._num_channels)]

    def get_frame(self, frame_idx):
        assert frame_idx < self.get_num_frames()
        return self._video[frame_idx]

    def get_frames(self, frame_idxs):
        assert np.all(frame_idxs < self.get_num_frames())
        planes = np.zeros((len(frame_idxs), self._size_x, self._size_y))
        for i, frame_idx in enumerate(frame_idxs):
            plane = self._video[frame_idx]
            planes[i] = plane
        return planes

    # TODO make decorator to check and correct inputs
    def get_video(self, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        end_frame = min(end_frame, self.get_num_frames())

        video = self._video[start_frame: end_frame]

        return video

    def get_image_size(self):
        return [self._size_x, self._size_y]

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_dtype(self):
        return self._video.dtype

    def get_channel_names(self):
        '''List of  channels in the recoding.

        Returns
        -------
        channel_names: list
            List of strings of channel names
        '''
        self._channel_names

    def get_num_channels(self):
        '''Total number of active channels in the recording

        Returns
        -------
        no_of_channels: int
            integer count of number of channels
        '''
        self._num_channels

    def write_imaging(imaging, savepath):
        pass
