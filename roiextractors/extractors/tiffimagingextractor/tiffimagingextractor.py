import numpy as np
from pathlib import Path
from ...imagingextractor import ImagingExtractor
from ...extraction_tools import PathType, get_video_shape, check_get_frames_args

try:
    import tiffile
    HAVE_TIFF = True
except:
    HAVE_TIFF = False


class TiffImagingExtractor(ImagingExtractor):
    extractor_name = 'TiffImaging'
    installed = HAVE_TIFF  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the TiffImagingExtractor install tifffile: \n\n pip install tiffile\n\n"  # error message when not installed

    def __init__(self, file_path: PathType, sampling_frequency,
                 channel_names=None):
        assert HAVE_TIFF, self.installation_mesg
        ImagingExtractor.__init__(self)
        self.file_path = Path(file_path)
        self._sampling_frequency = sampling_frequency
        self._channel_names = channel_names
        assert self.file_path.suffix in ['.tiff', '.tif', '.TIF', '.TIFF']

        with tiffile.TiffFile(self.file_path) as tif:
            self._num_channels = len(tif.series)

        # deal with multiple channels
        self._video = tiffile.memmap(self.file_path)
        self._num_channels, self._num_frames, self._size_x, self._size_y = get_video_shape(self._video)

        if len(self._video.shape) == 3:
            # check if this converts to np.ndarray
            self._video = self._video[np.newaxis, :]

        if self._channel_names is not None:
            assert len(self._channel_names) == self._num_channels, "'channel_names' length is different than number " \
                                                                   "of channels"
        else:
            self._channel_names = [f'channel_{ch}' for ch in range(self._num_channels)]

        self._kwargs = {'file_path': str(Path(file_path).absolute()),
                        'sampling_frequency': sampling_frequency, 'channel_names': channel_names}

    @check_get_frames_args
    def get_frames(self, frame_idxs, channel=0):
        return self._video[channel, frame_idxs]

    def get_image_size(self):
        return [self._size_x, self._size_y]

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_channel_names(self):
        """List of  channels in the recoding.

        Returns
        -------
        channel_names: list
            List of strings of channel names
        """
        return self._channel_names

    def get_num_channels(self):
        """Total number of active channels in the recording

        Returns
        -------
        no_of_channels: int
            integer count of number of channels
        """
        return self._num_channels

    @staticmethod
    def write_imaging(imaging, savepath):
        pass
