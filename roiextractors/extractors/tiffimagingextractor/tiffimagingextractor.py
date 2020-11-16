from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...extraction_tools import PathType, get_video_shape, check_get_frames_args
from ...imagingextractor import ImagingExtractor

try:
    import tifffile

    HAVE_TIFF = True
except:
    HAVE_TIFF = False


class TiffImagingExtractor(ImagingExtractor):
    extractor_name = 'TiffImaging'
    installed = HAVE_TIFF  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = "To use the TiffImagingExtractor install tifffile: \n\n pip install tifffile\n\n"  # error message when not installed

    def __init__(self, file_path: PathType, sampling_frequency,
                 channel_names=None):
        assert HAVE_TIFF, self.installation_mesg
        ImagingExtractor.__init__(self)
        self.file_path = Path(file_path)
        self._sampling_frequency = sampling_frequency
        self._channel_names = channel_names
        assert self.file_path.suffix in ['.tiff', '.tif', '.TIFF', '.TIF']

        with tifffile.TiffFile(self.file_path) as tif:
            self._num_channels = len(tif.series)

        # deal with multiple channels
        self._video = tifffile.memmap(self.file_path)
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
    def write_imaging(imaging, save_path, overwrite: bool = False, chunk_size=None, verbose=True):
        save_path = Path(save_path)
        assert save_path.suffix in ['.tiff', '.tif', '.TIFF', '.TIF'], "'save_path' file is not an .tiff file"

        if save_path.is_file():
            if not overwrite:
                raise FileExistsError("The specified path exists! Use overwrite=True to overwrite it.")
            else:
                save_path.unlink()

        if chunk_size is None:
            tifffile.imsave(save_path, imaging.get_video())
        else:
            num_frames = imaging.get_num_frames()
            # chunk size is not None
            n_chunk = num_frames // chunk_size
            if num_frames % chunk_size > 0:
                n_chunk += 1
            if verbose:
                chunks = tqdm(range(n_chunk), ascii=True, desc="Writing to .tiff file")
            else:
                chunks = range(n_chunk)
            with tifffile.TiffWriter(save_path) as tif:
                for i in chunks:
                    video = imaging.get_video(start_frame=i * chunk_size,
                                              end_frame=min((i + 1) * chunk_size, num_frames))
                    chunk_frames = np.squeeze(video)
                    tif.save(chunk_frames, contiguous=True, metadata=None)
