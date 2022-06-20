from pathlib import Path
import warnings

import numpy as np
from tqdm import tqdm

from ...extraction_tools import (
    PathType,
    get_video_shape,
    check_get_frames_args,
    FloatType,
    ArrayType,
    raise_multi_channel_or_depth_not_implemented,
)

from typing import Tuple
from ...imagingextractor import ImagingExtractor

try:
    import tifffile

    HAVE_TIFF = True
except ImportError:
    HAVE_TIFF = False


class TiffImagingExtractor(ImagingExtractor):
    extractor_name = "TiffImaging"
    installed = HAVE_TIFF  # check at class level if installed or not
    is_writable = True
    mode = "file"
    installation_mesg = "To use the TiffImagingExtractor install tifffile: \n\n pip install tifffile\n\n"

    def __init__(self, file_path: PathType, sampling_frequency: FloatType):
        assert HAVE_TIFF, self.installation_mesg
        ImagingExtractor.__init__(self)
        self.file_path = Path(file_path)
        self._sampling_frequency = sampling_frequency
        assert self.file_path.suffix in [".tiff", ".tif", ".TIFF", ".TIF"]

        with tifffile.TiffFile(self.file_path) as tif:
            self._num_channels = len(tif.series)

        try:
            self._video = tifffile.memmap(self.file_path, mode="r")
        except ValueError:
            warnings.warn("memmap of TIFF file could not be established. Reading entire matrix into memory.")
            with tifffile.TiffFile(self.file_path) as tif:
                self._video = tif.asarray()

        shape = self._video.shape
        if len(shape) == 3:
            self._num_frames, self._num_rows, self._num_cols = shape
            self._num_channels = 1
        else:
            raise_multi_channel_or_depth_not_implemented(extractor_name=self.extractor_name)

        self._kwargs = {
            "file_path": str(Path(file_path).absolute()),
            "sampling_frequency": sampling_frequency,
        }

    @check_get_frames_args
    def get_frames(self, frame_idxs, channel: int = 0):
        return self._video[frame_idxs, ...]

    def get_image_size(self) -> Tuple[int, int]:
        return (self._num_rows, self._num_cols)

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_num_channels(self):
        return self._num_channels

    def get_channel_names(self):
        pass

    @staticmethod
    def write_imaging(imaging, save_path, overwrite: bool = False, chunk_size=None, verbose=True):
        save_path = Path(save_path)
        assert save_path.suffix in [
            ".tiff",
            ".tif",
            ".TIFF",
            ".TIF",
        ], "'save_path' file is not an .tiff file"

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
                    video = imaging.get_video(
                        start_frame=i * chunk_size,
                        end_frame=min((i + 1) * chunk_size, num_frames),
                    )
                    chunk_frames = np.squeeze(video)
                    tif.save(chunk_frames, contiguous=True, metadata=None)
