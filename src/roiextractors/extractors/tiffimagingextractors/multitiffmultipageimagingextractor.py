from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from ...imagingextractor import ImagingExtractor
from ...extraction_tools import get_package
from ...utils import match_paths


class MultiTiffMultiPageImagingExtractor(ImagingExtractor):
    """A ImagingExtractor for multiple TIFF files that each have multiple pages."""

    extractor_name = "multi-tiff multi-page Imaging Extractor"
    is_writable = False

    def __init__(self, folder_path: str, pattern: str, sampling_frequency: float):
        """Create a MultiTiffMultiPageImagingExtractor instance.

        Parameters
        ----------
        folder_path : str
            List of path to each TIFF file.
        pattern : str
            F-string-style pattern to match the TIFF files.
        sampling_frequency : float
            The frequency at which the frames were sampled, in Hz.
        """

        super().__init__()
        self.folder_path = folder_path

        self.tif_paths = match_paths(folder_path, pattern)
        self._tifffile = get_package(package_name="tifffile", installation_instructions="pip install tifffile")

        page_tracker = []
        page_counter = 0
        for file_path in tqdm(self.tif_paths, "extracting page lengths"):
            with self._tifffile.TiffFile(file_path) as tif:
                page_tracker.append(page_counter)
                page_counter += len(tif.pages)
        self.page_tracker = np.array(page_tracker)

        page = tif.pages[0]

        self._num_frames = page_counter
        self._num_columns = page.imagewidth
        self._num_rows = page.imagelength

        self._sampling_frequency = sampling_frequency

        self._kwargs = {"folder_path": folder_path}

    def get_video(self, start_frame: int = None, end_frame: int = None, channel: Optional[int] = 0) -> np.ndarray:
        frame_idxs = np.arange(start_frame or 0, end_frame or self._num_frames)
        file_idxs = (
            np.searchsorted(self.page_tracker, frame_idxs, side="right") - 1
        )  # index of the file that contains the frame
        file_start_idxs = self.page_tracker[file_idxs]  # index of the first frame in the file
        frame_offset_idxs = frame_idxs - file_start_idxs  # index of the frame in the file
        # dict of file_idx: frame_offset_idxs
        index_dict = {x: frame_offset_idxs[file_idxs == x] for x in np.unique(file_idxs)}

        data = []
        for file_idx, frame_offset_idxs in index_dict.items():
            with self._tifffile.TiffFile(list(self.tif_paths)[file_idx]) as tif:
                for frame_offset_idx in frame_offset_idxs:
                    page = tif.pages[frame_offset_idx]
                    data.append(page.asarray())

        return np.array(data)

    def get_image_size(self) -> Tuple[int, int]:
        return self._num_rows, self._num_columns

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_num_channels(self):
        return 1

    def get_channel_names(self):
        return ["channel_0"]
