import json
import re
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from natsort import natsorted

from neuroconv.datainterfaces.behavior.video.video_utils import VideoCaptureContext

from ...imagingextractor import ImagingExtractor
from ...multiimagingextractor import MultiImagingExtractor
from ...extraction_tools import PathType, DtypeType


class MiniscopeImagingExtractor(MultiImagingExtractor):
    extractor_name = "MiniscopeImaging"
    is_writable = True
    mode = "folder"

    def __init__(self, folder_path: PathType, configuration_file_name: str = "metaData.json"):
        """
        The imaging extractor for the Miniscope video (.avi) format.
        This format consists of video (.avi) file(s) and configuration files (.json).


        Parameters
        ----------
        folder_path: PathType
           The folder path that contains the Miniscope data.
        configuration_file_name: str, optional
            The name of the JSON configuration file, default is 'metaData.json'.
        """

        self.folder_path = Path(folder_path)

        miniscope_avi_file_paths = natsorted(list(self.folder_path.glob("*/Miniscope/*.avi")))
        assert miniscope_avi_file_paths, f"The Miniscope movies (.avi files) are missing from '{self.folder_path}'."
        miniscope_config_files = natsorted(list(self.folder_path.glob(f"*/Miniscope/{configuration_file_name}")))
        assert miniscope_config_files, f"The configuration files ({configuration_file_name} files) are missing from '{self.folder_path}'."

        # Set the sampling frequency from the configuration file
        with open(miniscope_config_files[0], newline="") as f:
            self._miniscope_config = json.loads(f.read())
        self._sampling_frequency = float(re.search(r'\d+', self._miniscope_config["frameRate"]).group())

        imaging_extractors = []
        for file_path in miniscope_avi_file_paths:
            extractor = _MiniscopeImagingExtractor(file_path=file_path)
            extractor._sampling_frequency = self._sampling_frequency
            imaging_extractors.append(extractor)

        super().__init__(imaging_extractors=imaging_extractors)


class _MiniscopeImagingExtractor(ImagingExtractor):
    extractor_name = "_MiniscopeImaging"

    def __init__(self, file_path: PathType):
        self.file_path = file_path
        super().__init__()

        with VideoCaptureContext(file_path=str(file_path)) as video_obj:
            self._num_frames = video_obj.get_video_frame_count()
            self._image_size = video_obj.get_frame_shape()
            self._dtype = video_obj.get_video_frame_dtype()

        self._sampling_frequency = None

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_num_channels(self) -> int:
        return 1

    def get_image_size(self) -> Tuple[int, int]:
        return self._image_size[:-1]

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_dtype(self) -> DtypeType:
        return self._dtype

    def get_channel_names(self) -> List[str]:
        # What are the name of the channels here?
        # this depends on whether these color channels are duplicated or not
        # TODO: check in minian how they are reading these files
        # maybe there is a casting rule from uint8 to 16
        # they are def. not separate optical channels
        return ["channel_0"]

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:

        end_frame = end_frame or self.get_num_frames()
        start_frame = start_frame or 0

        video = np.empty(shape=(end_frame - start_frame, *self.get_image_size()))
        with VideoCaptureContext(file_path=str(self.file_path)) as video_obj:
            # Set the starting frame position
            video_obj.current_frame = start_frame
            for frame_number in range(end_frame - start_frame):
                frame = next(video_obj)
                # grayscale conversion is based on minian (todo: add ref)
                video[frame_number] = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        return video
