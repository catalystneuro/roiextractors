"""MiniscopeImagingExtractor class.

Classes
-------
MiniscopeImagingExtractor
    An ImagingExtractor for the Miniscope video (.avi) format.
"""

import json
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

from ...imagingextractor import ImagingExtractor
from ...multiimagingextractor import MultiImagingExtractor
from ...extraction_tools import PathType, DtypeType, get_package


class MiniscopeImagingExtractor(MultiImagingExtractor):  # TODO: rename to MiniscopeMultiImagingExtractor
    """An ImagingExtractor for the Miniscope video (.avi) format.

    This format consists of video (.avi) file(s) and configuration files (.json).
    One _MiniscopeImagingExtractor is created for each video file and then combined into the MiniscopeImagingExtractor.
    """

    extractor_name = "MiniscopeImaging"
    is_writable = True
    mode = "folder"

    def __init__(self, folder_path: PathType):
        """Create a MiniscopeImagingExtractor instance from a folder path.

        Parameters
        ----------
        folder_path: PathType
           The folder path that contains the Miniscope data.
        """
        natsort = get_package(package_name="natsort", installation_instructions="pip install natsort")

        self.folder_path = Path(folder_path)

        configuration_file_name = "metaData.json"
        miniscope_avi_file_paths = natsort.natsorted(list(self.folder_path.glob("*/Miniscope/*.avi")))
        assert miniscope_avi_file_paths, f"The Miniscope movies (.avi files) are missing from '{self.folder_path}'."
        miniscope_config_files = natsort.natsorted(
            list(self.folder_path.glob(f"*/Miniscope/{configuration_file_name}"))
        )
        assert (
            miniscope_config_files
        ), f"The configuration files ({configuration_file_name} files) are missing from '{self.folder_path}'."

        # Set the sampling frequency from the configuration file
        with open(miniscope_config_files[0], newline="") as f:
            self._miniscope_config = json.loads(f.read())
        self._sampling_frequency = float(re.search(r"\d+", self._miniscope_config["frameRate"]).group())

        imaging_extractors = []
        for file_path in miniscope_avi_file_paths:
            extractor = _MiniscopeImagingExtractor(file_path=file_path)
            extractor._sampling_frequency = self._sampling_frequency
            imaging_extractors.append(extractor)

        super().__init__(imaging_extractors=imaging_extractors)


class _MiniscopeImagingExtractor(ImagingExtractor):
    """An ImagingExtractor for the Miniscope video (.avi) format.

    This format consists of a single video (.avi) file and configuration file (.json).
    Multiple _MiniscopeImagingExtractor are combined into the MiniscopeImagingExtractor for public access.
    """

    extractor_name = "_MiniscopeImaging"

    def __init__(self, file_path: PathType):
        """Create a _MiniscopeImagingExtractor instance from a file path.

        Parameters
        ----------
        file_path: PathType
           The file path to the Miniscope video (.avi) file.
        """
        from neuroconv.datainterfaces.behavior.video.video_utils import VideoCaptureContext

        self._video_capture = VideoCaptureContext
        self._cv2 = get_package(package_name="cv2", installation_instructions="pip install opencv-python-headless")
        self.file_path = file_path
        super().__init__()

        with self._video_capture(file_path=str(file_path)) as video_obj:
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
        return ["OpticalChannel"]

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        """Get the video frames.

        Parameters
        ----------
        start_frame: int, optional
            Start frame index (inclusive).
        end_frame: int, optional
            End frame index (exclusive).
        channel: int, optional
            Channel index.

        Returns
        -------
        video: numpy.ndarray
            The video frames.

        Notes
        -----
        The grayscale conversion is based on minian
        https://github.com/denisecailab/minian/blob/f64c456ca027200e19cf40a80f0596106918fd09/minian/utilities.py#LL272C12-L272C12
        """
        if channel != 0:
            raise NotImplementedError(
                f"The {self.extractor_name}Extractor does not currently support multiple color channels."
            )

        end_frame = end_frame or self.get_num_frames()
        start_frame = start_frame or 0

        video = np.empty(shape=(end_frame - start_frame, *self.get_image_size()), dtype=self.get_dtype())
        with self._video_capture(file_path=str(self.file_path)) as video_obj:
            # Set the starting frame position
            video_obj.current_frame = start_frame
            for frame_number in range(end_frame - start_frame):
                frame = next(video_obj)
                video[frame_number] = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2GRAY)

        return video
