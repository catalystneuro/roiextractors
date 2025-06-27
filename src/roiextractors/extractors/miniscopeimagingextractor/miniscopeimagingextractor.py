"""MiniscopeImagingExtractor class.

Classes
-------
MiniscopeImagingExtractor
    An ImagingExtractor for the Miniscope video (.avi) format.
"""

import json
import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ...extraction_tools import DtypeType, PathType, get_package
from ...imagingextractor import ImagingExtractor
from ...multiimagingextractor import MultiImagingExtractor


class MiniscopeMultiRecordingImagingExtractor(MultiImagingExtractor):
    """
    ImagingExtractor processes multiple separate Miniscope recordings within the same session.

    Important, this extractor consolidates the recordings as a single continuous dataset.

    Expected directory structure:

    .
    ├── C6-J588_Disc5
    │   ├── 15_03_28  (timestamp)
    │   │   ├── BehavCam_2
    │   │   ├── metaData.json
    │   │   └── Miniscope
    │   ├── 15_06_28 (timestamp)
    │   │   ├── BehavCam_2
    │   │   ├── metaData.json
    │   │   └── Miniscope
    │   └── 15_07_58 (timestamp)
    │       ├── BehavCam_2
    │       ├── metaData.json
    │       └── Miniscope
    └──

    Where the Miniscope folders contain a collection of .avi files and a metaData.json file.

    For each video file, a _MiniscopeSingleVideoExtractor is created. These individual extractors
    are then combined into the MiniscopeMultiRecordingImagingExtractor to handle the session's recordings
    as a unified, continuous dataset.
    """

    extractor_name = "MiniscopeMultiRecordingImagingExtractor"
    mode = "folder"

    def __init__(self, folder_path: PathType):
        """Create a MiniscopeMultiRecordingImagingExtractor instance from a folder path.

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
            extractor = _MiniscopeSingleVideoExtractor(file_path=file_path)
            extractor._sampling_frequency = self._sampling_frequency
            imaging_extractors.append(extractor)

        super().__init__(imaging_extractors=imaging_extractors)


# Temporary renaming to keep backwards compatibility
class MiniscopeImagingExtractor(MiniscopeMultiRecordingImagingExtractor):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "MiniscopeImagingExtractor is unstable and might change its signature. "
            "Please use MiniscopeMultiRecordingImagingExtractor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class _MiniscopeSingleVideoExtractor(ImagingExtractor):
    """An auxiliary extractor to get data from a single Miniscope video (.avi) file.

    This format consists of a single video (.avi)
    Multiple _MiniscopeSingleVideoExtractor are combined by downstream extractors to extract the data
    """

    extractor_name = "_MiniscopeSingleVideo"

    def __init__(self, file_path: PathType):
        """Create a _MiniscopeSingleVideoExtractor instance from a file path.

        Parameters
        ----------
        file_path: PathType
           The file path to the Miniscope video (.avi) file.
        """
        from neuroconv.datainterfaces.behavior.video.video_utils import (
            VideoCaptureContext,
        )

        self._video_capture = VideoCaptureContext
        self._cv2 = get_package(package_name="cv2", installation_instructions="pip install opencv-python-headless")
        self.file_path = file_path
        super().__init__()

        with self._video_capture(file_path=str(file_path)) as video_obj:
            self._num_samples = video_obj.get_video_frame_count()
            self._image_size = video_obj.get_frame_shape()
            self._dtype = video_obj.get_video_frame_dtype()

        self._sampling_frequency = None

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_num_frames(self) -> int:
        """Get the number of frames in the video.

        Returns
        -------
        num_frames: int
            Number of frames in the video.

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_num_samples() instead.
        """
        warnings.warn(
            "get_num_frames() is deprecated and will be removed in or after September 2025. "
            "Use get_num_samples() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_num_samples()

    def get_num_channels(self) -> int:
        return 1

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._image_size[:-1]

    def get_image_size(self) -> Tuple[int, int]:
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._image_size[:-1]

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_dtype(self) -> DtypeType:
        return self._dtype

    def get_channel_names(self) -> List[str]:
        return ["OpticalChannel"]

    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
        end_sample = end_sample or self.get_num_samples()
        start_sample = start_sample or 0

        series = np.empty(shape=(end_sample - start_sample, *self.get_sample_shape()), dtype=self.get_dtype())
        with self._video_capture(file_path=str(self.file_path)) as video_obj:
            # Set the starting frame position
            video_obj.current_frame = start_sample
            for frame_number in range(end_sample - start_sample):
                frame = next(video_obj)
                series[frame_number] = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2GRAY)

        return series

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: Optional[int] = 0
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

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_series() instead.
        """
        warnings.warn(
            "get_video() is deprecated and will be removed in or after September 2025. " "Use get_series() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if channel != 0:
            raise NotImplementedError(
                f"The {self.extractor_name}Extractor does not currently support multiple color channels."
            )

        return self.get_series(start_sample=start_frame, end_sample=end_frame)

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # Miniscope videos do not have native timestamps
        return None
