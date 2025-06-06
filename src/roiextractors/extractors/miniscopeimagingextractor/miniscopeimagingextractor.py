"""MiniscopeImagingExtractor class.

Classes
-------
MiniscopeImagingExtractor
    An ImagingExtractor for the Miniscope video (.avi) format.
"""

import re
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

import numpy as np

from ...imagingextractor import ImagingExtractor
from ...multiimagingextractor import MultiImagingExtractor
from ...extraction_tools import PathType, DtypeType, get_package
from .miniscope_utils import validate_miniscope_files, load_miniscope_config


class MiniscopeMultiRecordingImagingExtractor(MultiImagingExtractor):
    """
    ImagingExtractor processes multiple separate Miniscope recordings within the same session.

    This extractor consolidates the recordings as a single continuous dataset.

    Parameters
    ----------
    file_paths : List[PathType]
        List of .avi file paths to be processed. These files should be from the same
        recording session and will be concatenated in the order provided.
    configuration_file_path : PathType
        Path to the metaData.json configuration file containing recording parameters.

    Examples
    --------
    >>> # Direct file specification
    >>> file_paths = ["/path/to/video1.avi", "/path/to/video2.avi"]
    >>> config_path = "/path/to/metaData.json"
    >>> extractor = MiniscopeMultiRecordingImagingExtractor(file_paths, config_path)

    >>> # Using utility function for automatic discovery
    >>> from .miniscope_utils import get_miniscope_files_from_folder
    >>> file_paths, config_path = get_miniscope_files_from_folder("/path/to/folder")
    >>> extractor = MiniscopeMultiRecordingImagingExtractor(file_paths, config_path)

    Notes
    -----
    For each video file, a _MiniscopeSingleVideoExtractor is created. These individual extractors
    are then combined into the MiniscopeMultiRecordingImagingExtractor to handle the session's recordings
    as a unified, continuous dataset.
    """

    extractor_name = "MiniscopeMultiRecordingImagingExtractor"
    is_writable = True
    mode = "file"

    def __init__(self, file_paths: List[PathType], configuration_file_path: PathType):
        """Create a MiniscopeMultiRecordingImagingExtractor instance from file paths.

        Parameters
        ----------
        file_paths : List[PathType]
            List of .avi file paths to be processed.
        configuration_file_path : PathType
            Path to the metaData.json configuration file.
        """
        # Validate input files
        validate_miniscope_files(file_paths, configuration_file_path)

        # Load configuration and extract sampling frequency
        self._miniscope_config = load_miniscope_config(configuration_file_path)
        frame_rate_match = re.search(r"\d+", self._miniscope_config["frameRate"])
        if frame_rate_match is None:
            raise ValueError(f"Could not extract frame rate from configuration: {self._miniscope_config['frameRate']}")
        self._sampling_frequency = float(frame_rate_match.group())

        # Create individual extractors for each video file
        imaging_extractors = []
        for file_path in file_paths:
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
    """An auxiliar extractor to get data from a single Miniscope video (.avi) file.

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
        from neuroconv.datainterfaces.behavior.video.video_utils import VideoCaptureContext

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

        series = np.empty(shape=(end_sample - start_sample, *self.get_image_size()), dtype=self.get_dtype())
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
