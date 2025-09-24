"""MiniscopeImagingExtractor class.

Classes
-------
MiniscopeImagingExtractor
    An ImagingExtractor for the Miniscope video (.avi) format.
"""

import re
import warnings
from typing import List, Optional, Tuple

import numpy as np

from .miniscope_utils import load_miniscope_config, validate_miniscope_files
from ...extraction_tools import DtypeType, PathType, get_package
from ...imagingextractor import ImagingExtractor
from ...multiimagingextractor import MultiImagingExtractor


class MiniscopeImagingExtractor(MultiImagingExtractor):
    """
    MiniscopeImagingExtractor processes .avi files within the same session.

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
    >>> extractor = MiniscopeImagingExtractor(file_paths, config_path)

    >>> # Using utility function for automatic discovery
    >>> from .miniscope_utils import get_miniscope_files_from_folder
    >>> file_paths, config_path = get_miniscope_files_from_folder("/path/to/folder")
    >>> extractor = MiniscopeImagingExtractor(file_paths, config_path)

    Notes
    -----
    For each video file, a _MiniscopeSingleVideoExtractor is created. These individual extractors
    are then combined into the MiniscopeImagingExtractor to handle the session's recordings
    as a unified, continuous dataset.
    """

    def __init__(self, file_paths: List[PathType], configuration_file_path: PathType):
        """Create a MiniscopeImagingExtractor instance from file paths."""
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
class MiniscopeMultiRecordingImagingExtractor(MiniscopeImagingExtractor):
    """
    MiniscopeMultiRecordingImagingExtractor processes multiple separate Miniscope recordings within the same session.

    This extractor consolidates the recordings as a single continuous dataset.

    Parameters
    ----------
        folder_path : PathType
            The folder path containing the Miniscope video (.avi) files and the metaData.json configuration file.

    Notes
    -----
    This extractor is designed to handle the Tye Lab format, where multiple recordings
    are organized in timestamp subfolders, each containing a Miniscope subfolder.
    The expected folder structure is as follows:
    ```
    parent_folder/
    ├── 15_03_28/  (timestamp folder)
    │   ├── Miniscope/
    │   │   ├── 0.avi
    │   │   ├── 1.avi
    │   │   └── metaData.json
    │   ├── BehavCam_2/
    │   └── metaData.json
    ├── 15_06_28/  (timestamp folder)
    │   ├── Miniscope/
    │   │   ├── 0.avi
    │   │   └── metaData.json
    │   └── BehavCam_2/
    └── 15_12_28/  (timestamp folder)
        └── Miniscope/
            ├── 0.avi
            └── metaData.json
    ```
    This extractor will automatically find all the .avi files and the metaData.json configuration file
    within the specified folder and its subfolders, and create a _MiniscopeSingleVideoExtractor for each .avi file.
    The individual extractors are then combined into the MiniscopeMultiRecordingImagingExtractor to handle
    the session's recordings as a unified, continuous dataset.
    """

    extractor_name = "MiniscopeMultiRecordingImagingExtractor"

    def __init__(self, folder_path: PathType):
        """Create a MiniscopeMultiRecordingImagingExtractor instance from folder_path."""
        # Get file paths and configuration file path
        file_paths, configuration_file_path = self._get_miniscope_files_from_multi_recordings_subfolders(folder_path)

        super().__init__(file_paths=file_paths, configuration_file_path=configuration_file_path)

    @staticmethod
    def _get_miniscope_files_from_multi_recordings_subfolders(
        folder_path: PathType, miniscopeDeviceName: str = "Miniscope"
    ) -> Tuple[List[PathType], PathType]:
        """
        Retrieve Miniscope files from a multi-session folder structure.

        This function handles the Tye Lab format where multiple recordings
        are organized in timestamp subfolders, each containing a Miniscope subfolder.

        Expected folder structure:
        ```
        parent_folder/
        ├── 15_03_28/  (timestamp folder)
        │   ├── Miniscope/
        │   │   ├── 0.avi
        │   │   ├── 1.avi
        │   │   └── metaData.json
        │   ├── BehavCam_2/
        │   └── metaData.json
        ├── 15_06_28/  (timestamp folder)
        │   ├── Miniscope/
        │   │   ├── 0.avi
        │   │   └── metaData.json
        │   └── BehavCam_2/
        └── 15_12_28/  (timestamp folder)
            └── Miniscope/
                ├── 0.avi
                └── metaData.json
        ```

        Parameters
        ----------
        folder_path : PathType
            Path to the parent folder containing timestamp subfolders.
        miniscopeDeviceName : str, optional
            Name of the Miniscope device subfolder. Defaults to "Miniscope".

        Returns
        -------
        Tuple[List[PathType], PathType]
            A tuple containing:
            - List of .avi file paths sorted naturally
            - Path to the first configuration file found (metaData.json)

        Raises
        ------
        AssertionError
            If no .avi files or configuration files are found.
        """
        from ...extraction_tools import get_package
        from pathlib import Path

        natsort = get_package(package_name="natsort", installation_instructions="pip install natsort")

        folder_path = Path(folder_path)
        configuration_file_name = "metaData.json"

        miniscope_avi_file_paths = natsort.natsorted(list(folder_path.glob(f"*/{miniscopeDeviceName}/*.avi")))
        miniscope_config_files = natsort.natsorted(
            list(folder_path.glob(f"*/{miniscopeDeviceName}/{configuration_file_name}"))
        )

        assert miniscope_avi_file_paths, f"No Miniscope .avi files found at '{folder_path}'"
        assert miniscope_config_files, f"No Miniscope configuration files found at '{folder_path}'"

        return miniscope_avi_file_paths, miniscope_config_files[0]


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
