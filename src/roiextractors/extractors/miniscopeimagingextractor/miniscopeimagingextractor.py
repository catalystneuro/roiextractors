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


def read_timestamps_from_csv_file(file_path: PathType) -> List[float]:
    """
    Retrieve the timestamps from a CSV file.

    Parameters
    ----------
    file_path : PathType
        Path to the CSV file containing the timestamps relative to the recording start.

    Returns
    -------
    timestamps
        The timestamps extracted from the CSV file.
    """
    import pandas as pd

    file_path = str(file_path)
    timestamps = pd.read_csv(file_path)["Time Stamp (ms)"].values.astype(float)
    timestamps /= 1000
    return timestamps


class MiniscopeImagingExtractor(MultiImagingExtractor):
    """
    The MiniscopeImagingExtractor consolidates multiple .avi video files from a Miniscope recording session
    as a single continuous dataset. It reads configuration parameters from a metaData.json
    file and optionally loads timestamps from a timeStamps.csv file.
    This file is typically located in the root directory of the Miniscope recording
    session alongside the video files.
    The JSON file should contain at least the following key:
        - "frameRate": String containing the frame rate value (e.g., "20FPS", "30.0")
        - "year", "month", "day", "hour", "minute", "second", "msec": Integers representing the recording start time. Either under the field "recordingStartTime" or at the top level of the JSON.
        - "miniscope": String representing the device name (e.g., "Miniscope", "MiniscopeV3", etc.)

    Notes:
    ------
    - The function expects a "recordingStartTime" key in the metadata JSON, which contains start time details.
      If not present, the top-level JSON object is assumed to contain the time information.
    - The "msec" field in the metadata is converted from milliseconds to microseconds for compatibility with the datetime
      microsecond field.
    Additional metadata such as recording settings, device parameters, and session information may also be present.

    The extractor expects the following file structure from a typical Miniscope recording:
    - miniscope folder/
      ├── metaData.json (required)
      ├── timeStamps.csv (optional)
      ├── video1.avi
      ├── video2.avi
      └── ...

    Parameters
    ----------
    file_paths : List[PathType]
        List of .avi file paths to be processed. These files should be from the same
        recording session and will be concatenated in the order provided.
    configuration_file_path : PathType
        Path to the metaData.json configuration file containing recording parameters.
        Usually located in the same directory as the .avi files.
    timestamps_path : Optional[PathType], optional
        Path to the timeStamps.csv file containing timestamps relative to the recording start.
        If not provided, the extractor will look for a timeStamps.csv file in the same directory
        as the configuration_file_path. If the file is not found, timestamps will be set to None.
        Default is None.

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

    >>> # If timestamps are available, provide the path
    >>> timestamps_path = "/path/to/timeStamps.csv"
    >>> extractor = MiniscopeImagingExtractor(file_paths, config_path, timestamps_path)

    Notes
    -----
    For each video file, a _MiniscopeSingleVideoExtractor is created. These individual extractors
    are then combined into the MiniscopeImagingExtractor to handle the session's recordings
    as a unified, continuous dataset.

    Examples of metaData.json content:
    -----------------------------------
    example 1:
    {
        "animalName": "Ca_EEG2-1",
        "baseDirectory": "C:/Users/CaiLab/Documents//Joe/Ca_EEG2/Ca_EEG2-1/2021_10_14/10_11_24",
        "cameras": [
        ],
        "day": 14,
        "experimentName": "Ca_EEG2",
        "hour": 10,
        "miniscopes": [
            "Miniscope"
        ],
        "minute": 11,
        "month": 10,
        "msec": 779,
        "msecSinceEpoch": 1634220684779,
        "researcherName": "Joe",
        "second": 24,
        "year": 2021
    }
    example 2:
    {
        "animalName": "",
        "baseDirectory": "C:/mData/2021_10_07/C6-J588_Disc5/15_03_28",
        "cameras": [
            "BehavCam 2"
        ],
        "experimentName": "",
        "miniscopes": [
            "Miniscope"
        ],
        "nameExpMouse": "C6-J588_Disc5",
        "recordingStartTime": {
            "day": 7,
            "hour": 15,
            "minute": 3,
            "month": 10,
            "msec": 635,
            "msecSinceEpoch": 1633644208635,
            "second": 28,
            "year": 2021
        },
        "researcherName": ""
    }

    """

    def __init__(
        self,
        file_paths: List[PathType],
        configuration_file_path: PathType,
        timestamps_path: Optional[PathType] = None,
    ):

        # Validate input files
        self.validate_miniscope_files(file_paths, configuration_file_path)

        # Load configuration and extract sampling frequency
        self._miniscope_config = self.load_miniscope_config(configuration_file_path)

        self.miniscope_folder_path = Path(configuration_file_path).parent
        self._timestamps_path = (
            Path(timestamps_path) if timestamps_path is not None else self.miniscope_folder_path / "timeStamps.csv"
        )
        if not self._timestamps_path.exists():
            warnings.warn(f"Timestamps file not found at {self._timestamps_path}. Timestamps will be None.")
            self._timestamps_path = None

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

    @staticmethod
    def validate_miniscope_files(file_paths: List[PathType], configuration_file_path: PathType) -> None:
        """
        Validate that the provided Miniscope files exist and are accessible.

        Parameters
        ----------
        file_paths : List[PathType]
            List of .avi file paths to validate.
        configuration_file_path : PathType
            Path to the configuration file to validate.

        Raises
        ------
        FileNotFoundError
            If any of the specified files do not exist.
        ValueError
            If the file lists are empty or contain invalid file types.
        """
        if not file_paths:
            raise ValueError("file_paths cannot be empty.")

        configuration_file_path = Path(configuration_file_path)
        if not configuration_file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {configuration_file_path}")

        if not configuration_file_path.suffix == ".json":
            raise ValueError(f"Configuration file must be a .json file, got: {configuration_file_path}")

        for file_path in file_paths:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Video file not found: {file_path}")
            if not file_path.suffix == ".avi":
                raise ValueError(f"Video files must be .avi files, got: {file_path}")

    @staticmethod
    def load_miniscope_config(configuration_file_path: PathType) -> dict:
        """
        Load and parse the Miniscope configuration file.

        Parameters
        ----------
        configuration_file_path : PathType
            Path to the metaData.json configuration file.

        Returns
        -------
        dict
            Parsed configuration data from the JSON file.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        json.JSONDecodeError
            If the configuration file is not valid JSON.
        """
        configuration_file_path = Path(configuration_file_path)

        if not configuration_file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {configuration_file_path}")

        try:
            with open(configuration_file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in configuration file {configuration_file_path}: {e}", e.doc, e.pos
            )

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> np.ndarray | None:

        if self._timestamps_path is None:
            warnings.warn("Timestamps file not provided or not found. Returning None for timestamps.")
            return None

        # Set defaults
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self.get_num_samples()
        # Read timestamps from CSV file
        native_timestamps = read_timestamps_from_csv_file(self._timestamps_path)

        return native_timestamps[start_sample:end_sample]


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

    def __init__(self, folder_path: PathType, miniscope_device_name: str = "Miniscope"):
        """Create a MiniscopeMultiRecordingImagingExtractor instance from folder_path."""
        # Get file paths and configuration file path

        self.miniscope_device_name = miniscope_device_name
        self.folder_path = Path(folder_path)

        file_paths, configuration_file_path = self._get_miniscope_files_from_multi_recordings_subfolders(
            folder_path, miniscope_device_name
        )

        super().__init__(file_paths=file_paths, configuration_file_path=configuration_file_path)

    @staticmethod
    def _get_miniscope_files_from_multi_recordings_subfolders(
        folder_path: PathType, miniscope_device_name: str = "Miniscope"
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
        from pathlib import Path

        from ...extraction_tools import get_package

        natsort = get_package(package_name="natsort", installation_instructions="pip install natsort")

        folder_path = Path(folder_path)
        configuration_file_name = "metaData.json"

        miniscope_avi_file_paths = natsort.natsorted(list(folder_path.glob(f"*/{miniscope_device_name}/*.avi")))
        miniscope_config_files = natsort.natsorted(
            list(folder_path.glob(f"*/{miniscope_device_name}/{configuration_file_name}"))
        )

        assert miniscope_avi_file_paths, f"No Miniscope .avi files found at '{folder_path}'"
        assert miniscope_config_files, f"No Miniscope configuration files found at '{folder_path}'"

        return miniscope_avi_file_paths, miniscope_config_files[0]

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> np.ndarray | None:
        """
        Retrieve timestamps for multiple recordings in a multi-recordings folder structure.

        Returns
        -------
        np.ndarray | None
            An array of floats representing the timestamps for the recordings.

        Raises
        ------
        AssertionError
            If no time files are found.
        """
        from .miniscope_utils import get_recording_start_times_for_multi_recordings

        natsort = get_package(package_name="natsort", installation_instructions="pip install natsort")

        time_file_name = "timeStamps.csv"

        timestamps_file_paths = natsort.natsorted(
            list(self.folder_path.glob(f"*/{self.miniscope_device_name}/{time_file_name}"))
        )

        assert timestamps_file_paths, f"No time files found at '{self.folder_path}'"

        recording_start_times = get_recording_start_times_for_multi_recordings(folder_path=self.folder_path)
        timestamps = []
        for file_ind, file_path in enumerate(timestamps_file_paths):
            timestamps_per_file = read_timestamps_from_csv_file(file_path=file_path)
            if recording_start_times:
                offset = (recording_start_times[file_ind] - recording_start_times[0]).total_seconds()
                timestamps_per_file += offset

            timestamps.extend(timestamps_per_file)

        # Set defaults
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self.get_num_samples()

        return np.array(timestamps)[start_sample:end_sample]


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
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        return None
