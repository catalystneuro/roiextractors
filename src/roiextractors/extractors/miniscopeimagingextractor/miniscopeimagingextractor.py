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
    The MiniscopeImagingExtractor consolidates multiple .avi video files from a Miniscope recording session.

    As a single continuous dataset. It reads configuration parameters from a metaData.json
    file and optionally loads timestamps from a timeStamps.csv file.
    This file is typically located in the root directory of the Miniscope recording
    session alongside the video files.
    The JSON file should contain at least the following key:
        - "frameRate": String containing the frame rate value (e.g., "20FPS", "30.0")
        - "deviceName": String representing the device name (e.g., "Miniscope", "MiniscopeV3", etc.)

    Notes
    -----
    - The function expects a "recordingStartTime" key in the metadata JSON, which contains start time details.
      If not present, the top-level JSON object is assumed to contain the time information.
    - The "msec" field in the metadata is converted from milliseconds to microseconds for compatibility with the datetime
      microsecond field.
    Additional metadata such as recording settings, device parameters, and session information may also be present.

    If folder_path is provided, the extractor expects the following file structure from a typical Miniscope recording:
    - miniscope folder/
      ├── metaData.json (required)
      ├── timeStamps.csv (optional)
      ├── video1.avi
      ├── video2.avi
      └── ...

    Parameters
    ----------
    folder_path : Optional[PathType], optional
        The folder path containing the Miniscope video (.avi) files, the metaData.json configuration file and potentially the timeStamps.csv file.
    file_paths : Optional[List[PathType]], optional
        List of .avi file paths to be processed. These files should be from the same
        recording session and will be concatenated in the order provided.
    configuration_file_path : Optional[PathType], optional
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

    >>> # If timestamps are available, provide the path
    >>> timestamps_path = "/path/to/timeStamps.csv"
    >>> extractor = MiniscopeImagingExtractor(file_paths, config_path, timestamps_path)

    >>> # Folder-based initialization (auto-detects .avi files, metaData.json and timeStamps.csv)
    >>> folder_path = "/path/to/miniscope_folder"
    >>> extractor = MiniscopeImagingExtractor(folder_path=folder_path)

    Notes
    -----
    For each video file, a _MiniscopeSingleVideoExtractor is created. These individual extractors
    are then combined into the MiniscopeImagingExtractor to handle the session's recordings
    as a unified, continuous dataset.

    Example of metaData.json content:
    -----------------------------------
    {
        "ROI": {
            "height": 608,
            "leftEdge": 0,
            "topEdge": 0,
            "width": 608
        },
        "compression": "FFV1",
        "deviceDirectory": "C:/data/Joe/Ca_EEG3/Ca_EEG3-4/2022_09_19/09_18_41/miniscope",
        "deviceID": 0,
        "deviceName": "miniscope",
        "deviceType": "Miniscope_V4_BNO",
        "ewl": 70,
        "frameRate": "30FPS",
        "framesPerFile": 1000,
        "gain": 3.5,
        "led0": 1
    }

    """

    def __init__(
        self,
        folder_path: Optional[PathType] = None,
        file_paths: Optional[list[PathType]] = None,
        configuration_file_path: Optional[PathType] = None,
        timestamps_path: Optional[PathType] = None,
    ):
        # Determine file paths and configuration file path based on folder_path or provided arguments
        if folder_path is not None:
            if file_paths is not None or configuration_file_path is not None:
                raise ValueError(
                    "When folder_path is provided, file_paths and configuration_file_path cannot be specified. "
                    "Use either folder_path alone or provide file_paths with configuration_file_path."
                )

            file_paths, configuration_file_path, timestamps_path = self._get_miniscope_files_from_direct_folder(
                folder_path
            )
        else:
            if file_paths is None or configuration_file_path is None:
                raise ValueError(
                    "When folder_path is not provided, both file_paths and configuration_file_path must be specified."
                )

        # Validate input files
        self.validate_miniscope_files(file_paths, configuration_file_path, timestamps_path)

        # Load configuration and extract sampling frequency
        self._miniscope_config = self.load_miniscope_config(configuration_file_path)

        self._miniscope_folder_path = Path(configuration_file_path).parent
        self._timestamps_path = (
            Path(timestamps_path) if timestamps_path is not None else self._miniscope_folder_path / "timeStamps.csv"
        )
        if not self._timestamps_path.exists():
            warnings.warn(
                f"`timeStamps.csv` file not found at {self._miniscope_folder_path}. Timestamps will be None. Set it with the `timestamps_path` if available."
            )
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
    def _get_miniscope_files_from_direct_folder(
        folder_path: PathType,
    ) -> tuple[list[PathType], PathType, Optional[PathType]]:
        """
        Retrieve Miniscope files from a folder containing .avi files directly.

        This function handles cases where .avi files and metaData.json are located
        directly in the specified folder without subfolders.

        Expected folder structure:
        ```
        folder/
        ├── 0.avi
        ├── 1.avi
        ├── 2.avi
        ├── metaData.json
        ├── timeStamps.csv
        └── headOrientation.csv
        ```

        Parameters
        ----------
        folder_path : PathType
            Path to the folder containing .avi files and metaData.json directly.

        Returns
        -------
        Tuple[List[PathType], PathType, Optional[PathType]]
            A tuple containing:
            - List of .avi file paths sorted naturally
            - Path to the configuration file (metaData.json)
            - Path to the timestamps file (timeStamps.csv), if it exists
            or None if it does not exist.

        Raises
        ------
        AssertionError
            If no .avi files or configuration files are found.
        ValueError
            If the file lists are empty or contain invalid file types.
        Warning
            If the timestamps file is not found, timestamps will be set to None.
        """
        natsort = get_package(package_name="natsort", installation_instructions="pip install natsort")

        folder_path = Path(folder_path)

        miniscope_avi_file_paths = natsort.natsorted(list(folder_path.glob("*.avi")))
        miniscope_config_files = natsort.natsorted(list(folder_path.glob("metaData.json")))
        miniscope_timestamps_files = natsort.natsorted(list(folder_path.glob("timeStamps.csv")))

        assert miniscope_avi_file_paths, f"No .avi files found in direct folder structure at '{folder_path}'"
        # check that the list of file paths follow the expected naming convention (0.avi, 1.avi, 2.avi, ...)
        for i, file_path in enumerate(miniscope_avi_file_paths):
            expected_file_name = f"{i}.avi"
            if file_path.name != expected_file_name:
                raise ValueError(
                    f"Unexpected file name '{file_path.name}'. Expected '{expected_file_name}'. "
                    "Ensure .avi files are named sequentially starting from 0 (e.g., 0.avi, 1.avi, 2.avi, ...)."
                )

        # check that the configuration file exists and is unique
        assert miniscope_config_files, f"No configuration file found at '{folder_path}', expected 'metaData.json'"
        assert len(miniscope_config_files) == 1, f"Multiple configuration files found at '{folder_path}'"
        configuration_file_path = miniscope_config_files[0]

        # timestamps file is optional
        if miniscope_timestamps_files:
            assert (
                len(miniscope_timestamps_files) == 1
            ), f"Multiple timestamps files found at '{folder_path}', expected only one 'timeStamps.csv'"
            timestamps_path = miniscope_timestamps_files[0]
        else:
            warnings.warn(
                f"No timestamps file found at '{folder_path}', expected 'timeStamps.csv'. Timestamps will be None."
            )
            timestamps_path = None

        return miniscope_avi_file_paths, configuration_file_path, timestamps_path

    @staticmethod
    def validate_miniscope_files(
        file_paths: List[PathType], configuration_file_path: PathType, timestamps_path: Optional[PathType] = None
    ) -> None:
        """
        Validate that the provided Miniscope files exist and are accessible.

        Parameters
        ----------
        file_paths : List[PathType]
            List of .avi file paths to validate.
        configuration_file_path : PathType
            Path to the configuration file to validate.
        timestamps_path : Optional[PathType], optional
            Path to the timestamps file to validate, by default None.

        Raises
        ------
        FileNotFoundError
            If any of the specified files do not exist.
        ValueError
            If the file lists are empty or contain invalid file types.
        """
        configuration_file_path = Path(configuration_file_path)
        if not configuration_file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {configuration_file_path}")

        if not configuration_file_path.suffix == ".json":
            raise ValueError(f"Configuration file must be a .json file, got: {configuration_file_path}")

        if not file_paths:
            raise ValueError("file_paths cannot be empty.")

        for file_path in file_paths:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Video file not found: {file_path}")
            if not file_path.suffix == ".avi":
                raise ValueError(f"Video files must be .avi files, got: {file_path}")

        if timestamps_path is not None:
            timestamps_path = Path(timestamps_path)
            if not timestamps_path.exists():
                raise FileNotFoundError(f"Timestamps file not found: {timestamps_path}")
            if not timestamps_path.suffix == ".csv":
                raise ValueError(f"Timestamps file must be a .csv file, got: {timestamps_path}")

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
        miniscope_device_name : str, optional
            The name of the Miniscope device subfolder. Default is "Miniscope".

    Notes
    -----
    This extractor is designed to handle the Tye Lab format, where multiple recordings
    are organized in timestamp subfolders, each containing a Miniscope subfolder.
    The expected folder structure is as follows:
    ```
    parent_folder/
    ├── 15_03_28/  (timestamp folder)
    │   ├── Miniscope/  (miniscope_device_name folder)
    │   │   ├── 0.avi
    │   │   ├── 1.avi
    │   │   └── metaData.json
    │   ├── BehavCam_2/
    │   └── metaData.json
    ├── 15_06_28/  (timestamp folder)
    │   ├── Miniscope/ (miniscope_device_name folder)
    │   │   ├── 0.avi
    │   │   └── metaData.json
    │   └── BehavCam_2/
    └── 15_12_28/  (timestamp folder)
        └── Miniscope/ (miniscope_device_name folder)
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
