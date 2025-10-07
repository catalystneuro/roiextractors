"""MiniscopeImagingExtractor class.

Classes
-------
MiniscopeImagingExtractor
    An ImagingExtractor for the Miniscope video (.avi) format.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from ...extraction_tools import DtypeType, PathType, get_package
from ...imagingextractor import ImagingExtractor
from ...multiimagingextractor import MultiImagingExtractor


def read_timestamps_from_csv_file(file_path: PathType) -> np.ndarray:
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
    Extractor for Miniscope calcium imaging data recorded with Miniscope-DAQ-QT-Software.

    This extractor consolidates multiple .avi video files from a Miniscope recording session
    into a single continuous dataset. It uses hardware-generated timestamps from timeStamps.csv
    for accurate timing information.

    The extractor works at the device folder level, where a typical Miniscope recording has
    the following structure:

    device_folder/ (e.g., "Miniscope", "HPC_miniscope1", "ACC_miniscope2")
    ├── 0.avi, 1.avi, 2.avi, ...    # Video files (FFV1 lossless codec)
    ├── timeStamps.csv               # Hardware-generated timestamps (required)
    ├── metaData.json                # Device configuration (optional, for reference)
    └── headOrientation.csv          # IMU data (optional)

    Key Features
    ------------
    - Automatically discovers and concatenates multiple .avi files from a recording
    - Calculates sampling frequency from hardware timestamps (timeStamps.csv)
    - Supports both folder-based and file-list initialization
    - Provides access to device and session metadata via static methods

    See Also
    --------
    MiniscopeMultiRecordingImagingExtractor : For multi-session recordings (Tye Lab format)
    """

    def __init__(
        self,
        folder_path: PathType | None = None,
        file_paths: list[PathType] | None = None,
        configuration_file_path: PathType | None = None,
        timestamps_path: PathType | None = None,
        *,
        sampling_frequency: float | None = None,
    ):
        """
        Initialize MiniscopeImagingExtractor.

        Parameters
        ----------
        folder_path : PathType | None, optional
            Path to the device folder containing the Miniscope recording files (.avi videos,
            metaData.json, and timeStamps.csv). This is the recommended way to initialize the
            extractor as it automatically discovers all necessary files.

            Note: This is the device-level folder (e.g., "HPC_miniscope1"), not the session
            folder (which may contain multiple device folders).
        file_paths : list[PathType] | None, optional
            List of .avi file paths to be processed. These files should be from the same
            recording session and will be concatenated in the order provided.
        configuration_file_path : PathType | None, optional
            **DEPRECATED**: This parameter is deprecated and will be removed in March 2026.
            **This parameter is no longer used and is ignored.**

            The device folder is now automatically inferred from file_paths. If provided,
            this parameter only triggers validation (checking if the file exists and is valid JSON),
            but has no functional effect on the extractor's behavior.

            Migration: Simply remove this parameter from your code. For example:
                # Old (deprecated):
                extractor = MiniscopeImagingExtractor(file_paths=files, configuration_file_path=config)
                # New (recommended):
                extractor = MiniscopeImagingExtractor(file_paths=files)

            If you need to read metaData.json files, use the private static methods:
            - MiniscopeImagingExtractor._read_device_folder_metadata(file_path)
            - MiniscopeImagingExtractor._read_session_folder_metadata(file_path)

            Deprecation rationale:
            - The metaData.json frameRate field is user-settable, not measured. The DAQ system
              attempts to achieve this rate but cannot guarantee it due to hardware limitations.
            - Analysis shows sampling frequency calculated from timeStamps.csv often differs from
              the configured frameRate, making metaData.json unreliable for timing information.
            - timeStamps.csv provides ground truth timing (hardware-generated, always produced by DAQ)
            - Requiring both file_paths and configuration_file_path is redundant when timestamps
              are available

            Default is None.
        timestamps_path : PathType | None, optional
            Path to the timeStamps.csv file containing timestamps relative to the recording start.
            If not provided, the extractor will look for a timeStamps.csv file in the same directory
            as the video files as a fallback. This file is required for accurate timing.
            Default is None.
        sampling_frequency : float | None, optional
            Explicit sampling frequency in Hz. If provided, this overrides the calculated value
            from timeStamps.csv. Only use this if you have a specific reason to override the
            measured sampling frequency (e.g., working with incomplete data).
            Default is None (calculate from timeStamps.csv).

        Examples
        --------
        >>> # Recommended: Folder-based initialization (auto-detects .avi files and timeStamps.csv)
        >>> folder_path = "/path/to/miniscope_folder"
        >>> extractor = MiniscopeImagingExtractor(folder_path=folder_path)

        >>> # Direct file specification (device folder inferred from file_paths)
        >>> file_paths = ["/path/to/device_folder/0.avi", "/path/to/device_folder/1.avi"]
        >>> extractor = MiniscopeImagingExtractor(file_paths=file_paths)

        >>> # With explicit timestamps path
        >>> timestamps_path = "/path/to/device_folder/timeStamps.csv"
        >>> extractor = MiniscopeImagingExtractor(file_paths=file_paths, timestamps_path=timestamps_path)

        >>> # Legacy usage with configuration_file_path (deprecated, triggers warning)
        >>> config_path = "/path/to/device_folder/metaData.json"
        >>> extractor = MiniscopeImagingExtractor(file_paths=file_paths, configuration_file_path=config_path)

        Notes
        -----
        For each video file, a _MiniscopeSingleVideoExtractor is created. These individual extractors
        are then combined into the MiniscopeImagingExtractor to handle the session's recordings
        as a unified, continuous dataset.
        """
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
            if file_paths is None:
                raise ValueError("When folder_path is not provided, file_paths must be specified.")

            # Emit deprecation warning if configuration_file_path is provided
            if configuration_file_path is not None:
                warnings.warn(
                    "The 'configuration_file_path' parameter is deprecated and will be removed in March 2026. "
                    "The device folder is now automatically inferred from file_paths. "
                    "To silence this warning, remove the configuration_file_path parameter from your code.",
                    FutureWarning,
                    stacklevel=2,
                )

        # Validate input files
        self.validate_miniscope_files(file_paths, timestamps_path)

        # Determine timestamps path
        # If not provided, look for timeStamps.csv in the same folder as the video files
        if timestamps_path is not None:
            self._timestamps_path = Path(timestamps_path)
        else:
            # Infer device folder from file_paths (all .avi files are in the same folder)
            device_folder = Path(file_paths[0]).parent
            self._timestamps_path = device_folder / "timeStamps.csv"

        # Determine sampling frequency with priority order:
        # 1. Explicit sampling_frequency parameter (user override)
        # 2. Calculated from timeStamps.csv (measured ground truth)
        # 3. Error if timeStamps.csv is missing
        if sampling_frequency is not None:
            # User explicitly provided - use it
            self._sampling_frequency = float(sampling_frequency)

            # Still validate timestamps exist (for completeness checking)
            if not self._timestamps_path.exists():
                warnings.warn(
                    f"timeStamps.csv not found at {self._timestamps_path}. "
                    f"Using user-provided sampling_frequency={sampling_frequency} Hz. "
                    f"Note: Miniscope recordings should always include timeStamps.csv."
                )
        elif self._timestamps_path.exists():
            # Calculate from timestamps (preferred path)
            self._sampling_frequency = self._calculate_sampling_frequency_from_timestamps(
                self._timestamps_path, num_samples=10_000
            )
        else:
            # Missing timestamps - fail with helpful message
            raise FileNotFoundError(
                f"timeStamps.csv not found at {self._timestamps_path}. "
                f"This file is required for accurate timing and should be automatically "
                f"generated by Miniscope-DAQ-QT-Software. Possible causes:\n"
                f"  - Incomplete recording\n"
                f"  - Files copied without timeStamps.csv\n"
                f"  - Non-standard data acquisition setup\n\n"
                f"If you have a specific reason to proceed without timestamps, "
                f"you can provide the sampling_frequency parameter explicitly:\n"
                f"  MiniscopeImagingExtractor(..., sampling_frequency=30.0)"
            )

        # Create individual extractors for each video file
        imaging_extractors = []
        for file_path in file_paths:
            extractor = _MiniscopeSingleVideoExtractor(file_path=file_path)
            extractor._sampling_frequency = self._sampling_frequency
            imaging_extractors.append(extractor)

        super().__init__(imaging_extractors=imaging_extractors)

    def _calculate_sampling_frequency_from_timestamps(
        self, timestamps_file_path: PathType, num_samples: int = 10_000
    ) -> float:
        """
        Calculate sampling frequency from timeStamps.csv.

        Parameters
        ----------
        timestamps_file_path : PathType
            Path to timeStamps.csv file
        num_samples : int
            Number of samples to read for calculation (default: 10_000).
            Reading fewer samples makes this fast (~50ms) while still accurate.

        Returns
        -------
        float
            Measured sampling frequency in Hz

        Notes
        -----
        - Reads only first `num_samples` rows for speed
        - Removes outliers (>3 std dev) to handle dropped frames
        - Expected precision: ±2-5% due to USB/OS jitter
        """
        import pandas as pd

        # Read only first N samples (fast)
        df = pd.read_csv(timestamps_file_path, nrows=num_samples)
        timestamps_ms = df["Time Stamp (ms)"].values
        timestamps_s = timestamps_ms / 1000.0

        if len(timestamps_s) < 2:
            raise ValueError(
                f"timeStamps.csv has insufficient data: only {len(timestamps_s)} frames. "
                f"Need at least 2 frames to calculate sampling frequency."
            )

        # Calculate intervals
        intervals = np.diff(timestamps_s)

        # Remove outliers (e.g., dropped frames, buffer overruns)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        valid_intervals = intervals[np.abs(intervals - mean_interval) < 3 * std_interval]

        if len(valid_intervals) == 0:
            raise ValueError(
                f"Could not calculate sampling frequency from {timestamps_file_path}: "
                f"all intervals appear to be outliers. The recording may be corrupted."
            )

        # Calculate sampling frequency
        mean_interval_clean = np.mean(valid_intervals)
        sampling_frequency = 1.0 / mean_interval_clean

        return sampling_frequency

    @staticmethod
    def _get_miniscope_files_from_direct_folder(
        folder_path: PathType,
    ) -> tuple[list[PathType], PathType, PathType | None]:
        """
        Retrieve Miniscope files from a folder containing .avi files directly.

        This function handles cases where .avi files and metaData.json are located
        directly in the specified folder without subfolders.

        Expected folder structure:
        ```
        device_folder_path/
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
        tuple[list[PathType], PathType, PathType | None]
            - list[PathType]: .avi file paths sorted naturally
            - PathType: path to the configuration file (metaData.json)
            - PathType | None: path to the timestamps file (timeStamps.csv) if present, otherwise None

        Raises
        ------
        AssertionError
            If no .avi files or configuration files are found.
        ValueError
            If file names do not follow the expected sequential naming convention.
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

        # timestamps file is optional
        if miniscope_timestamps_files:
            assert (
                len(miniscope_timestamps_files) == 1
            ), f"Multiple timestamps files found at '{folder_path}', expected only one 'timeStamps.csv'"
            timestamps_path = miniscope_timestamps_files[0]
        else:
            timestamps_path = None

        return miniscope_avi_file_paths, miniscope_config_files[0], timestamps_path

    @staticmethod
    def validate_miniscope_files(file_paths: list[PathType], timestamps_path: PathType | None = None) -> None:
        """
        Validate that the provided Miniscope files exist and are accessible.

        Parameters
        ----------
        file_paths : List[PathType]
            List of .avi file paths to validate.
        timestamps_path : PathType | None
            Path to the timestamps file to validate, by default None.

        Raises
        ------
        FileNotFoundError
            If any of the specified files do not exist.
        ValueError
            If the file lists are empty or contain invalid file types.

        Notes
        -----
        Only the .avi files are strictly required. The timestamps file is optional.
        """
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

        This is a generic method that can read any metaData.json file (session or device level).
        For more explicit metadata reading with specialized documentation, consider using:
        - _read_session_folder_metadata() for session-level metadata
        - _read_device_folder_metadata() for device-level metadata

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

    @staticmethod
    def _read_device_folder_metadata(metadata_file_path: PathType) -> dict:
        """
        Read device-level metaData.json containing Miniscope hardware configuration.

        The Miniscope-DAQ-QT-Software creates two levels of metaData.json files:
        1. Session-level: Contains experiment/animal info and recording start time
        2. Device-level: Contains hardware settings for each Miniscope/camera

        This method reads the DEVICE-level metadata.

        File Location
        -------------
        Device metadata is located in each device's subfolder:
        ```
        session_folder/
        ├── metaData.json                    # Session-level (use read_session_folder_metadata)
        ├── HPC_miniscope1/                  # Device folder
        │   ├── metaData.json               # Device-level (this method)
        │   ├── timeStamps.csv
        │   └── 0.avi, 1.avi, ...
        └── ACC_miniscope2/                  # Another device
            ├── metaData.json               # Device-level (this method)
            └── ...
        ```

        Device Metadata Contents
        ------------------------
        {
            "deviceType": "Miniscope_V4_BNO",       # Hardware version
            "deviceName": "HPC_miniscope1",         # User-assigned name
            "deviceID": 1,                          # Numeric ID
            "frameRate": "30FPS",                   # Configured frame rate
            "compression": "FFV1",                  # Video codec
            "framesPerFile": 1000,                  # Frames per AVI file
            "gain": "Medium",                       # Sensor gain setting
            "ewl": 70,                              # Excitation wavelength
            "led0": 24,                             # LED power (0-100)
            "ROI": {                                # Region of interest (optional)
                "height": 608,
                "width": 608,
                "leftEdge": 0,
                "topEdge": 0
            }
        }

        Parameters
        ----------
        metadata_file_path : PathType
            Path to the device metaData.json file
            (e.g., "session_folder/HPC_miniscope1/metaData.json")

        Returns
        -------
        dict
            Device metadata containing hardware configuration and acquisition parameters.
            Fields include:
            - deviceType: Hardware version (e.g., "Miniscope_V4_BNO", "Miniscope_V3")
            - deviceName: User-assigned device name
            - deviceID: Numeric device identifier
            - frameRate: Configured frame rate (NOTE: may not match actual rate)
            - compression: Video compression codec
            - framesPerFile: Number of frames per video file
            - gain: Sensor gain setting
            - ewl: Excitation wavelength (electrowetting lens position)
            - led0: LED power setting
            - ROI: Region of interest settings (if used)

        Raises
        ------
        FileNotFoundError
            If metaData.json file is not found

        Examples
        --------
        >>> metadata = MiniscopeImagingExtractor.read_device_folder_metadata(
        ...     "path/to/session/HPC_miniscope1/metaData.json"
        ... )
        >>> print(f"Device: {metadata['deviceType']}")
        Device: Miniscope_V4_BNO
        >>> print(f"Gain: {metadata['gain']}, LED: {metadata['led0']}")
        Gain: Medium, LED: 24

        Notes
        -----
        - The frameRate field is user-configured and may not reflect the actual
            acquisition rate. Use timeStamps.csv for ground truth timing.
        - Acquisition parameters (gain, ewl, led0) are captured at recording start
            and may have been adjusted during the session.

        See Also
        --------
        read_session_folder_metadata : Read session-level metadata
        """
        metadata_file_path = Path(metadata_file_path)

        if not metadata_file_path.exists():
            raise FileNotFoundError(f"Device metadata file not found: {metadata_file_path}")

        with open(metadata_file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def _read_session_folder_metadata(metadata_file_path: PathType) -> dict:
        """
        Read session-level metaData.json containing experiment and recording information.

        The Miniscope-DAQ-QT-Software creates two levels of metaData.json files:
        1. Session-level: Contains experiment/animal info and recording start time (this method)
        2. Device-level: Contains hardware settings for each Miniscope/camera

        This method reads the SESSION-level metadata.

        File Location
        -------------
        Session metadata is located in the recording session's root folder:
        ```
        session_folder/                      # Recording session
        ├── metaData.json                   # Session-level (this method)
        ├── notes.csv                       # User notes with timestamps
        ├── HPC_miniscope1/                 # Device folder
        │   ├── metaData.json              # Device-level (use read_device_folder_metadata)
        │   └── ...
        └── ACC_miniscope2/                 # Another device folder
            └── ...
        ```

        Session Metadata Contents
        -------------------------
        {
            "researcherName": "researcher_name",    # Researcher identifier
            "animalName": "animal_name",            # Subject identifier
            "experimentName": "experiment_name",    # Experiment identifier
            "baseDirectory": "D:/path/to/session",  # Full path to session
            "recordingStartTime": {                 # Timestamp when recording started
                "year": 2025,
                "month": 6,
                "day": 12,
                "hour": 15,
                "minute": 26,
                "second": 31,
                "msec": 176,
                "msecSinceEpoch": 1749756391176     # Unix timestamp in milliseconds
            },
            "miniscopes": [                         # List of Miniscope device names
                "HPC_miniscope1",
                "ACC_miniscope2"
            ],
            "cameras": [                            # List of behavior camera names
                "BehavCam_1"
            ],
            "framesPerFile": 1000                   # Default frames per video file
        }

        Parameters
        ----------
        metadata_file_path : PathType
            Path to the session metaData.json file
            (e.g., "path/to/2025_06_12/15_26_31/metaData.json")

        Returns
        -------
        dict
            Session metadata containing experiment information and recording details.
            Fields include:
            - researcherName: Researcher identifier
            - animalName: Subject/animal identifier
            - experimentName: Experiment identifier
            - baseDirectory: Original recording path
            - recordingStartTime: Recording start timestamp (dict with year, month, day, etc.)
            - miniscopes: List of Miniscope device names in this session
            - cameras: List of behavior camera names in this session
            - framesPerFile: Default frames per video file

        Raises
        ------
        FileNotFoundError
            If metaData.json file is not found

        Examples
        --------
        >>> metadata = MiniscopeImagingExtractor.read_session_folder_metadata(
        ...     "path/to/2025_06_12/15_26_31/metaData.json"
        ... )
        >>> print(f"Experiment: {metadata['experimentName']}")
        Experiment: experiment_name
        >>> print(f"Devices: {', '.join(metadata['miniscopes'])}")
        Devices: HPC_miniscope1, ACC_miniscope2
        >>> start_time = metadata['recordingStartTime']
        >>> print(f"Started: {start_time['year']}-{start_time['month']}-{start_time['day']}")
        Started: 2025-6-12

        Notes
        -----
        - The recordingStartTime is when the DAQ software started recording,
            not when individual frames were captured (use timeStamps.csv for that)
        - Device lists (miniscopes, cameras) reflect what was configured,
            not necessarily what has complete data

        See Also
        --------
        read_device_folder_metadata : Read device-level metadata
        """
        metadata_file_path = Path(metadata_file_path)

        if not metadata_file_path.exists():
            raise FileNotFoundError(f"Session metadata file not found: {metadata_file_path}")

        with open(metadata_file_path, "r") as f:
            return json.load(f)

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
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

    def has_time_vector(self) -> bool:
        """Detect if the ImagingExtractor has a time vector set or not.

        Notes
        -----
        Miniscope recordings should always have native timestamps from timeStamps.csv.
        This method overrides the parent implementation to ensure timestamps are properly
        loaded and returned, as Miniscope data is fundamentally time-based with
        hardware-generated timestamps that provide ground truth timing.

        Returns
        -------
        has_times: bool
            True if the ImagingExtractor has a time vector set, otherwise False.
        """
        if self._times is None:
            self._times = self.get_native_timestamps()
        return self._times is not None

    @staticmethod
    def _get_session_start_time(miniscope_folder_path) -> datetime | None:
        from .miniscope_utils import get_recording_start_time

        try:
            session_start_time = get_recording_start_time(file_path=Path(miniscope_folder_path) / "metaData.json")
            return session_start_time
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            warnings.warn(f"Could not retrieve session start time for folder {miniscope_folder_path}: \n {e}")
            return None


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
    ) -> tuple[list[PathType], PathType]:
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
        self, start_sample: int | None = None, end_sample: int | None = None
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

    def get_image_shape(self) -> tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._image_size[:-1]

    def get_image_size(self) -> tuple[int, int]:
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

    def get_channel_names(self) -> list[str]:
        return ["OpticalChannel"]

    def get_series(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
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
        self, start_frame: int | None = None, end_frame: int | None = None, channel: int | None = 0
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
