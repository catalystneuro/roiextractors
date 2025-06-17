"""Utility functions for Miniscope data extraction.

This module provides utility functions to automatically discover Miniscope files
from folder structures, supporting various Miniscope folder organizations.
"""

import json
from pathlib import Path
from typing import List, Tuple
import datetime

from ...extraction_tools import PathType, get_package


def get_miniscope_files_from_multi_recordings_subfolders(
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


def get_miniscope_files_from_direct_folder(folder_path: PathType) -> Tuple[List[PathType], PathType]:
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
    Tuple[List[PathType], PathType]
        A tuple containing:
        - List of .avi file paths sorted naturally
        - Path to the configuration file (metaData.json)

    Raises
    ------
    AssertionError
        If no .avi files or configuration files are found.
    """
    natsort = get_package(package_name="natsort", installation_instructions="pip install natsort")

    folder_path = Path(folder_path)
    configuration_file_name = "metaData.json"

    miniscope_avi_file_paths = natsort.natsorted(list(folder_path.glob("*.avi")))
    miniscope_config_files = natsort.natsorted(list(folder_path.glob(configuration_file_name)))

    assert miniscope_avi_file_paths, f"No .avi files found in direct folder structure at '{folder_path}'"
    assert miniscope_config_files, f"No configuration file found in direct folder structure at '{folder_path}'"

    return miniscope_avi_file_paths, miniscope_config_files[0]


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
        raise json.JSONDecodeError(f"Invalid JSON in configuration file {configuration_file_path}: {e}", e.doc, e.pos)


def get_recording_start_time(file_path: PathType) -> datetime.datetime:
    """
    Retrieve the recording start time from metadata in the specified folder.

    Parameters:
    -----------
    file_path : str, Path
        Path to the "metaData.json" file with recording start time details.

    Returns:
    --------
    datetime.datetime
        A datetime object representing the session start time, based on the metadata's year, month, day, hour, minute,
        second, and millisecond fields.

    Raises:
    -------
    AssertionError
        If the "metaData.json" file is not found in the specified folder path.
    KeyError
        If any of the required time fields ("year", "month", "day", "hour", "minute", "second", "msec") are missing
        from the metadata.

    Notes:
    ------
    - The function expects a "recordingStartTime" key in the metadata JSON, which contains start time details.
      If not present, the top-level JSON object is assumed to contain the time information.
    - The "msec" field in the metadata is converted from milliseconds to microseconds for compatibility with the datetime
      microsecond field.
    """

    ## Read metadata
    with open(file_path) as f:
        general_metadata = json.load(f)

    if "recordingStartTime" in general_metadata:  # Miniscope Version 4
        start_time_info = general_metadata["recordingStartTime"]
    else:  # Miniscope Version 3
        start_time_info = general_metadata

    required_keys = ["year", "month", "day", "hour", "minute", "second", "msec"]
    for key in required_keys:
        if key not in start_time_info:
            raise KeyError(f"Missing required key '{key}' in the metadata")

    session_start_time = datetime.datetime(
        year=start_time_info["year"],
        month=start_time_info["month"],
        day=start_time_info["day"],
        hour=start_time_info["hour"],
        minute=start_time_info["minute"],
        second=start_time_info["second"],
        microsecond=start_time_info["msec"] * 1000,  # Convert milliseconds to microseconds
    )

    return session_start_time


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


def get_recording_start_times_for_multi_recordings(
    folder_path: PathType, miniscopeDeviceName: str = "Miniscope"
) -> List[datetime.datetime]:
    """
    Retrieve recording start times for multiple recordings in a multi-session folder structure.

    Parameters
    ----------
    folder_path : PathType
        Path to the parent folder containing timestamp subfolders.
    miniscopeDeviceName : str, optional
        Name of the Miniscope device subfolder. Defaults to "Miniscope".

    Returns
    -------
    List[datetime.datetime]
        A list of datetime objects representing the start times of each recording.

    Raises
    ------
    AssertionError
        If no configuration files are found.
    """
    natsort = get_package(package_name="natsort", installation_instructions="pip install natsort")

    folder_path = Path(folder_path)
    configuration_file_name = "metaData.json"

    miniscope_config_files = natsort.natsorted(list(folder_path.glob(f"*/{configuration_file_name}")))

    assert miniscope_config_files, f"No Miniscope configuration files found at '{folder_path}'"

    start_times = []
    for config_file in miniscope_config_files:
        config = load_miniscope_config(config_file)
        has_timestamp_info = any(key in config for key in ["year", "month", "day", "recordingStartTime"])
        if has_timestamp_info:
            start_time = get_recording_start_time(config_file)
            start_times.append(start_time)

    return start_times


def get_timestamps_for_multi_recordings(
    folder_path: PathType, miniscopeDeviceName: str = "Miniscope"
) -> List[List[float]]:
    """
    Retrieve timestamps for multiple recordings in a multi-recordings folder structure.

    Parameters
    ----------
    folder_path : PathType
        Path to the parent folder containing multi-recordings subfolders.
    miniscopeDeviceName : str, optional
        Name of the Miniscope device subfolder. Defaults to "Miniscope".

    Returns
    -------
    List[List[float]]
        A list of lists, where each inner list contains the times for a recording.

    Raises
    ------
    AssertionError
        If no time files are found.
    """
    natsort = get_package(package_name="natsort", installation_instructions="pip install natsort")

    folder_path = Path(folder_path)
    time_file_name = "timeStamps.csv"

    timestamps_file_paths = natsort.natsorted(list(folder_path.glob(f"*/{miniscopeDeviceName}/{time_file_name}")))

    assert timestamps_file_paths, f"No time files found at '{folder_path}'"

    recording_start_times = get_recording_start_times_for_multi_recordings(folder_path=folder_path)
    timestamps = []
    for file_ind, file_path in enumerate(timestamps_file_paths):
        timestamps_per_file = read_timestamps_from_csv_file(file_path=file_path)
        if recording_start_times:
            offset = (recording_start_times[file_ind] - recording_start_times[0]).total_seconds()
            timestamps_per_file += offset

        timestamps.extend(timestamps_per_file)

    return timestamps
