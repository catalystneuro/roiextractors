"""Utility functions for Miniscope data extraction.

This module provides utility functions to automatically discover Miniscope files
from folder structures, supporting various Miniscope folder organizations.
"""

import datetime
import json
from pathlib import Path
from typing import List, Tuple

from .miniscopeimagingextractor import MiniscopeImagingExtractor
from ...extraction_tools import PathType, get_package


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


def get_recording_start_time(file_path: PathType) -> datetime.datetime:
    """
    Retrieve the recording start time from metadata in the specified folder.

    Parameters
    ----------
    file_path : str, Path
        Path to the "metaData.json" file with recording start time details.

    Returns
    -------
    datetime.datetime
        A datetime object representing the session start time, based on the metadata's year, month, day, hour, minute,
        second, and millisecond fields.

    Raises
    ------
    AssertionError
        If the "metaData.json" file is not found in the specified folder path.
    KeyError
        If any of the required time fields ("year", "month", "day", "hour", "minute", "second", "msec") are missing
        from the metadata.

    Notes
    -----
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
        config = MiniscopeImagingExtractor.load_miniscope_config(config_file)
        has_timestamp_info = any(key in config for key in ["year", "month", "day", "recordingStartTime"])
        if has_timestamp_info:
            start_time = get_recording_start_time(config_file)
            start_times.append(start_time)

    return start_times
