"""Utility functions for Miniscope data extraction.

This module provides utility functions to automatically discover Miniscope files
from folder structures, supporting various Miniscope folder organizations.
"""

import datetime
import json
from pathlib import Path

from .miniscopeimagingextractor import MiniscopeImagingExtractor
from ...extraction_tools import PathType, get_package


def get_recording_start_time(file_path: PathType) -> datetime.datetime:
    """
    Retrieve the recording start time from metadata in the specified folder.

    Parameters
    ----------
    file_path : str, Path
        Path to the "metaData.json" file with recording start time details.
        This metadata file is typically located in the recording folder that also contains the folder with the video files.
        parent_folder/
        ├── recording_folder/
        │   ├── Miniscope/
        │   │   ├── 0.avi
        │   │   ├── 1.avi
        │   │   └── metaData.json
        │   ├── BehavCam_2/
        │   └── metaData.json <-- This is the file_path

    Examples
    --------
    >>> from roiextractors.extractors.miniscopeimagingextractor.miniscope_utils import get_recording_start_time
    >>> start_time = get_recording_start_time('path/to/recording_folder/metaData.json')

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

    Examples of metaData.json content:
    ----------------------------------
    Example 1 (Miniscope V4 format with recordingStartTime):
    {
        "animalName": "animal_name",
        "baseDirectory": "C:/data/researcher_name/experiment_name/animal_name/2022_09_19/09_18_41",
        "cameras": [
        ],
        "experimentName": "experiment_name",
        "miniscopes": [
            "miniscope"
        ],
        "recordingStartTime": {
            "day": 19,
            "hour": 9,
            "minute": 18,
            "month": 9,
            "msec": 7,
            "msecSinceEpoch": 1663597121007,
            "second": 41,
            "year": 2022
        },
        "researcherName": "researcher_name"
    }

    Example 2 (Miniscope V3 format with flat time fields):
    {
        "animalName": "animal_name",
        "baseDirectory": "C:/data/researcher_name/experiment_name/animal_name/2021_10_14/10_11_24",
        "cameras": [
        ],
        "day": 14,
        "experimentName": "experiment_name",
        "hour": 10,
        "miniscopes": [
            "Miniscope"
        ],
        "minute": 11,
        "month": 10,
        "msec": 779,
        "msecSinceEpoch": 1634220684779,
        "researcherName": "researcher_name",
        "second": 24,
        "year": 2021
    }

    """
    ## Read metadata
    with open(file_path) as f:
        general_metadata = json.load(f)

    if "recordingStartTime" in general_metadata:  # Miniscope Version 4
        start_time_info = general_metadata["recordingStartTime"]
    else:  # Miniscope Version 3
        start_time_info = general_metadata

    required_keys = ["year", "month", "day", "hour", "minute", "second", "msec"]
    available_keys = list(start_time_info.keys())
    missing_keys = [key for key in required_keys if key not in start_time_info]
    if missing_keys:
        raise KeyError(
            f"Missing required keys {missing_keys} in the metadata. "
            f"Available keys: {available_keys}. "
            f"Expected keys: {required_keys}."
        )

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
) -> list[datetime.datetime]:
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
