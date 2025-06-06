"""Utility functions for Miniscope data extraction.

This module provides utility functions to automatically discover Miniscope files
from folder structures, supporting various Miniscope folder organizations.
"""

import json
from pathlib import Path
from typing import List, Tuple

from ...extraction_tools import PathType, get_package


def get_miniscope_files_from_multi_timestamp_subfolders(
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
