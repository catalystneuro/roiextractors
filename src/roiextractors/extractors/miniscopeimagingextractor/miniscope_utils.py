"""Utility functions for Miniscope data extraction.

This module provides utility functions to automatically discover Miniscope files
from folder structures, supporting various Miniscope folder organizations.
"""

import json
from pathlib import Path
from typing import List, Tuple

from ...extraction_tools import PathType, get_package


def get_miniscope_files_from_folder(folder_path: PathType) -> Tuple[List[PathType], PathType]:
    """
    Automatically retrieve .avi file paths and configuration file path from a folder.

    This function scans the provided folder for Miniscope .avi files and configuration
    files, supporting the common Miniscope folder structures.

    Parameters
    ----------
    folder_path : PathType
        The folder path containing Miniscope data. This can be either:
        - A parent folder containing multiple timestamp subfolders with Miniscope data
        - A direct path to a folder containing .avi files and metaData.json

    Returns
    -------
    Tuple[List[PathType], PathType]
        A tuple containing:
        - List of .avi file paths sorted naturally
        - Path to the configuration file (metaData.json)

    Raises
    ------
    AssertionError
        If no .avi files or configuration files are found in the specified folder.

    Examples
    --------
    >>> file_paths, config_path = get_miniscope_files_from_folder("/path/to/miniscope/data")
    >>> extractor = MiniscopeMultiRecordingImagingExtractor(file_paths, config_path)
    """
    natsort = get_package(package_name="natsort", installation_instructions="pip install natsort")

    folder_path = Path(folder_path)
    configuration_file_name = "metaData.json"

    # Try to find .avi files in the current folder structure
    # Pattern 1: Parent folder with timestamp subfolders containing Miniscope folders
    miniscope_avi_file_paths = natsort.natsorted(list(folder_path.glob("*/Miniscope/*.avi")))
    miniscope_config_files = natsort.natsorted(list(folder_path.glob(f"*/Miniscope/{configuration_file_name}")))

    # Pattern 2: Direct folder containing .avi files
    if not miniscope_avi_file_paths:
        miniscope_avi_file_paths = natsort.natsorted(list(folder_path.glob("*.avi")))
        miniscope_config_files = natsort.natsorted(list(folder_path.glob(configuration_file_name)))

    # Pattern 3: Folder with Miniscope subfolder
    if not miniscope_avi_file_paths:
        miniscope_avi_file_paths = natsort.natsorted(list(folder_path.glob("Miniscope/*.avi")))
        miniscope_config_files = natsort.natsorted(list(folder_path.glob(f"Miniscope/{configuration_file_name}")))

    # Validate that files were found
    assert miniscope_avi_file_paths, f"The Miniscope movies (.avi files) are missing from '{folder_path}'."
    assert (
        miniscope_config_files
    ), f"The configuration files ({configuration_file_name} files) are missing from '{folder_path}'."

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
