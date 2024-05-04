"""Utility functions for ScanImage TIFF Extractors."""

import numpy as np
import json
from ...extraction_tools import PathType, get_package


def _get_scanimage_reader() -> type:
    """Import the scanimage-tiff-reader package and return the ScanImageTiffReader class."""
    return get_package(
        package_name="ScanImageTiffReader", installation_instructions="pip install scanimage-tiff-reader"
    ).ScanImageTiffReader


def extract_extra_metadata(
    file_path: PathType,
) -> dict:  # TODO: Refactor neuroconv to reference this implementation to avoid duplication
    """Extract metadata from a ScanImage TIFF file.

    Parameters
    ----------
    file_path : PathType
        Path to the TIFF file.

    Returns
    -------
    extra_metadata: dict
        Dictionary of metadata extracted from the TIFF file.

    Notes
    -----
    Known to work on SI versions v3.8.0, v2019bR0, v2022.0.0, and v2023.0.0
    """
    ScanImageTiffReader = _get_scanimage_reader()
    io = ScanImageTiffReader(str(file_path))
    extra_metadata = {}
    for metadata_string in (io.description(iframe=0), io.metadata()):
        system_metadata_dict = {
            x.split("=")[0].strip(): x.split("=")[1].strip()
            for x in metadata_string.replace("\n", "\r").split("\r")
            if "=" in x
        }
        extra_metadata = dict(**extra_metadata, **system_metadata_dict)
    if "\n\n" in io.metadata():
        additional_metadata_string = io.metadata().split("\n\n")[1]
        additional_metadata = json.loads(additional_metadata_string)
        extra_metadata = dict(**extra_metadata, **additional_metadata)
    return extra_metadata


def parse_matlab_vector(matlab_vector: str) -> list:
    """Parse a MATLAB vector string into a list of integer values.

    Parameters
    ----------
    matlab_vector : str
        MATLAB vector string.

    Returns
    -------
    vector: list of int
        List of integer values.

    Raises
    ------
    ValueError
        If the MATLAB vector string cannot be parsed.

    Notes
    -----
    MATLAB vector string is of the form "[1 2 3 ... N]" or "[1,2,3,...,N]" or "[1;2;3;...;N]".
    There may or may not be whitespace between the values. Ex. "[1, 2, 3]" or "[1,2,3]".
    """
    vector = matlab_vector.strip("[]")
    if ";" in vector:
        vector = vector.split(";")
    elif "," in vector:
        vector = vector.split(",")
    elif " " in vector:
        vector = vector.split(" ")
    elif len(vector) == 1:
        pass
    else:
        raise ValueError(f"Could not parse vector from {matlab_vector}.")
    vector = [int(x.strip()) for x in vector if x != ""]
    return vector


def parse_metadata(metadata: dict) -> dict:
    """Parse metadata dictionary to extract relevant information and store it standard keys for ImagingExtractors.

    Currently supports
    - sampling_frequency
    - num_planes
    - frames_per_slice
    - channel_names
    - num_channels

    Parameters
    ----------
    metadata : dict
        Dictionary of metadata extracted from the TIFF file.

    Returns
    -------
    metadata_parsed: dict
        Dictionary of parsed metadata.

    Notes
    -----
    Known to work on SI versions v2019bR0, v2022.0.0, and v2023.0.0. Fails on v3.8.0.
    SI.hChannels.channelsActive = string of MATLAB-style vector with channel integers (see parse_matlab_vector).
    SI.hChannels.channelName = "{'channel_name_1' 'channel_name_2' ... 'channel_name_M'}"
        where M is the number of channels (active or not).
    """
    sampling_frequency = float(metadata["SI.hRoiManager.scanFrameRate"])
    num_planes = int(metadata["SI.hStackManager.numSlices"])
    frames_per_slice = int(metadata["SI.hStackManager.framesPerSlice"])
    active_channels = parse_matlab_vector(metadata["SI.hChannels.channelsActive"])
    channel_indices = np.array(active_channels) - 1  # Account for MATLAB indexing
    channel_names = np.array(metadata["SI.hChannels.channelName"].split("'")[1::2])
    channel_names = channel_names[channel_indices].tolist()
    num_channels = len(channel_names)
    if "RoiGroups" in metadata.keys():
        roi_metadata = metadata["RoiGroups"]
    else:
        roi_metadata = None
    metadata_parsed = dict(
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        num_planes=num_planes,
        frames_per_slice=frames_per_slice,
        channel_names=channel_names,
        roi_metadata=roi_metadata,
    )
    return metadata_parsed


def parse_metadata_v3_8(metadata: dict) -> dict:
    """Parse metadata dictionary to extract relevant information and store it standard keys for ImagingExtractors.

    Requires old version of metadata (v3.8).
    Currently supports
    - sampling frequency
    - num_channels
    - num_planes

    Parameters
    ----------
    metadata : dict
        Dictionary of metadata extracted from the TIFF file.

    Returns
    -------
    metadata_parsed: dict
        Dictionary of parsed metadata.
    """
    sampling_frequency = float(metadata["state.acq.frameRate"])
    num_channels = int(metadata["state.acq.numberOfChannelsSave"])
    num_planes = int(metadata["state.acq.numberOfZSlices"])
    metadata_parsed = dict(
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        num_planes=num_planes,
    )
    return metadata_parsed


def extract_timestamps_from_file(file_path: PathType) -> np.ndarray:
    """Extract the frame timestamps from a ScanImage TIFF file.

    Parameters
    ----------
    file_path : PathType
        Path to the TIFF file.

    Returns
    -------
    timestamps : numpy.ndarray
        Array of frame timestamps in seconds.

    Raises
    ------
    AssertionError
        If the frame timestamps are not found in the TIFF file.

    Notes
    -----
    Known to work on SI versions v2019bR0, v2022.0.0, and v2023.0.0. Fails on v3.8.0.
    """
    ScanImageTiffReader = _get_scanimage_reader()
    io = ScanImageTiffReader(str(file_path))
    assert "frameTimestamps_sec" in io.description(iframe=0), "frameTimestamps_sec not found in TIFF file"
    num_frames = io.shape()[0]
    timestamps = np.zeros(num_frames)
    for iframe in range(num_frames):
        description = io.description(iframe=iframe)
        description_lines = description.split("\n")
        for line in description_lines:
            if "frameTimestamps_sec" in line:
                timestamps[iframe] = float(line.split("=")[1].strip())
                break

    return timestamps
