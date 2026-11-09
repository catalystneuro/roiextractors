"""Utility functions for ScanImage TIFF Extractors."""

import json
import mmap
import struct
from typing import Sequence

import numpy as np

from ...extraction_tools import PathType, get_package


def _get_scanimage_reader() -> type:
    """Import the scanimage-tiff-reader package and return the ScanImageTiffReader class."""
    return get_package(
        package_name="ScanImageTiffReader", installation_instructions="pip install scanimage-tiff-reader"
    ).ScanImageTiffReader


def _read_image_descriptions(file_path: PathType, page_indices: Sequence[int]) -> list[str | None]:
    """Read the `ImageDescription` tag of chosen pages of a ScanImage TIFF file.

    ScanImage writes the header of each frame, with its timestamp, into the `ImageDescription` tag of its page.
    Building a page with `tifffile` parses all of its tags, while this reads only the tag directory of each page and
    its description. ScanImage files are little-endian BigTIFF, which `read_scanimage_metadata` requires, so only that
    layout is read.

    Parameters
    ----------
    file_path : PathType
        Path to the TIFF file.
    page_indices : Sequence[int]
        Indices of the pages in the file, in any order.

    Returns
    -------
    list[str | None]
        The description of each page in the order of `page_indices`, or None for a page without one.
    """
    tag_size = 20
    # Mapping the file reads only the bytes that are touched, without a system call for each read
    with open(file_path, "rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as tiff_bytes:
        if tiff_bytes[:4] != b"II\x2b\x00":
            raise ValueError(f"{file_path} is not a little-endian BigTIFF file.")

        # Follow the chain of tag directories to the offset of each page, up to the last page needed. A directory is
        # the number of tags (8 bytes), the tags (20 bytes each) and the offset of the next directory (8 bytes)
        page_offsets = [struct.unpack_from("<Q", tiff_bytes, 8)[0]]
        last_page_index = max(page_indices)
        while len(page_offsets) <= last_page_index:
            page_offset = page_offsets[-1]
            num_tags = struct.unpack_from("<Q", tiff_bytes, page_offset)[0]
            next_page_offset = struct.unpack_from("<Q", tiff_bytes, page_offset + 8 + num_tags * tag_size)[0]
            page_offsets.append(next_page_offset)

        descriptions = []
        for page_index in page_indices:
            page_offset = page_offsets[page_index]
            num_tags = struct.unpack_from("<Q", tiff_bytes, page_offset)[0]
            description = None
            for tag_index in range(num_tags):
                # A tag is its code, its type, the number of values and the values or their offset (8 bytes)
                tag_start = page_offset + 8 + tag_index * tag_size
                code, _, count, value_offset = struct.unpack_from("<HHQQ", tiff_bytes, tag_start)
                if code != 270:
                    continue
                # Values of up to 8 bytes are stored in the tag itself instead of at an offset
                if count <= 8:
                    value = tiff_bytes[tag_start + 12 : tag_start + 12 + count]
                else:
                    value = tiff_bytes[value_offset : value_offset + count]
                description = value.rstrip(b"\x00").decode("utf-8", errors="replace")
                break
            descriptions.append(description)
    return descriptions


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


def read_scanimage_metadata(file_path: PathType) -> dict:
    """
    Read and parse metadata from a ScanImage TIFF file.

    This function extracts both the non-varying frame metadata and ROI group metadata
    (if available) from a ScanImage TIFF file and processes them to extract key imaging parameters.

    The function returns a python dictionary with fields already parsed to python objects (in opposition to a string)

    Parameters
    ----------
    file_path : PathType
        Path to the ScanImage TIFF file.

    Returns
    -------
    metadata_dict : dict
        Dictionary containing three nested dictionaries:
        - scan_image_non_varying_frame_metadata: Raw non-varying frame metadata
        - scan_image_roi_group_metadata: Raw ROI group metadata (if present)
        - roiextractors_parsed_metadata: Parsed metadata with standardized keys:
            - sampling_frequency: Frame or volume scan rate in Hz
            - num_channels: Number of available imaging channels
            - num_planes: Number of imaging planes (slices)
            - frames_per_slice: Number of frames per Z slice
            - channel_names: List of available channel names
            - roi_metadata: ROI definitions (if present)

    Notes
    -----
    The ScanImage TIFF format includes:
    1. TIFF Header Section: Defines byte order and offsets
    2. ScanImage Static Metadata Section: Contains metadata applicable to all frames
        - Non-Varying Frame Data: System configuration for all frames
        - ROI Group Data: Defined regions of interest (if available)
    3. Frame Sections: One per image frame, containing:
        - IFD Header: Image File Directory with tags and values
        - Frame-specific data: Timestamps and other frame metadata
        - Image data: The actual image pixels

    The function uses the tifffile module's read_scanimage_metadata function to extract
    the raw metadata, then processes it to standardize key imaging parameters.
    """
    from tifffile import read_scanimage_metadata

    with open(file_path, "rb") as fh:
        all_metadata = read_scanimage_metadata(fh)
        non_varying_frame_metadata = all_metadata[0]
        roi_group_metadata = all_metadata[1]

    if non_varying_frame_metadata["SI.hStackManager.enable"]:
        num_slices = non_varying_frame_metadata["SI.hStackManager.numSlices"]

        flyback_frames = (
            non_varying_frame_metadata["SI.hStackManager.numFramesPerVolumeWithFlyback"]
            - non_varying_frame_metadata["SI.hStackManager.numFramesPerVolume"]
        )
        # num_planes = num_slices + flyback_frames   # TODO: discuss on issue, leave the current behavior now
        num_planes = num_slices
        frames_per_slice = non_varying_frame_metadata["SI.hStackManager.framesPerSlice"]
    else:
        num_planes = 1
        frames_per_slice = 1

    if num_planes == 1:
        sampling_frequency = non_varying_frame_metadata["SI.hRoiManager.scanFrameRate"]
    else:
        sampling_frequency = non_varying_frame_metadata["SI.hRoiManager.scanVolumeRate"]

    # `channelSave` indicates whether the channel is saved. Note that a channel might not be saved even if it is active.
    # We check `channelSave` first but keep the `channelsActive` check for backward compatibility.
    channel_availability_keys = ["SI.hChannels.channelSave", "SI.hChannels.channelsActive"]
    channel_availability = None
    for key in channel_availability_keys:
        if key in non_varying_frame_metadata.keys():
            channel_availability = key
            break
    if channel_availability is None:
        raise ValueError(f"Could not find any of {channel_availability_keys} in metadata.")

    available_channels = non_varying_frame_metadata[channel_availability]
    available_channels = [available_channels] if not isinstance(available_channels, list) else available_channels
    channel_indices = np.array(available_channels) - 1  # Account for MATLAB indexing
    channel_names = non_varying_frame_metadata["SI.hChannels.channelName"]
    channel_names_available = [channel_names[i] for i in channel_indices]
    num_channels = len(channel_names_available)
    if roi_group_metadata:
        roi_metadata = roi_group_metadata["RoiGroups"]
    else:
        roi_metadata = None

    metadata_parsed = dict(
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        num_planes=num_planes,
        frames_per_slice=frames_per_slice,
        channel_names=channel_names_available,
        roi_metadata=roi_metadata,
    )

    metadata_dict = dict(
        scan_image_non_varying_frame_metadata=non_varying_frame_metadata,
        scan_image_roi_group_metadata=roi_group_metadata,
        roiextractors_parsed_metadata=metadata_parsed,
    )

    return metadata_dict


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
    if metadata["SI.hStackManager.enable"] == "true":
        num_planes = int(metadata["SI.hStackManager.numSlices"])
        frames_per_slice = int(metadata["SI.hStackManager.framesPerSlice"])
    else:
        num_planes = 1
        frames_per_slice = 1

    # `channelSave` indicates whether the channel is saved. Note that a channel might not be saved even if it is active.
    # We check `channelSave` first but keep the `channelsActive` check for backward compatibility.
    channel_availability_keys = ["SI.hChannels.channelSave", "SI.hChannels.channelsActive"]
    channel_availability = None
    for key in channel_availability_keys:
        if key in metadata.keys():
            channel_availability = key
            break
    if channel_availability is None:
        raise ValueError(f"Could not find any of {channel_availability_keys} in metadata.")

    available_channels = parse_matlab_vector(metadata[channel_availability])
    channel_indices = np.array(available_channels) - 1  # Account for MATLAB indexing
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
