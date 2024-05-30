from pathlib import Path


import numpy as np
import aicsimageio
from aicsimageio.formats import FORMAT_IMPLEMENTATIONS
from ome_types import OME

from roiextractors.extraction_tools import PathType


def check_file_format_is_supported(file_path: PathType):
    """
    Check if the file format is supported by BioformatsReader from aicsimageio.

    Returns ValueError if the file format is not supported.

    Parameters
    ----------
    file_path : PathType
        Path to the file.
    """
    bioformats_reader = "aicsimageio.readers.bioformats_reader.BioformatsReader"
    supported_file_suffixes = [
        suffix_name for suffix_name, reader in FORMAT_IMPLEMENTATIONS.items() if bioformats_reader in reader
    ]

    file_suffix = Path(file_path).suffix.replace(".", "")
    if file_suffix not in supported_file_suffixes:
        raise ValueError(f"File '{file_path}' is not supported by BioformatsReader.")


def extract_ome_metadata(
    file_path: PathType,
) -> OME:
    """
    Extract OME metadata from a file using aicsimageio.

    Parameters
    ----------
    file_path : PathType
        Path to the file.
    """
    check_file_format_is_supported(file_path)

    with aicsimageio.readers.bioformats_reader.BioFile(file_path) as reader:
        ome_metadata = reader.ome_metadata

    return ome_metadata


def parse_ome_metadata(metadata: OME) -> dict:
    """
    Parse metadata in OME format to extract relevant information and store it standard keys for ImagingExtractors.

    Currently supports:
    - num_frames
    - sampling_frequency
    - num_channels
    - num_planes
    - num_rows (height of the image)
    - num_columns (width of the image)
    - dtype
    - channel_names

    """
    images_metadata = metadata.images[0]
    pixels_metadata = images_metadata.pixels

    sampling_frequency = None
    if pixels_metadata.time_increment is not None:
        sampling_frequency = 1 / pixels_metadata.time_increment

    channel_names = [channel.id for channel in pixels_metadata.channels]

    metadata_parsed = dict(
        num_frames=images_metadata.pixels.size_t,
        sampling_frequency=sampling_frequency,
        num_channels=images_metadata.pixels.size_c,
        num_planes=images_metadata.pixels.size_z,
        num_rows=images_metadata.pixels.size_y,
        num_columns=images_metadata.pixels.size_x,
        dtype=np.dtype(pixels_metadata.type.numpy_dtype),
        channel_names=channel_names,
    )

    return metadata_parsed
