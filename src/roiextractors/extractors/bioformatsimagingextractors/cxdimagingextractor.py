"""ImagingExtractor for the CXD image format produced by Hamamatsu Photonics.

Classes
-------
CxdImagingExtractor
    A specialised ImagingExtractor for CXD files from Hamamatsu Photonics.
"""

import os
from pathlib import Path
from typing import List

import numpy as np

from ...extraction_tools import PathType
from .bioformatsimagingextractor import BioFormatsImagingExtractor


class CxdImagingExtractor(BioFormatsImagingExtractor):
    """Imaging extractor for reading Hamamatsu Photonics imaging data from .cxd files."""

    extractor_name = "CxdImaging"

    @classmethod
    def get_available_channels(cls, file_path) -> List[str]:
        """Get the available channel names from a CXD file produced by Hamamatsu Photonics.

        Parameters
        ----------
        file_path : PathType
            Path to the Bio-Formats file.

        Returns
        -------
        channel_names: list
            List of channel names.
        """
        from .bioformats_utils import extract_ome_metadata, parse_ome_metadata

        ome_metadata = extract_ome_metadata(file_path=file_path)
        parsed_metadata = parse_ome_metadata(metadata=ome_metadata)
        channel_names = parsed_metadata["channel_names"]
        return channel_names

    @classmethod
    def get_available_planes(cls, file_path):
        """Get the available plane names from a CXD file produced by Hamamatsu Photonics.

        Parameters
        ----------
        file_path : PathType
            Path to the Bio-Formats file.

        Returns
        -------
        plane_names: list
            List of plane names.
        """
        from .bioformats_utils import extract_ome_metadata, parse_ome_metadata

        ome_metadata = extract_ome_metadata(file_path=file_path)
        parsed_metadata = parse_ome_metadata(metadata=ome_metadata)
        num_planes = parsed_metadata["num_planes"]
        plane_names = [f"{i}" for i in range(num_planes)]
        return plane_names

    def __init__(
        self,
        file_path: PathType,
        channel_name: str = None,
        plane_name: str = None,
        sampling_frequency: float = None,
    ):
        r"""
        Create a CxdImagingExtractor instance from a CXD file produced by Hamamatsu Photonics.

        This extractor requires `bioformats_jar` to be installed in the environment,
        and requires the java executable to be available on the path (or via the JAVA_HOME environment variable),
        along with the mvn executable.

        If you are using conda, you can install with `conda install -c conda-forge bioformats_jar`.
        Note: you may need to reactivate your conda environment after installing.
        If you are still getting a JVMNotFoundException, try:
        # mac and linux:
        `export JAVA_HOME=$CONDA_PREFIX`

        # windows:
        `set JAVA_HOME=%CONDA_PREFIX%\\Library`

        Parameters
        ----------
        file_path : PathType
            Path to the CXD file.
        channel_name : str
            The name of the channel for this extractor. (default=None)
        plane_name : str
            The name of the plane for this extractor. (default=None)
        sampling_frequency : float
            The sampling frequency of the imaging data. (default=None)
            Has to be provided manually if not found in the metadata.
        """
        from .bioformats_utils import extract_ome_metadata, parse_ome_metadata

        if "JAVA_HOME" not in os.environ:
            conda_home = os.environ.get("CONDA_PREFIX")
            os.environ["JAVA_HOME"] = conda_home

        if ".cxd" not in Path(file_path).suffixes:
            raise ValueError("The file suffix must be .cxd!")

        dimension_order = "TCZYX"

        self.ome_metadata = extract_ome_metadata(file_path=file_path)
        parsed_metadata = parse_ome_metadata(metadata=self.ome_metadata)

        channel_names = parsed_metadata["channel_names"]
        if channel_name is None:
            if parsed_metadata["num_channels"] > 1:
                raise ValueError(
                    "More than one channel is detected! Please specify which channel you wish to load "
                    "with the `channel_name` argument. To see which channels are available, use "
                    "`CxdImagingExtractor.get_available_channels(file_path=...)`"
                )
            channel_name = channel_names[0]

        plane_names = [f"{i}" for i in range(parsed_metadata["num_planes"])]
        if plane_name is None:
            if parsed_metadata["num_planes"] > 1:
                raise ValueError(
                    "More than one plane is detected! Please specify which plane you wish to load "
                    "with the `plane_name` argument. To see which planes are available, use "
                    "`CxdImagingExtractor.get_available_planes(file_path=...)`"
                )
            plane_name = plane_names[0]

        sampling_frequency = sampling_frequency or parsed_metadata["sampling_frequency"]
        if sampling_frequency is None:
            raise ValueError(
                "Sampling frequency is not found in the metadata. Please provide it manually with the 'sampling_frequency' argument."
            )

        super().__init__(
            file_path=file_path,
            channel_name=channel_name,
            plane_name=plane_name,
            dimension_order=dimension_order,
            parsed_metadata=parsed_metadata,
        )

        pixels_metadata = self.ome_metadata.images[0].pixels
        timestamps = [plane.delta_t for plane in pixels_metadata.planes]
        if np.any(timestamps):
            self._times = np.array(timestamps)
