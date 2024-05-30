"""ImagingExtractor for reading files supported by Bio-Formats.

Classes
-------
BioFormatsImagingExtractor
    The base ImagingExtractor for Bio-Formats.
"""

from typing import Tuple

import numpy as np

from ...imagingextractor import ImagingExtractor
from ...extraction_tools import PathType, DtypeType


class BioFormatsImagingExtractor(ImagingExtractor):
    """Imaging extractor for files supported by Bio-Formats."""

    extractor_name = "BioFormatsImaging"

    def __init__(
        self,
        file_path: PathType,
        channel_name: str,
        plane_name: str,
        dimension_order: str,
        parsed_metadata: dict,
    ):
        r"""
        Create a BioFormatsImagingExtractor instance from a file supported by Bio-Formats.

        Supported file formats: https://bio-formats.readthedocs.io/en/stable/supported-formats.html

        This extractor requires bioformats_jar to be installed in the environment,
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
            Path to the file.
        channel_name : str
            The name of the channel for this extractor.
        plane_name : str
            The name of the plane for this extractor.
        dimension_order : str
            The order of dimension for reading the frames. For .cxd format it is "TCZYX".
            See aicsimageio.dimensions.DimensionNames and aicsimageio.dimensions.Dimensions for more information.
        parsed_metadata: dict
            Parsed metadata dictionary in the form outputted by parse_ome_metadata in order to be parsed
            correctly.
        """
        from roiextractors.extractors.bioformatsimagingextractor.bioformats_utils import check_file_format_is_supported
        import aicsimageio

        self.file_path = file_path
        super().__init__()

        check_file_format_is_supported(self.file_path)

        self.dimension_order = dimension_order

        self._num_frames = parsed_metadata["num_frames"]
        self._num_channels = parsed_metadata["num_channels"]
        self._num_planes = parsed_metadata["num_planes"]
        self._num_rows = parsed_metadata["num_rows"]
        self._num_columns = parsed_metadata["num_columns"]
        self._dtype = parsed_metadata["dtype"]
        self._sampling_frequency = parsed_metadata["sampling_frequency"]
        self._channel_names = parsed_metadata["channel_names"]
        self._plane_names = [f"{i}" for i in range(self._num_planes)]

        if channel_name not in self._channel_names:
            raise ValueError(f"Channel name ({channel_name}) not found in channel names ({self._channel_names}).")
        self.channel_index = self._channel_names.index(channel_name)

        if plane_name not in self._plane_names:
            raise ValueError(f"Plane name ({plane_name}) not found in plane names ({self._plane_names}).")
        self.plane_index = self._plane_names.index(plane_name)

        with aicsimageio.readers.bioformats_reader.BioFile(self.file_path) as reader:
            self._video = reader.to_dask()

    def get_channel_names(self) -> list:
        return self._channel_names

    def get_dtype(self) -> DtypeType:
        return self._dtype

    def get_image_size(self) -> Tuple[int, int]:
        return self._num_rows, self._num_columns

    def get_num_channels(self) -> int:
        return self._num_channels

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_video(self, start_frame=None, end_frame=None, channel: int = 0) -> np.ndarray:
        dimension_dict = {
            "T": slice(start_frame, end_frame),
            "C": self.channel_index,
            "Z": self.plane_index,
            "Y": slice(None),
            "X": slice(None),
        }
        slices = [dimension_dict[dimension] for dimension in self.dimension_order]
        video = self._video[tuple(slices)]

        # re-arrange axis to ensure video axes are time x height x width
        axis_order = tuple("TYX".index(dim) for dim in self.dimension_order if dim in "TYX")
        video = video.transpose(axis_order)

        return video.compute()
