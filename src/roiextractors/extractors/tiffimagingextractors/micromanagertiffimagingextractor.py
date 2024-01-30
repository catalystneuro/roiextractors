"""A ImagingExtractor for TIFF files produced by Micro-Manager.

Classes
-------
MicroManagerTiffImagingExtractor
    A ImagingExtractor for TIFF files produced by Micro-Manager.
"""

import json
import logging
import re
from collections import Counter
from itertools import islice
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple, Dict

from xml.etree import ElementTree
import numpy as np

from ...imagingextractor import ImagingExtractor
from ...extraction_tools import PathType, get_package, DtypeType
from ...multiimagingextractor import MultiImagingExtractor


def filter_tiff_tag_warnings(record):
    """Filter out the warning messages from tifffile package."""
    return not record.msg.startswith("<tifffile.TiffTag 270 @42054>")


logging.getLogger("tifffile.tifffile").addFilter(filter_tiff_tag_warnings)


def _get_tiff_reader() -> ModuleType:
    """Import the tifffile package and return the module."""
    return get_package(package_name="tifffile", installation_instructions="pip install tifffile")


class MicroManagerTiffImagingExtractor(MultiImagingExtractor):
    """Specialized extractor for reading TIFF files produced via Micro-Manager.

    The image file stacks are saved into multipage TIF files in OME-TIFF format (.ome.tif files),
    each of which are up to around 4GB in size.
    The 'DisplaySettings' JSON file contains the properties of Micro-Manager.
    """

    extractor_name = "MicroManagerTiffImaging"

    def __init__(self, folder_path: PathType):
        """Create a MicroManagerTiffImagingExtractor instance from a folder path that contains the image files.

        Parameters
        ----------
        folder_path: PathType
           The folder path that contains the multipage OME-TIF image files (.ome.tif files) and
           the 'DisplaySettings' JSON file.
        """
        self.tifffile = _get_tiff_reader()
        self.folder_path = Path(folder_path)

        self._ome_tif_files = list(self.folder_path.glob("*.ome.tif"))
        assert self._ome_tif_files, f"The TIF image files are missing from '{folder_path}'."

        # load the 'DisplaySettings.json' file that contains the sampling frequency of images
        settings = self._load_settings_json()
        self._sampling_frequency = float(settings["PlaybackFPS"]["scalar"])

        first_tif = self.tifffile.TiffFile(self._ome_tif_files[0])
        # extract metadata from Micro-Manager
        micromanager_metadata = first_tif.micromanager_metadata
        assert "Summary" in micromanager_metadata, "The 'Summary' field is not found in Micro-Manager metadata."
        self.micromanager_metadata = micromanager_metadata
        self._width = self.micromanager_metadata["Summary"]["Width"]
        self._height = self.micromanager_metadata["Summary"]["Height"]
        self._num_channels = self.micromanager_metadata["Summary"]["Channels"]
        if self._num_channels > 1:
            raise NotImplementedError(
                f"The {self.extractor_name}Extractor does not currently support multiple color channels."
            )
        self._channel_names = self.micromanager_metadata["Summary"]["ChNames"]

        # extact metadata from OME-XML specification
        self._ome_metadata = first_tif.ome_metadata
        ome_metadata_root = self._get_ome_xml_root()

        schema_name = re.findall("\{(.*)\}", ome_metadata_root.tag)[0]
        pixels_element = ome_metadata_root.find(f"{{{schema_name}}}Image/{{{schema_name}}}Pixels")
        self._num_frames = int(pixels_element.attrib["SizeT"])
        self._dtype = np.dtype(pixels_element.attrib["Type"])

        # all the file names are repeated under the TiffData tag
        # the number of occurences of each file path corresponds to the number of frames for a given TIF file
        tiff_data_elements = pixels_element.findall(f"{{{schema_name}}}TiffData")
        file_names = [element[0].attrib["FileName"] for element in tiff_data_elements]

        # count the number of occurrences of each file path and their names
        file_counts = Counter(file_names)
        self._check_missing_files_in_folder(expected_list_of_files=list(file_counts.keys()))
        # Initialize the private imaging extractors with the number of frames for each file
        imaging_extractors = []
        for file_path, num_frames_per_file in file_counts.items():
            extractor = _MicroManagerTiffImagingExtractor(self.folder_path / file_path)
            extractor._dtype = self._dtype
            extractor._num_frames = num_frames_per_file
            extractor._image_size = (self._height, self._width)
            imaging_extractors.append(extractor)
        super().__init__(imaging_extractors=imaging_extractors)

    def _load_settings_json(self) -> Dict[str, Dict[str, str]]:
        """Load the 'DisplaySettings' JSON file.

        Returns
        -------
        settings: Dict[str, Dict[str, str]]
            The dictionary that contains the properties of Micro-Manager.
        """
        file_name = "DisplaySettings.json"
        settings_json_file_path = self.folder_path / file_name
        assert settings_json_file_path.exists(), f"The '{file_name}' file is not found at '{self.folder_path}'."

        with open(settings_json_file_path, "r") as f:
            settings = json.load(f)
        assert "map" in settings, "The Micro-Manager property 'map' key is missing."
        return settings["map"]

    def _get_ome_xml_root(self) -> ElementTree:
        """Parse the OME-XML configuration from string format into element tree and returns the root of this tree.

        Returns
        -------
        root: ElementTree
            The root of the element tree that contains the OME-XML configuration.
        """
        ome_metadata_element = ElementTree.fromstring(self._ome_metadata)
        tree = ElementTree.ElementTree(ome_metadata_element)
        return tree.getroot()

    def _check_missing_files_in_folder(self, expected_list_of_files):
        """Check the presence of each TIF file that is expected to be found in the folder.

        Parameters
        ----------
        expected_list_of_files: list
            The list of file names that are expected to be found in the folder.

        Raises
        ------
        AssertionError
            Raises an error when the files are not found with the name of the missing files.
        """
        missing_files = [
            file_name for file_name in expected_list_of_files if self.folder_path / file_name not in self._ome_tif_files
        ]
        assert (
            not missing_files
        ), f"Some of the TIF image files at '{self.folder_path}' are missing. The list of files that are missing: {missing_files}"

    def _check_consistency_between_imaging_extractors(self):
        """Override the parent class method as none of the properties that are checked are from the sub-imaging extractors."""
        return True

    def get_image_size(self) -> Tuple[int, int]:
        return self._height, self._width

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_channel_names(self) -> list:
        return self._channel_names

    def get_num_channels(self) -> int:
        return self._num_channels

    def get_dtype(self) -> DtypeType:
        return self._dtype


class _MicroManagerTiffImagingExtractor(ImagingExtractor):
    """Private imaging extractor for OME-TIF image format produced by Micro-Manager.

    The private imaging extractor for OME-TIF image format produced by Micro-Manager,
    which defines the get_video() method to return the requested frames from a given file.
    This extractor is not meant to be used as a standalone ImagingExtractor.
    """

    extractor_name = "_MicroManagerTiffImaging"
    is_writable = True
    mode = "file"

    SAMPLING_FREQ_ERROR = "The {}Extractor does not support retrieving the imaging rate."
    CHANNEL_NAMES_ERROR = "The {}Extractor does not support retrieving the name of the channels."
    DATA_TYPE_ERROR = "The {}Extractor does not support retrieving the data type."

    def __init__(self, file_path: PathType):
        """Create a _MicroManagerTiffImagingExtractor instance from a TIFF image file (.ome.tif).

        Parameters
        ----------
        file_path : PathType
            The path to the TIF image file (.ome.tif)
        """
        self.tifffile = _get_tiff_reader()
        self.file_path = file_path

        super().__init__()

        self.pages = self.tifffile.TiffFile(self.file_path).pages
        self._dtype = None
        self._num_frames = None
        self._image_size = None

    def get_num_frames(self):
        return self._num_frames

    def get_num_channels(self) -> int:
        return 1

    def get_image_size(self):
        return self._image_size

    def get_sampling_frequency(self):
        raise NotImplementedError(self.SAMPLING_FREQ_ERROR.format(self.extractor_name))

    def get_channel_names(self) -> list:
        raise NotImplementedError(self.CHANNEL_NAMES_ERROR.format(self.extractor_name))

    def get_dtype(self):
        return self._dtype

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        if start_frame is not None and end_frame is not None and start_frame == end_frame:
            return self.pages[start_frame].asarray()

        end_frame = end_frame or self.get_num_frames()
        start_frame = start_frame or 0
        video = np.zeros(shape=(end_frame - start_frame, *self.get_image_size()), dtype=self.get_dtype())
        for page_ind, page in enumerate(islice(self.pages, start_frame, end_frame)):
            video[page_ind] = page.asarray()
        return video
