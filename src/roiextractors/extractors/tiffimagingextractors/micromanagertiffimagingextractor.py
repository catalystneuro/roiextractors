import json
import re
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple

from xml.etree import ElementTree
import numpy as np

from ...imagingextractor import ImagingExtractor
from ...extraction_tools import PathType, get_package, DtypeType
from ...multiimagingextractor import MultiImagingExtractor


def _get_tiff_reader() -> ModuleType:
    return get_package(package_name="tifffile", installation_instructions="pip install tifffile")


class MicroManagerTiffImagingExtractor(MultiImagingExtractor):
    extractor_name = "MicroManagerTiffImaging"
    installed = True
    installation_mesg = ""

    def __init__(self, folder_path: PathType):
        """
        The imaging extractor for the Micro-Manager TIF image format.
        This format consists of multiple TIF image files in multipage OME-TIF format (.ome.tif files)
        and 'DisplaySettings' JSON file.

        Parameters
        ----------
        folder_path: PathType
           The folder path that contains the multipage OME-TIF image files (.ome.tif files) and
           the 'DisplaySettings' JSON file.
        """
        self.tifffile = _get_tiff_reader()
        self.folder_path = Path(folder_path)

        ome_tif_files = list(self.folder_path.glob("*.ome.tif"))
        assert ome_tif_files, f"The TIF image files are missing from '{folder_path}'."

        # load the 'DisplaySettings.json' file that contains the sampling frequency of images
        settings = self._load_settings_json()
        self._sampling_frequency = float(settings["map"]["PlaybackFPS"]["scalar"])

        # load the first tif
        first_tif = self.tifffile.TiffFile(self.folder_path / ome_tif_files[0])

        # extract metadata from Micro-Manager
        self.micromanager_metadata = first_tif.micromanager_metadata
        self._width = self.micromanager_metadata["Summary"]["Width"]
        self._height = self.micromanager_metadata["Summary"]["Height"]
        self._num_channels = self.micromanager_metadata["Summary"]["Channels"]
        self._channel_names = self.micromanager_metadata["Summary"]["ChNames"]

        # extact metadata from OME XML
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
        # Initialize the private imaging extractors with the number of frames for each file
        imaging_extractors = []
        # TODO make sure Counter returns the right order of image files
        for file_path, num_frames_per_file in file_counts.items():
            extractor = _SubMicroManagerTiffImagingExtractor(self.folder_path / file_path)
            extractor._num_frames = num_frames_per_file
            imaging_extractors.append(extractor)
        super().__init__(imaging_extractors=imaging_extractors)

    def _load_settings_json(self):
        file_name = "DisplaySettings.json"
        settings_json_file_path = self.folder_path / file_name
        assert settings_json_file_path.exists(), f"'{file_name}' file not found at '{self.folder_path}'."

        with open(settings_json_file_path, "r") as f:
            settings = json.load(f)
        return settings

    def _get_ome_xml_root(self):
        """
        Parses the OME-XML configuration from string format into element tree and returns the root of this tree.
        """
        ome_metadata_element = ElementTree.fromstring(self._ome_metadata)
        tree = ElementTree.ElementTree(ome_metadata_element)
        return tree.getroot()

    def _check_consistency_between_imaging_extractors(self):
        """Overrides the parent class method as none of the properties that are checked are from the sub-imaging extractors."""
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


class _SubMicroManagerTiffImagingExtractor(ImagingExtractor):
    extractor_name = "_SubMicroManagerTiffImaging"
    is_writable = True
    mode = "file"

    IMAGE_SIZE_ERROR = "The {}Extractor does not support retrieving the image size."
    SAMPLING_FREQ_ERROR = "The {}Extractor does not support retrieving the imaging rate."
    CHANNEL_NAMES_ERROR = "The {}Extractor does not support retrieving the name of the channels."
    DATA_TYPE_ERROR = "The {}Extractor does not support retrieving the data type."

    def __init__(self, file_path: PathType):
        """
        The private imaging extractor for the WideField OME-TIF image format.
        This extractor is not meant to be used as a standalone ImagingExtractor.
        TODO: more explanation

        Parameters
        ----------
        file_path : PathType
            The path to the TIF image file (.ome.tif)
        """
        self.tifffile = _get_tiff_reader()
        self.file_path = file_path

        super().__init__()

        self.pages = self.tifffile.TiffFile(self.file_path).pages
        self._num_frames = None

    def get_num_frames(self):
        return self._num_frames

    def get_num_channels(self) -> int:
        return 1

    def get_image_size(self):
        raise NotImplementedError(self.IMAGE_SIZE_ERROR.format(self.extractor_name))

    def get_sampling_frequency(self):
        raise NotImplementedError(self.SAMPLING_FREQ_ERROR.format(self.extractor_name))

    def get_channel_names(self) -> list:
        raise NotImplementedError(self.CHANNEL_NAMES_ERROR.format(self.extractor_name))

    def get_dtype(self):
        raise NotImplementedError(self.DATA_TYPE_ERROR.format(self.extractor_name))

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        if channel != 0:
            raise NotImplementedError(
                f"The {self.extractor_name}Extractor does not currently support multiple color channels."
            )
        if start_frame is not None and end_frame is not None and start_frame == end_frame:
            page = self.pages[start_frame]
            return page.asarray()[np.newaxis, ...]

        pages = self.pages[start_frame:end_frame]
        frames = [page.asarray() for page in pages]
        return np.stack(frames)
