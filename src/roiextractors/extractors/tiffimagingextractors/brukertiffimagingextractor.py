import logging
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple
from xml.etree import ElementTree

import numpy as np

from ...imagingextractor import ImagingExtractor
from ...extraction_tools import PathType, get_package, DtypeType


def filter_read_uic_tag_warnings(record):
    return not record.msg.startswith("<tifffile.read_uic_tag>")


logging.getLogger("tifffile.tifffile").addFilter(filter_read_uic_tag_warnings)


def _get_tiff_reader() -> ModuleType:
    return get_package(package_name="tifffile", installation_instructions="pip install tifffile")


class BrukerTiffImagingExtractor(ImagingExtractor):
    extractor_name = "BrukerTiffImaging"
    is_writable = True
    mode = "folder"

    def __init__(self, folder_path: PathType):
        tifffile = _get_tiff_reader()

        super().__init__()
        self.folder_path = Path(folder_path)
        self.xml_metadata = self._get_xml_metadata()
        self._file_paths = self._get_files_names()
        assert list(self.folder_path.glob("*.ome.tif")), f"The TIF image files are missing from '{self.folder_path}'."
        assert len(self._file_paths) == len(
            list(self.folder_path.glob("*.ome.tif"))
        ), f"The number of TIF image files at '{self.folder_path}' should be equal to the number of frames ({len(self._file_paths)}) specified in the XML configuration file."

        with tifffile.TiffFile(self.folder_path / self._file_paths[0], _multifile=False) as tif:
            self._height, self._width = tif.pages.first.shape
            self._dtype = tif.pages.first.dtype

        self._sampling_frequency = 1 / float(self.xml_metadata["framePeriod"])
        file = self._get_xml_root().find(".//File")
        self._channel_names = [file.attrib["channelName"]]

    def _get_xml_root(self):
        xml_file_path = self.folder_path / f"{self.folder_path.stem}.xml"
        assert Path(xml_file_path).is_file(), f"The XML configuration file is not found at '{self.folder_path}'."
        tree = ElementTree.parse(xml_file_path)
        return tree.getroot()

    def _get_xml_metadata(self):
        root = self._get_xml_root()
        xml_metadata = dict()
        xml_metadata.update(**root.attrib)
        for child in root.findall(".//PVStateValue"):
            metadata_root_key = child.attrib["key"]
            if "value" in child.attrib:
                if metadata_root_key in xml_metadata:
                    continue
                xml_metadata[metadata_root_key] = child.attrib["value"]
            else:
                xml_metadata[metadata_root_key] = []
                for indexed_value in child:
                    if "value" not in indexed_value.attrib:
                        for subindexed_value in indexed_value:
                            if "description" in subindexed_value.attrib:
                                xml_metadata[metadata_root_key].append(
                                    {subindexed_value.attrib["description"]: subindexed_value.attrib["value"]}
                                )
                            else:
                                xml_metadata[child.attrib["key"]].append(
                                    {indexed_value.attrib["index"]: subindexed_value.attrib["value"]}
                                )
                    if "description" in indexed_value.attrib:
                        xml_metadata[child.attrib["key"]].append(
                            {indexed_value.attrib["description"]: indexed_value.attrib["value"]}
                        )
                    elif "value" in indexed_value.attrib:
                        xml_metadata[child.attrib["key"]].append(
                            {indexed_value.attrib["index"]: indexed_value.attrib["value"]}
                        )
        return xml_metadata

    def _get_files_names(self):
        return [file.attrib["filename"] for file in self._get_xml_root().findall(".//File")]

    def get_image_size(self) -> Tuple[int, int]:
        return self._height, self._width

    def get_num_frames(self) -> int:
        return len(self._file_paths)

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_channel_names(self) -> list:
        return self._channel_names

    def get_num_channels(self) -> int:
        channel_names = self.get_channel_names()
        return len(channel_names)

    def get_dtype(self) -> DtypeType:
        return self._dtype

    def _frames_iterator(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ):
        tiffile = _get_tiff_reader()

        if start_frame is not None and end_frame is not None:
            if end_frame == start_frame:
                yield tiffile.memmap(self.folder_path / self._file_paths[start_frame], mode="r", _multifile=False)

        for file in self._file_paths[start_frame:end_frame]:
            yield tiffile.memmap(self.folder_path / file, mode="r", _multifile=False)

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        if channel != 0:
            raise NotImplementedError(
                f"The {self.extractor_name}Extractor does not currently support multiple color channels."
            )
        frames = list(self._frames_iterator(start_frame=start_frame, end_frame=end_frame))
        video = np.stack(frames, axis=0)
        return video
