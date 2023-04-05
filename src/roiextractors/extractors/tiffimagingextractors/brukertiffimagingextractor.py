import logging
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple, Union, List, Iterable, Dict
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
        """
        The imaging extractor for the Bruker TIF image format.
        This format consists of multiple TIF image files (.ome.tif) and configuration files (.xml, .env).

        Parameters
        ----------
        folder_path : PathType
            The path to the folder that contains the Bruker TIF image files (.ome.tif) and configuration files (.xml, .env).
        """
        tifffile = _get_tiff_reader()

        super().__init__()
        self.folder_path = Path(folder_path)
        tif_file_paths = list(self.folder_path.glob("*.ome.tif"))
        assert tif_file_paths, f"The TIF image files are missing from '{self.folder_path}'."

        xml_root = self._get_xml_root()
        sequences = xml_root.findall("Sequence")
        num_sequences = len(sequences)
        num_frames_first_sequence = len(sequences[0].findall("./Frame"))
        # The number of "Sequence" elements in the document determine whether multiple z-planes are present.
        if num_sequences == 1:
            self._num_frames = num_frames_first_sequence
            self._num_z_planes = 1
        else:
            # Note: could it happen that for the last sequence we have less frames then for the first one?
            num_frames_last_sequence = len(sequences[-1].findall("Frame"))
            if num_frames_first_sequence != num_frames_last_sequence:
                raise NotImplementedError(
                    f"Not sure how to handle final stack because it was found with fewer z-planes ({num_frames_last_sequence}, expected: {num_frames_first_sequence}).",
                )
            self._num_frames = num_sequences
            self._num_z_planes = num_frames_first_sequence

        # The ordered list of files in the XML document
        files = xml_root.findall(".//File")
        self._file_paths = [file.attrib["filename"] for file in files]
        tif_file_paths = list(self.folder_path.glob("*.ome.tif"))

        assert len(self._file_paths) == len(
                tif_file_paths
            ), f"The number of TIF image files at '{self.folder_path}' should be equal to the number of frames ({len(self._file_paths)}) specified in the XML configuration file."

        with tifffile.TiffFile(self.folder_path / self._file_paths[0], _multifile=False) as tif:
            self._height, self._width = tif.pages.first.shape
            self._dtype = tif.pages.first.dtype

        self.xml_metadata = self._get_xml_metadata()
        self._sampling_frequency = 1 / float(self.xml_metadata["framePeriod"])

        channel_names = [file.attrib["channelName"] for file in files]
        unique_channel_names = list(set(channel_names))
        self._channel_names = unique_channel_names

    def _get_xml_root(self):
        """
        Parses the XML configuration file into element tree and returns the root of this tree.
        """
        xml_file_path = self.folder_path / f"{self.folder_path.name}.xml"
        assert Path(xml_file_path).is_file(), f"The XML configuration file is not found at '{self.folder_path}'."
        tree = ElementTree.parse(xml_file_path)
        return tree.getroot()

    def _get_xml_metadata(self) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """
        Parses the metadata in the root element that are under "PVStateValue" tag into
        a dictionary.
        """
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
                    if "description" in indexed_value.attrib:
                        xml_metadata[child.attrib["key"]].append(
                            {indexed_value.attrib["description"]: indexed_value.attrib["value"]}
                        )
                    elif "value" in indexed_value.attrib:
                        xml_metadata[child.attrib["key"]].append(
                            {indexed_value.attrib["index"]: indexed_value.attrib["value"]}
                        )
                    else:
                        for subindexed_value in indexed_value:
                            if "description" in subindexed_value.attrib:
                                xml_metadata[metadata_root_key].append(
                                    {subindexed_value.attrib["description"]: subindexed_value.attrib["value"]}
                                )
                            else:
                                xml_metadata[child.attrib["key"]].append(
                                    {indexed_value.attrib["index"]: subindexed_value.attrib["value"]}
                                )
        return xml_metadata

    def get_image_size(self) -> Tuple[int, int, int]:
        return self._height, self._width, self._num_z_planes

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_channel_names(self) -> List[str]:
        return self._channel_names

    def get_num_channels(self) -> int:
        channel_names = self.get_channel_names()
        return len(channel_names)

    def get_dtype(self) -> DtypeType:
        return self._dtype

    def _frames_iterator_for_single_z_plane(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> Iterable[np.memmap]:
        tifffile = _get_tiff_reader()

        if start_frame is not None and end_frame is not None:
            if end_frame == start_frame:
                for file in self._file_paths[start_frame]:
                    yield tifffile.memmap(self.folder_path / file, mode="r", _multifile=False)

        files_range = self._file_paths[start_frame:end_frame]
        for file in files_range:
            yield tifffile.memmap(self.folder_path / file, mode="r", _multifile=False)

    def _get_video_for_multi_z_planes(
            self,
            start_frame: Optional[int] = None,
            end_frame: Optional[int] = None,
    ) -> np.ndarray:
        tifffile = _get_tiff_reader()

        image_shape = (end_frame - start_frame, *self.get_image_size())
        video = np.zeros(shape=image_shape, dtype=self.get_dtype())
        for frame_ind, frame_num in enumerate(np.arange(start_frame, end_frame)):
            start_f = frame_num * self._num_z_planes
            end_f = (frame_num * self._num_z_planes) + self._num_z_planes
            frames = []
            for file in self._file_paths[start_f:end_f]:
                frames.append(tifffile.memmap(self.folder_path / file, mode="r", _multifile=False))
            video[frame_ind] = np.stack(frames, axis=-1)
        return video

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        if channel != 0:
            raise NotImplementedError(
                f"The {self.extractor_name}Extractor does not currently support multiple color channels."
            )
        if self._num_z_planes == 1:
            frames = list(self._frames_iterator_for_single_z_plane(start_frame=start_frame, end_frame=end_frame))
            video = np.stack(frames, axis=0)
            return video[..., np.newaxis]

        start = start_frame if start_frame is not None else 0
        stop = end_frame if end_frame is not None else self.get_num_frames()
        video = self._get_video_for_multi_z_planes(start_frame=start, end_frame=stop)
        return video
