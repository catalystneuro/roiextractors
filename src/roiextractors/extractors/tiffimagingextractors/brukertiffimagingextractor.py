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


def _determine_frame_rate(element: ElementTree.Element) -> Union[float, None]:
    """
    Determines the frame rate from the difference in relative timestamps of the frame elements.
    """
    from neuroconv.utils import calculate_regular_series_rate
    frame_elements = element.findall(".//Frame")
    relative_times = [float(frame.attrib["relativeTime"]) for frame in frame_elements]
    frame_rate = calculate_regular_series_rate(np.array(relative_times))

    return frame_rate


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
        self._tifffile = _get_tiff_reader()

        super().__init__()

        self.folder_path = Path(folder_path)
        tif_file_paths = list(self.folder_path.glob("*.ome.tif"))
        assert tif_file_paths, f"The TIF image files are missing from '{self.folder_path}'."

        self._xml_file_path = self.folder_path / f"{self.folder_path.name}.xml"
        assert self._xml_file_path.is_file(), f"The XML configuration file is not found at '{self.folder_path}'."
        xml_root = self._get_xml_root()

        sequence_elements = xml_root.findall("Sequence")
        self._is_imaging_volumetric = self._check_imaging_is_volumetric()
        if self._is_imaging_volumetric:
            # the number of "Sequence" elements in the XML determine the number of frames for each z plane
            # the number of "Frame" repetitions within a "Sequence" determines the number of z-planes.
            num_frames_first_sequence = len(sequence_elements[0].findall("./Frame"))
            self._num_frames = len(sequence_elements)
            self._num_z_planes = num_frames_first_sequence
        else:
            num_frames = len(xml_root.findall(".//Frame"))
            self._num_frames = num_frames
            self._num_z_planes = 1

        file_elements = xml_root.findall(".//File")
        self._file_paths = [file.attrib["filename"] for file in file_elements]
        assert len(self._file_paths) == len(
            tif_file_paths
        ), f"The number of TIF image files at '{self.folder_path}' should be equal to the number of frames ({len(self._file_paths)}) specified in the XML configuration file."

        with self._tifffile.TiffFile(self.folder_path / self._file_paths[0], _multifile=False) as tif:
            self._height, self._width = tif.pages[0].shape
            self._dtype = tif.pages[0].dtype

        self.xml_metadata = self._get_xml_metadata()
        # determine the true sampling frequency
        # the "framePeriod" in the XML is not trusted (usually higher than the true frame rate)
        frame_rate = _determine_frame_rate(element=xml_root)
        if frame_rate is None and len(sequence_elements) > 1:
            frame_rate = _determine_frame_rate(element=sequence_elements[0])
        assert frame_rate is not None, "Could not determine the frame rate from the XML file."
        self._sampling_frequency = frame_rate

        channel_names = [file.attrib["channelName"] for file in file_elements]
        unique_channel_names = list(set(channel_names))
        # This will be changed soon
        assert len(unique_channel_names) == 1
        self._channel_names = unique_channel_names

    def _get_xml_root(self) -> ElementTree.Element:
        """
        Parses the XML configuration file into element tree and returns the root Element.
        """
        tree = ElementTree.parse(self._xml_file_path)
        return tree.getroot()

    def _check_imaging_is_volumetric(self) -> bool:
        """
        Determines whether imaging is volumetric based on 'zDevice' configuration value.
        The value is expected to be '1' for volumetric and '0' for single plane images.
        """
        xml_root = self._get_xml_root()
        z_device_element = xml_root.find(".//PVStateValue[@key='zDevice']")
        is_volumetric = bool(int(z_device_element.attrib["value"]))

        return is_volumetric

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

    def get_image_size(self) -> Union[Tuple[int, int], Tuple[int, int, int]]:
        if self._is_imaging_volumetric:
            return self._height, self._width, self._num_z_planes

        return self._height, self._width

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

        files_range = self._file_paths[start_frame:end_frame]
        for file in files_range:
            yield self._tifffile.memmap(self.folder_path / file, mode="r", _multifile=False)

    def _get_video_for_volumetric_imaging(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:

        start_frame = start_frame or 0
        end_frame = end_frame or self.get_num_frames()

        image_shape = (end_frame - start_frame, *self.get_image_size())
        video = np.zeros(shape=image_shape, dtype=self.get_dtype())
        for frame_ind, frame_num in enumerate(np.arange(start_frame, end_frame)):
            start = frame_num * self._num_z_planes
            stop = (frame_num * self._num_z_planes) + self._num_z_planes
            frames = []
            for file in self._file_paths[start:stop]:
                frames.append(self._tifffile.memmap(self.folder_path / file, mode="r", _multifile=False))
            video[frame_ind] = np.stack(frames, axis=-1)
        return video

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        tifffile = _get_tiff_reader()

        if channel != 0:
            raise NotImplementedError(
                f"The {self.extractor_name}Extractor does not currently support multiple color channels."
            )
        if self._is_imaging_volumetric:
            return self._get_video_for_volumetric_imaging(
                start_frame=start_frame,
                end_frame=end_frame,
            )

        if start_frame is not None and end_frame is not None:
            if end_frame == start_frame:
                return tifffile.memmap(
                    self.folder_path / self._file_paths[start_frame],
                    mode="r",
                    _multifile=False,
                )

        frames = list(self._frames_iterator_for_single_z_plane(start_frame=start_frame, end_frame=end_frame))
        return np.stack(frames, axis=0)
