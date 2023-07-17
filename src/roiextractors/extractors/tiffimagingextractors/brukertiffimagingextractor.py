import logging
import re
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple, Union, List, Dict
from xml.etree import ElementTree

import numpy as np

from ...multiimagingextractor import MultiImagingExtractor
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


class BrukerTiffImagingExtractor(MultiImagingExtractor):
    extractor_name = "BrukerTiffImaging"
    is_writable = True
    mode = "folder"

    @classmethod
    def get_streams(cls, folder_path: PathType) -> List[str]:
        natsort = get_package(package_name="natsort", installation_instructions="pip install natsort")

        folder_path = Path(folder_path)
        file_paths = list(folder_path.glob("*.ome.tif"))
        stream_names = natsort.natsorted(
            set((re.findall(r"[Cc][Hh]\d+", file_name.name)[0] for file_name in file_paths))
        )
        return stream_names

    def __init__(self, folder_path: PathType, stream_name: Optional[str] = None):
        """
        The imaging extractor for the Bruker TIF image format.
        This format consists of multiple TIF image files (.ome.tif) and configuration files (.xml, .env).

        Parameters
        ----------
        folder_path : PathType
            The path to the folder that contains the Bruker TIF image files (.ome.tif) and configuration files (.xml, .env).
        stream_name: str, optional
            The name of the recording channel (e.g. "Ch2").
        """
        natsort = get_package(package_name="natsort", installation_instructions="pip install natsort")
        self._tifffile = _get_tiff_reader()

        folder_path = Path(folder_path)
        tif_file_paths = natsort.natsorted(folder_path.glob("*.ome.tif"))
        assert tif_file_paths, f"The TIF image files are missing from '{folder_path}'."

        stream_names = self.get_streams(folder_path=folder_path)
        if stream_name is None:
            if len(stream_names) > 1:
                raise ValueError(
                    "More than one recording stream is detected! Please specify which stream you wish to load with the `stream_name` argument. "
                    "To see what streams are available, call `BrukerTiffImagingExtractor.get_stream_names(folder_path=...)`."
                )
            stream_name = stream_names[0]

        if stream_name is not None and stream_name not in stream_names:
            raise ValueError(f"The selected stream '{stream_name}' is not in the available streams '{stream_names}'!")
        self._stream_name = stream_name

        if len(stream_names) > 1:
            tif_file_paths = [file_path for file_path in tif_file_paths if stream_name in file_path.name]

        self._xml_file_path = folder_path / f"{folder_path.name}.xml"
        assert self._xml_file_path.is_file(), f"The XML configuration file is not found at '{folder_path}'."
        self._xml_root = self._get_xml_root()

        file_elements = self._xml_root.findall(f".//File[@channelName='{stream_name}']")
        file_names = [file.attrib["filename"] for file in file_elements]
        # count the number of occurrences of each file path and their names
        # files that contain stacks of images (multi-page tiffs) will appear repeated (number of repetition is the number of frames in the tif file)
        file_counts = Counter(file_names)
        file_paths_per_cycle = defaultdict(list)  # Use defaultdict to automatically create lists for each cycle
        pattern = r"_(?P<cycle>Cycle\d+)_"
        for file_name in file_counts.keys():
            cycle_match = re.search(pattern, file_name)
            file_paths_per_cycle[cycle_match["cycle"]].append(str(Path(folder_path) / file_name))

        with self._tifffile.TiffFile(folder_path / tif_file_paths[0], _multifile=False) as tif:
            self._height, self._width = tif.pages[0].shape
            self._dtype = tif.pages[0].dtype

        self.xml_metadata = self._get_xml_metadata()

        sequence_elements = self._xml_root.findall("Sequence")
        # determine the true sampling frequency
        # the "framePeriod" in the XML is not trusted (usually higher than the true frame rate)
        frame_rate = _determine_frame_rate(element=self._xml_root)
        if frame_rate is None and len(sequence_elements) > 1:
            frame_rate = _determine_frame_rate(element=sequence_elements[0])
        assert frame_rate is not None, "Could not determine the frame rate from the XML file."
        self._sampling_frequency = frame_rate
        self._channel_names = [stream_name]

        imaging_extractors = []
        is_imaging_volumetric = self._check_imaging_is_volumetric()
        if is_imaging_volumetric:
            # the number of "Sequence" elements in the XML determine the number of frames for each z plane
            # the number of "Frame" repetitions within a "Sequence" determines the number of z-planes.
            num_frames_first_sequence = len(sequence_elements[0].findall("./Frame"))
            self._num_z_planes = num_frames_first_sequence

            for cycle_num, file_paths in file_paths_per_cycle.items():
                extractor = _BrukerTiffMultiPlaneImagingExtractor(file_paths=natsort.natsorted(file_paths))
                extractor._image_size = (self._height, self._width, self._num_z_planes)
                extractor._num_z_planes = self._num_z_planes
                extractor._dtype = self._dtype
                imaging_extractors.append(extractor)

        else:
            for file_name, num_frames in file_counts.items():
                extractor = _BrukerTiffSinglePlaneImagingExtractor(file_path=str(Path(folder_path) / file_name))
                extractor._num_frames = num_frames
                extractor._image_size = (self._height, self._width)
                extractor._dtype = self._dtype
                imaging_extractors.append(extractor)

        super().__init__(imaging_extractors=imaging_extractors)

    def _check_consistency_between_imaging_extractors(self):
        """Overrides the parent class method as none of the properties that are checked are from the sub-imaging extractors."""
        return True

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
        z_device_element = self._xml_root.find(".//PVStateValue[@key='zDevice']")
        is_volumetric = bool(int(z_device_element.attrib["value"]))

        return is_volumetric

    def _get_xml_metadata(self) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """
        Parses the metadata in the root element that are under "PVStateValue" tag into
        a dictionary.
        """
        xml_metadata = dict()
        xml_metadata.update(**self._xml_root.attrib)
        for child in self._xml_root.findall(".//PVStateValue"):
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

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_channel_names(self) -> List[str]:
        return self._channel_names

    def get_num_channels(self) -> int:
        return 1

    def get_dtype(self) -> DtypeType:
        return self._dtype


class _BrukerTiffSinglePlaneImagingExtractor(ImagingExtractor):
    extractor_name = "_BrukerTiffSinglePlaneImaging"
    is_writable = True
    mode = "file"

    SAMPLING_FREQ_ERROR = "The {}Extractor does not support retrieving the imaging rate."
    CHANNEL_NAMES_ERROR = "The {}Extractor does not support retrieving the name of the channels."
    DATA_TYPE_ERROR = "The {}Extractor does not support retrieving the data type."

    def __init__(self, file_path: PathType):
        """
        The private imaging extractor for OME-TIF image format produced by Bruker,
        which defines the get_video() method to return the requested frames from a given file.
        This extractor is not meant to be used as a standalone ImagingExtractor.

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

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_num_channels(self) -> int:
        return 1

    def get_image_size(self) -> Tuple[int, int]:
        return self._image_size

    def get_sampling_frequency(self):
        raise NotImplementedError(self.SAMPLING_FREQ_ERROR.format(self.extractor_name))

    def get_channel_names(self) -> list:
        raise NotImplementedError(self.CHANNEL_NAMES_ERROR.format(self.extractor_name))

    def get_dtype(self):
        raise NotImplementedError(self.DATA_TYPE_ERROR.format(self.extractor_name))

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        if start_frame is not None and end_frame is not None and start_frame == end_frame:
            return self.pages[start_frame].asarray()

        end_frame = end_frame or self.get_num_frames()
        start_frame = start_frame or 0

        image_shape = (end_frame - start_frame, *self.get_image_size())
        video = np.zeros(shape=image_shape, dtype=self._dtype)
        for page_ind, page in enumerate(islice(self.pages, start_frame, end_frame)):
            video[page_ind] = page.asarray()

        return video


class _BrukerTiffMultiPlaneImagingExtractor(ImagingExtractor):
    extractor_name = "_BrukerTiffMultiPlaneImaging"
    is_writable = True
    mode = "file"

    SAMPLING_FREQ_ERROR = "The {}Extractor does not support retrieving the imaging rate."
    CHANNEL_NAMES_ERROR = "The {}Extractor does not support retrieving the name of the channels."
    DATA_TYPE_ERROR = "The {}Extractor does not support retrieving the data type."

    def __init__(self, file_paths: List[PathType]):
        """
        The private imaging extractor for OME-TIF image format produced by Bruker,
        which defines the get_video() method to return the requested frames from a volume.
        This extractor is not meant to be used as a standalone ImagingExtractor.

        Parameters
        ----------
        file_paths : List of PathType
            The list of file path to the TIF image file (.ome.tif) that belong to the same volume.
        """
        self.tifffile = _get_tiff_reader()
        self.file_paths = file_paths

        self._num_z_planes = None
        self._image_size = None

        with self.tifffile.TiffFile(file_paths[0]) as tif:
            self._num_frames = len(tif.pages)
            self._dtype = tif.pages[0].dtype

        super().__init__()

    def get_image_size(self) -> Tuple[int, int, int]:
        return self._image_size

    def get_num_channels(self) -> int:
        return 1

    def get_num_frames(self):
        return self._num_frames

    def get_dtype(self):
        return self._dtype

    def get_sampling_frequency(self):
        raise NotImplementedError(self.SAMPLING_FREQ_ERROR.format(self.extractor_name))

    def get_channel_names(self) -> list:
        raise NotImplementedError(self.CHANNEL_NAMES_ERROR.format(self.extractor_name))

    def _get_video(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:
        if start_frame is not None and end_frame is not None and start_frame == end_frame:
            return self.pages[start_frame].asarray()[np.newaxis, ...]

        end_frame = end_frame or self.get_num_frames()
        start_frame = start_frame or 0

        video = np.zeros(shape=(end_frame - start_frame, *self.pages[0].shape), dtype=self.get_dtype())
        for page_ind, page in enumerate(islice(self.pages, start_frame, end_frame)):
            video[page_ind] = page.asarray()
        return video

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        if start_frame is not None and end_frame is not None and start_frame == end_frame:
            end_frame = start_frame + 1

        if end_frame is None:
            end_frame = self.get_num_frames()
        start_frame = start_frame or 0

        image_shape = (end_frame - start_frame, *self.get_image_size())
        video = np.zeros(shape=image_shape, dtype=self.get_dtype())
        for plane_ind in range(self._num_z_planes):
            self.pages = self.tifffile.TiffFile(self.file_paths[plane_ind]).pages
            video[..., plane_ind] = self._get_video(start_frame, end_frame)

        return video
