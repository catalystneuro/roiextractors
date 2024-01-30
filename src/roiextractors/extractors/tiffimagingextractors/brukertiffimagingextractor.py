"""ImagingExtractors for the TIFF image format produced by Bruker.

Classes
-------
BrukerTiffSinglePlaneImagingExtractor
    A ImagingExtractor for TIFF files produced by Bruker with only 1 plane.
BrukerTiffMultiPlaneImagingExtractor
    A MultiImagingExtractor for TIFF files produced by Bruker with multiple planes.
"""

import logging
import re
from collections import Counter
from itertools import islice
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple, Union, List, Dict
from xml.etree import ElementTree

import numpy as np

from ...multiimagingextractor import MultiImagingExtractor
from ...imagingextractor import ImagingExtractor
from ...extraction_tools import PathType, get_package, DtypeType, ArrayType


def filter_read_uic_tag_warnings(record):
    """Filter out the warnings from tifffile.read_uic_tag() that are not relevant to the user."""
    return not record.msg.startswith("<tifffile.read_uic_tag>")


logging.getLogger("tifffile.tifffile").addFilter(filter_read_uic_tag_warnings)


def _get_tiff_reader() -> ModuleType:
    """Return the tifffile module."""
    return get_package(package_name="tifffile", installation_instructions="pip install tifffile")


def _determine_frame_rate(element: ElementTree.Element, file_names: Optional[List[str]] = None) -> Union[float, None]:
    """Determine the frame rate from the difference in relative timestamps of the frame elements."""
    from neuroconv.utils import calculate_regular_series_rate

    frame_elements = element.findall(".//Frame")
    if file_names:
        frame_elements = [
            frame for frame in frame_elements for file in frame.findall("File") if file.attrib["filename"] in file_names
        ]

    relative_times = [float(frame.attrib["relativeTime"]) for frame in frame_elements]
    frame_rate = calculate_regular_series_rate(np.array(relative_times))

    return frame_rate


def _determine_imaging_is_volumetric(folder_path: PathType) -> bool:
    """Determine whether imaging is volumetric based on 'zDevice' configuration value.

    Parameters
    ----------
    folder_path : PathType
        The path to the folder that contains the Bruker TIF image files (.ome.tif) and configuration files (.xml, .env).

    Returns
    -------
    is_volumetric: bool
        True if the imaging is volumetric (multiplane), False otherwise (single plane).
    """
    folder_path = Path(folder_path)
    xml_file_path = folder_path / f"{folder_path.name}.xml"
    assert xml_file_path.is_file(), f"The XML configuration file is not found at '{xml_file_path}'."

    is_volumetric = False
    for event, elem in ElementTree.iterparse(xml_file_path, events=("start",)):
        if elem.tag == "PVStateValue" and elem.attrib.get("key") == "zDevice":
            is_volumetric = bool(int(elem.attrib["value"]))
            break  # Stop parsing as we've found the required element

    return is_volumetric


def _parse_xml(folder_path: PathType) -> ElementTree.Element:
    """Parse the XML configuration file into element tree and returns the root Element."""
    folder_path = Path(folder_path)
    xml_file_path = folder_path / f"{folder_path.name}.xml"
    assert xml_file_path.is_file(), f"The XML configuration file is not found at '{folder_path}'."
    tree = ElementTree.parse(xml_file_path)
    return tree.getroot()


class BrukerTiffMultiPlaneImagingExtractor(MultiImagingExtractor):
    """A MultiImagingExtractor for TIFF files produced by Bruke with multiple planes.

    This format consists of multiple TIF image files (.ome.tif) and configuration files (.xml, .env).
    """

    extractor_name = "BrukerTiffMultiPlaneImaging"
    is_writable = True
    mode = "folder"

    @classmethod
    def get_streams(cls, folder_path: PathType) -> dict:
        """Get the available streams from the Bruker TIF image files (.ome.tif) and configuration files (.xml, .env).

        Parameters
        ----------
        folder_path : PathType
            The path to the folder that contains the Bruker TIF image files (.ome.tif) and configuration files (.xml, .env).

        Returns
        -------
        streams: dict
            The dictionary of available streams.
        """
        natsort = get_package(package_name="natsort", installation_instructions="pip install natsort")

        folder_path = Path(folder_path)
        xml_file_path = folder_path / f"{folder_path.name}.xml"
        assert xml_file_path.is_file(), f"The XML configuration file is not found at '{folder_path}'."

        channel_names = set()
        channel_ids = set()
        file_names = []

        # Parse the XML file iteratively to find the first Sequence element
        first_sequence_element = None
        for _, elem in ElementTree.iterparse(xml_file_path, events=("end",)):
            if elem.tag == "Sequence":
                first_sequence_element = elem
                break

        if first_sequence_element is None:
            raise ValueError("No Sequence element found in the XML configuration file. Can't get streams")

        # Then in the first Sequence we find all the Frame elements
        if first_sequence_element is not None:
            # Iterate over all Frame elements within the first Sequence
            frame_elements = first_sequence_element.findall(".//Frame")
            for frame_elemenet in frame_elements:
                # Iterate over all File elements within each Frame
                for file_elem in frame_elemenet.findall("File"):
                    channel_names.add(file_elem.attrib["channelName"])
                    channel_ids.add(file_elem.attrib["channel"])
                    file_names.append(file_elem.attrib["filename"])

        unique_channel_names = natsort.natsorted(channel_names)
        unique_channel_ids = natsort.natsorted(channel_ids)

        streams = dict(channel_streams=unique_channel_names)
        streams["plane_streams"] = dict()

        if not _determine_imaging_is_volumetric(folder_path=folder_path):
            return streams

        for channel_id, channel_name in zip(unique_channel_ids, unique_channel_names):
            plane_naming_pattern = rf"(?P<stream_name>Ch{channel_id}_\d+)"
            regular_expression_matches = [re.search(plane_naming_pattern, filename) for filename in file_names]
            plane_stream_names = [matches["stream_name"] for matches in regular_expression_matches if matches]

            unique_plane_stream_names = natsort.natsorted(set(plane_stream_names))
            streams["plane_streams"][channel_name] = unique_plane_stream_names

        return streams

    def __init__(
        self,
        folder_path: PathType,
        stream_name: Optional[str] = None,
    ):
        """Create a BrukerTiffMultiPlaneImagingExtractor instance from a folder path that contains the image files.

        Parameters
        ----------
        folder_path : PathType
            The path to the folder that contains the Bruker TIF image files (.ome.tif) and configuration files (.xml, .env).
        stream_name: str, optional
            The name of the recording channel (e.g. "Ch2").

        Raises
        ------
        ValueError
            If more than one recording stream is detected.
        ValueError
            If the selected stream is not in the available plane_streams.
        AssertionError
            If the TIF image files are missing from the folder.
        AssertionError
            If the imaging is not volumetric.
        """
        self._tifffile = _get_tiff_reader()

        folder_path = Path(folder_path)
        tif_file_paths = list(folder_path.glob("*.ome.tif"))
        assert tif_file_paths, f"The TIF image files are missing from '{folder_path}'."

        streams = self.get_streams(folder_path=folder_path)
        plane_streams = streams["plane_streams"]

        assert len(plane_streams) > 0, (
            f"{self.extractor_name}Extractor is for volumetric imaging. "
            "For single imaging plane data use BrukerTiffSinglePlaneImagingExtractor."
        )

        if stream_name is None:
            if len(streams["channel_streams"]) > 1:
                raise ValueError(
                    "More than one recording stream is detected! Please specify which stream you wish to load with the `stream_name` argument. "
                    "The following channel streams are available:  \n"
                    f"{streams['channel_streams']}"
                )
            channel_stream_name = streams["channel_streams"][0]
            stream_name = streams["plane_streams"][channel_stream_name][0]

        channel_stream_name = stream_name.split("_")[0]
        plane_stream_names = streams["plane_streams"][channel_stream_name]
        if stream_name is not None and stream_name not in plane_stream_names:
            raise ValueError(
                f"The selected stream '{stream_name}' is not in the available plane_streams '{plane_stream_names}'!"
            )

        self.folder_path = Path(folder_path)

        self.stream_name = stream_name
        self._num_planes_per_channel_stream = len(plane_stream_names)

        imaging_extractors = []
        for stream_name in plane_stream_names:
            extractor = BrukerTiffSinglePlaneImagingExtractor(folder_path=folder_path, stream_name=stream_name)
            imaging_extractors.append(extractor)

        super().__init__(imaging_extractors=imaging_extractors)

        self._num_frames = self._imaging_extractors[0].get_num_frames()
        self._image_size = *self._imaging_extractors[0].get_image_size(), self._num_planes_per_channel_stream
        self.xml_metadata = self._imaging_extractors[0].xml_metadata

        self._start_frames = [0] * self._num_planes_per_channel_stream
        self._end_frames = [self._num_frames] * self._num_planes_per_channel_stream

    # TODO: fix this method so that it is consistent with base multiimagingextractor method (i.e. num_rows, num_columns)
    def get_image_size(self) -> Tuple[int, int, int]:
        return self._image_size

    def get_num_frames(self) -> int:
        return self._imaging_extractors[0].get_num_frames()

    def get_sampling_frequency(self) -> float:
        return self._imaging_extractors[0].get_sampling_frequency() * self._num_planes_per_channel_stream

    def get_frames(self, frame_idxs: ArrayType, channel: Optional[int] = 0) -> np.ndarray:
        if isinstance(frame_idxs, (int, np.integer)):
            frame_idxs = [frame_idxs]
        frame_idxs = np.array(frame_idxs)
        assert np.all(frame_idxs < self.get_num_frames()), "'frame_idxs' exceed number of frames"

        frames_shape = (len(frame_idxs),) + self.get_image_size()
        frames = np.empty(shape=frames_shape, dtype=self.get_dtype())

        for plane_ind, extractor in enumerate(self._imaging_extractors):
            frames[..., plane_ind] = extractor.get_frames(frame_idxs)

        return frames

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        if channel != 0:
            raise NotImplementedError(
                f"MultiImagingExtractors for multiple channels have not yet been implemented! (Received '{channel}'."
            )

        start = start_frame if start_frame is not None else 0
        stop = end_frame if end_frame is not None else self.get_num_frames()

        video_shape = (stop - start,) + self.get_image_size()
        video = np.empty(shape=video_shape, dtype=self.get_dtype())

        for plane_ind, extractor in enumerate(self._imaging_extractors):
            video[..., plane_ind] = extractor.get_video(start_frame=start, end_frame=stop)

        return video


class BrukerTiffSinglePlaneImagingExtractor(MultiImagingExtractor):
    """A MultiImagingExtractor for TIFF files produced by Bruker with only 1 plane."""

    extractor_name = "BrukerTiffSinglePlaneImaging"
    is_writable = True
    mode = "folder"

    @classmethod
    def get_streams(cls, folder_path: PathType) -> dict:
        """Get the available streams from the Bruker TIF image files (.ome.tif) and configuration files (.xml, .env).

        Parameters
        ----------
        folder_path : PathType
            The path to the folder that contains the Bruker TIF image files (.ome.tif) and configuration files (.xml, .env).

        Returns
        -------
        streams: dict
            The dictionary of available streams.
        """
        natsort = get_package(package_name="natsort", installation_instructions="pip install natsort")
        xml_root = _parse_xml(folder_path=folder_path)
        channel_names = [file.attrib["channelName"] for file in xml_root.findall(".//File")]
        unique_channel_names = natsort.natsorted(set(channel_names))
        streams = dict(channel_streams=unique_channel_names)
        return streams

    def __init__(self, folder_path: PathType, stream_name: Optional[str] = None):
        """Create a BrukerTiffSinglePlaneImagingExtractor instance from a folder path that contains the image files.

        Parameters
        ----------
        folder_path : PathType
            The path to the folder that contains the Bruker TIF image files (.ome.tif) and configuration files (.xml, .env).
        stream_name: str, optional
            The name of the recording channel (e.g. "Ch2").
        """
        self._tifffile = _get_tiff_reader()

        folder_path = Path(folder_path)
        tif_file_paths = list(folder_path.glob("*.ome.tif"))
        assert tif_file_paths, f"The TIF image files are missing from '{folder_path}'."

        streams = self.get_streams(folder_path=folder_path)
        if stream_name is None:
            if len(streams["channel_streams"]) > 1:
                raise ValueError(
                    "More than one recording stream is detected! Please specify which stream you wish to load with the `stream_name` argument. "
                    "To see what streams are available, call `BrukerTiffSinglePlaneImagingExtractor.get_stream_names(folder_path=...)`."
                )
            stream_name = streams["channel_streams"][0]

        self.stream_name = stream_name
        channel_stream_name = self.stream_name.split("_")[0]
        if self.stream_name is not None and channel_stream_name not in streams["channel_streams"]:
            raise ValueError(
                f"The selected stream '{self.stream_name}' is not in the available channel_streams '{streams['channel_streams']}'!"
            )

        self._xml_root = _parse_xml(folder_path=folder_path)
        file_elements = self._xml_root.findall(".//File")
        file_names = [file.attrib["filename"] for file in file_elements]
        file_names_for_stream = [file for file in file_names if self.stream_name in file]
        # determine image shape and data type from first file
        with self._tifffile.TiffFile(folder_path / file_names_for_stream[0], _multifile=False) as tif:
            self._height, self._width = tif.pages[0].shape
            self._dtype = tif.pages[0].dtype

        sequence_elements = self._xml_root.findall("Sequence")
        # determine the true sampling frequency
        # the "framePeriod" in the XML is not trusted (usually higher than the true frame rate)
        frame_rate = _determine_frame_rate(element=self._xml_root, file_names=file_names_for_stream)
        if frame_rate is None and len(sequence_elements) > 1:
            frame_rate = _determine_frame_rate(element=sequence_elements[0], file_names=file_names_for_stream)
        assert frame_rate is not None, "Could not determine the frame rate from the XML file."
        self._sampling_frequency = frame_rate
        self._channel_names = [self.stream_name.split("_")[0]]

        # count the number of occurrences of each file path and their names
        # files that contain stacks of images (multi-page tiffs) will appear repeated (number of repetition is the number of frames in the tif file)
        file_counts = Counter(file_names_for_stream)

        imaging_extractors = []
        for file_name, num_frames in file_counts.items():
            extractor = _BrukerTiffSinglePlaneImagingExtractor(file_path=str(Path(folder_path) / file_name))
            extractor._num_frames = num_frames
            extractor._image_size = (self._height, self._width)
            extractor._dtype = self._dtype
            imaging_extractors.append(extractor)

        self.xml_metadata = self._get_xml_metadata()

        super().__init__(imaging_extractors=imaging_extractors)

    def _get_xml_metadata(self) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """Parse the metadata in the root element that are under "PVStateValue" tag into a dictionary.

        Returns
        -------
        xml_metadata: dict
            The dictionary of metadata extracted from the XML file.
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

    def _check_consistency_between_imaging_extractors(self):
        """Override the parent class method as none of the properties that are checked are from the sub-imaging extractors."""
        return True

    def get_image_size(self) -> Tuple[int, int]:
        return self._height, self._width

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_channel_names(self) -> List[str]:
        return self._channel_names

    def get_num_channels(self) -> int:
        return 1

    def get_dtype(self) -> DtypeType:
        return self._dtype


class _BrukerTiffSinglePlaneImagingExtractor(ImagingExtractor):
    """A private ImagingExtractor for TIFF files produced by Bruker with only 1 plane.

    The private imaging extractor for OME-TIF image format produced by Bruker,
    which defines the get_video() method to return the requested frames from a given file.
    This extractor is not meant to be used as a standalone ImagingExtractor.
    """

    extractor_name = "_BrukerTiffSinglePlaneImaging"
    is_writable = True
    mode = "file"

    SAMPLING_FREQ_ERROR = "The {}Extractor does not support retrieving the imaging rate."
    CHANNEL_NAMES_ERROR = "The {}Extractor does not support retrieving the name of the channels."
    DATA_TYPE_ERROR = "The {}Extractor does not support retrieving the data type."

    def __init__(self, file_path: PathType):
        """Create a _BrukerTiffSinglePlaneImagingExtractor instance from a TIFF image file (.ome.tif).

        Parameters
        ----------
        file_path : PathType
            The path to the TIF image file (.ome.tif)
        """
        self.tifffile = _get_tiff_reader()
        self.file_path = file_path

        super().__init__()

        self._num_frames = None
        self._image_size = None
        self._dtype = None

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
        with self.tifffile.TiffFile(self.file_path, _multifile=False) as tif:
            pages = tif.pages

            if start_frame is not None and end_frame is not None and start_frame == end_frame:
                return pages[start_frame].asarray()

            end_frame = end_frame or self.get_num_frames()
            start_frame = start_frame or 0

            image_shape = (end_frame - start_frame, *self.get_image_size())
            video = np.zeros(shape=image_shape, dtype=self._dtype)
            for page_ind, page in enumerate(islice(pages, start_frame, end_frame)):
                video[page_ind] = page.asarray()

        return video
