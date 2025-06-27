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
import warnings
from collections import Counter
from itertools import islice
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree

import numpy as np
from lxml import etree

from ...extraction_tools import ArrayType, DtypeType, PathType, get_package
from ...imagingextractor import ImagingExtractor
from ...multiimagingextractor import MultiImagingExtractor


def filter_read_uic_tag_warnings(record):
    """Filter out the warnings from tifffile.read_uic_tag() that are not relevant to the user."""
    return not record.msg.startswith("<tifffile.read_uic_tag>")


logging.getLogger("tifffile.tifffile").addFilter(filter_read_uic_tag_warnings)


def _get_tiff_reader() -> ModuleType:
    """Return the tifffile module."""
    return get_package(package_name="tifffile", installation_instructions="pip install tifffile")


def _determine_frame_rate(element: etree.Element, file_names: Optional[List[str]] = None) -> Union[float, None]:
    """Determine the frame rate from the difference in relative timestamps of the frame elements."""
    from neuroconv.utils import calculate_regular_series_rate

    # Use a single XPath expression if file_names are provided
    if file_names:
        file_names_set = set(file_names)
        frame_elements = element.xpath(".//Frame[File/@filename]")
        filtered_frame_elements = []
        for frame in frame_elements:
            for file in frame.xpath("File"):
                if file.attrib.get("filename") in file_names_set:
                    filtered_frame_elements.append(frame)
                    break
        frame_elements = filtered_frame_elements
    else:
        frame_elements = element.xpath(".//Frame")

    # Extract relativeTime attributes and convert to float
    try:
        relative_times = [float(frame.attrib["relativeTime"]) for frame in frame_elements]
    except KeyError:
        raise ValueError("One or more Frame elements are missing the 'relativeTime' attribute.")
    except ValueError:
        raise ValueError("One or more 'relativeTime' attributes cannot be converted to float.")

    # Calculate frame rate
    frame_rate = calculate_regular_series_rate(np.array(relative_times)) if relative_times else None

    return frame_rate


def _determine_imaging_is_volumetric(folder_path: PathType) -> bool:
    """Determine whether imaging is volumetric.

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

    is_series_type_volumetric = {
        "TSeries ZSeries Element": True,  # XYZT
        "TSeries Timed Element": False,  # XYT
        "ZSeries": True,  # ZT (not a time series)
        "Single": False,  # Single image (not a time series)
        "BrightnessOverTime": False,  # XYT (not a volumetric series)
    }

    is_volumetric = False
    for event, elem in etree.iterparse(xml_file_path, events=("start",)):
        if elem.tag == "Sequence":
            series_type = elem.attrib.get("type")
            if series_type in is_series_type_volumetric:
                is_volumetric = is_series_type_volumetric[series_type]
                break
            else:
                raise ValueError(
                    f"Unknown series type: {series_type}, please raise an issue in the roiextractor repository"
                )

    return is_volumetric


def _parse_xml(folder_path: PathType) -> etree.Element:
    """Parse the XML configuration file into element tree and returns the root Element."""
    folder_path = Path(folder_path)
    xml_file_path = folder_path / f"{folder_path.name}.xml"
    assert xml_file_path.is_file(), f"The XML configuration file is not found at '{folder_path}'."
    tree = etree.parse(str(xml_file_path))
    return tree.getroot()


class BrukerTiffMultiPlaneImagingExtractor(MultiImagingExtractor):
    """A MultiImagingExtractor for TIFF files produced by Bruke with multiple planes.

    This format consists of multiple TIF image files (.ome.tif) and configuration files (.xml, .env).
    """

    extractor_name = "BrukerTiffMultiPlaneImaging"
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

        self._num_samples = self._imaging_extractors[0].get_num_samples()
        self._image_size = *self._imaging_extractors[0].get_frame_shape(), self._num_planes_per_channel_stream
        self.xml_metadata = self._imaging_extractors[0].xml_metadata

        self._start_frames = [0] * self._num_planes_per_channel_stream
        self._end_frames = [self._num_samples] * self._num_planes_per_channel_stream
        self.is_volumetric = True

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._image_size[0], self._image_size[1]

    # TODO: fix this method so that it is consistent with base multiimagingextractor method (i.e. num_rows, num_columns)
    def get_image_size(self) -> Tuple[int, int, int]:
        import warnings

        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._image_size

    def get_num_samples(self) -> int:
        return self._imaging_extractors[0].get_num_samples()

    def get_num_frames(self) -> int:
        """Get the number of frames in the video.

        Returns
        -------
        num_frames: int
            Number of frames in the video.

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_num_samples() instead.
        """
        warnings.warn(
            "get_num_frames() is deprecated and will be removed in or after September 2025. "
            "Use get_num_samples() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_num_samples()

    def get_sampling_frequency(self) -> float:
        return self._imaging_extractors[0].get_sampling_frequency() * self._num_planes_per_channel_stream

    def get_frames(self, frame_idxs: ArrayType, channel: Optional[int] = 0) -> np.ndarray:
        """Get specific frames from the video.

        Parameters
        ----------
        frame_idxs: ArrayType
            The indices of the frames to get.
        channel: int, optional
            Channel index. Deprecated: This parameter will be removed in August 2025.

        Returns
        -------
        frames: numpy.ndarray
            The requested frames.
        """
        if channel != 0:
            from warnings import warn

            warn(
                "The 'channel' parameter in get_frames() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(frame_idxs, (int, np.integer)):
            frame_idxs = [frame_idxs]
        frame_idxs = np.array(frame_idxs)
        assert np.all(frame_idxs < self.get_num_frames()), "'frame_idxs' exceed number of frames"

        frames_shape = (len(frame_idxs),) + self.get_image_size()
        frames = np.empty(shape=frames_shape, dtype=self.get_dtype())

        for plane_ind, extractor in enumerate(self._imaging_extractors):
            frames[..., plane_ind] = extractor.get_frames(frame_idxs)

        return frames

    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
        start = start_sample if start_sample is not None else 0
        stop = end_sample if end_sample is not None else self.get_num_samples()

        series_shape = (stop - start,) + self.get_image_size()
        series = np.empty(shape=series_shape, dtype=self.get_dtype())

        for plane_ind, extractor in enumerate(self._imaging_extractors):
            series[..., plane_ind] = extractor.get_series(start_sample=start, end_sample=stop)

        return series

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        """Get a chunk of video.

        Parameters
        ----------
        start_frame: int, optional
            Start frame index (inclusive).
        end_frame: int, optional
            End frame index (exclusive).
        channel: int, optional
            Channel index. Deprecated: This parameter will be removed in August 2025.

        Returns
        -------
        video: numpy.ndarray
            The video chunk.

        Raises
        ------
        NotImplementedError
            If channel is not 0, as multiple channels are not yet supported.

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_series() instead.
        """
        warnings.warn(
            "get_video() is deprecated and will be removed in or after September 2025. " "Use get_series() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if channel != 0:
            warnings.warn(
                "The 'channel' parameter in get_video() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
            raise NotImplementedError(
                f"MultiImagingExtractors for multiple channels have not yet been implemented! (Received '{channel}'."
            )

        return self.get_series(start_sample=start_frame, end_sample=end_frame)

    def get_num_planes(self) -> int:
        """Get the number of depth planes.

        Returns
        -------
        num_planes: int
            The number of depth planes.
        """
        return self._num_planes_per_channel_stream

    def get_volume_shape(self) -> Tuple[int, int, int]:
        """Get the shape of the volumetric video (num_rows, num_columns, num_planes).

        Returns
        -------
        video_shape: tuple
            Shape of the volumetric video (num_rows, num_columns, num_planes).
        """
        return (self._image_size[0], self._image_size[1], self.get_num_planes())


class BrukerTiffSinglePlaneImagingExtractor(MultiImagingExtractor):
    """A MultiImagingExtractor for TIFF files produced by Bruker with only 1 plane."""

    extractor_name = "BrukerTiffSinglePlaneImaging"
    mode = "folder"

    @classmethod
    def get_streams(cls, folder_path: PathType) -> dict:
        """
        Get the available streams from the Bruker TIF image files (.ome.tif) and configuration files (.xml, .env).

        Parameters
        ----------
        folder_path : PathType
            The path to the folder that contains the Bruker TIF image files (.ome.tif) and configuration files (.xml, .env).

        Returns
        -------
        streams: dict
            The dictionary of available streams.
        """
        channel_names = cls.get_available_channels(folder_path=folder_path)

        channel_names = cls.get_available_channels(folder_path=folder_path)

        natsort = get_package(package_name="natsort", installation_instructions="pip install natsort")
        unique_channel_names = natsort.natsorted(channel_names)
        unique_channel_names = natsort.natsorted(channel_names)
        streams = dict(channel_streams=unique_channel_names)
        return streams

    @staticmethod
    def get_available_channels(folder_path: PathType) -> set[str]:
        """
        Extract set of available channel names from the XML configuration file in the specified folder.

        Parameters
        ----------
        folder_path : PathType
            The path to the folder containing the XML configuration file. It can be either a string
            or a Path object.

        Returns
        -------
        Set[str]
            A set of channel names available in the first 'Frame' element found in the XML configuration file.
        """
        folder_path = Path(folder_path)
        xml_file_path = folder_path / f"{folder_path.name}.xml"
        assert xml_file_path.is_file(), f"The XML configuration file is not found at '{folder_path}'."

        channel_names = set()
        for event, elem in etree.iterparse(xml_file_path, events=("start",)):
            if elem.tag == "Frame":
                # Get all the sub-elements in this Frame element
                for subelem in elem:
                    if subelem.tag == "File":
                        channel_names.add(subelem.attrib["channelName"])

                break  # Exit after processing the first "Frame" element

        return channel_names

    def __init__(self, folder_path: PathType, stream_name: Optional[str] = None):
        """Create a BrukerTiffSinglePlaneImagingExtractor instance from a folder path that contains the image files.

        Parameters
        ----------
        folder_path : PathType
            The path to the folder that contains the Bruker TIF image files (.ome.tif) and configuration files (.xml, .env).
        stream_name: str, optional
            The name of the recording channel (e.g. "Ch2" or "Green").
        """
        self._tifffile = _get_tiff_reader()

        folder_path = Path(folder_path)
        tif_file_paths = list(folder_path.glob("*.ome.tif"))
        assert tif_file_paths, f"The TIF image files are missing from '{folder_path}'."

        streams = self.get_streams(folder_path=folder_path)
        channel_streams = streams["channel_streams"]
        channel_streams = streams["channel_streams"]
        if stream_name is None:
            if len(channel_streams) > 1:
                raise ValueError(
                    "More than one recording stream is detected! Please specify which stream you wish to load with the `stream_name` argument. "
                    f"To see what streams are available, call `BrukerTiffSinglePlaneImagingExtractor.get_stream_names(folder_path=...)`."
                )
            stream_name = channel_streams[0]

        self.stream_name = stream_name

        self._xml_root = _parse_xml(folder_path=folder_path)
        file_elements = self._xml_root.findall(".//File")

        # This is the case when stream_name is a channel name (e.g. "Green" or "Ch2")
        if stream_name in channel_streams:
            file_names_for_stream = [
                f.attrib["filename"] for f in file_elements if f.attrib["channelName"] == stream_name
            ]
        else:  # This is the case for when stream_name is a plane_stream
            file_names = [file.attrib["filename"] for file in file_elements]
            file_names_for_stream = [file for file in file_names if self.stream_name in file]
            if file_names_for_stream == []:
                raise ValueError(
                    f"The selected stream '{self.stream_name}' is not in the available channel_streams '{streams['channel_streams']}'!"
                )

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
        for file_name, num_samples in file_counts.items():
            extractor = _BrukerTiffSinglePlaneImagingExtractor(file_path=str(Path(folder_path) / file_name))
            extractor._num_samples = num_samples
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
        xml_metadata.update(self._xml_root.attrib)

        # Use a single XPath to get all PVStateValue elements
        pv_state_values = self._xml_root.xpath(".//PVStateValue")

        for child in pv_state_values:
            metadata_root_key = child.attrib["key"]
            if "value" in child.attrib:
                if metadata_root_key not in xml_metadata:
                    xml_metadata[metadata_root_key] = child.attrib["value"]
            else:
                xml_metadata[metadata_root_key] = []
                for indexed_value in child:
                    if "description" in indexed_value.attrib:
                        xml_metadata[metadata_root_key].append(
                            {indexed_value.attrib["description"]: indexed_value.attrib["value"]}
                        )
                    elif "value" in indexed_value.attrib:
                        xml_metadata[metadata_root_key].append(
                            {indexed_value.attrib["index"]: indexed_value.attrib["value"]}
                        )
                    else:
                        for subindexed_value in indexed_value:
                            if "description" in subindexed_value.attrib:
                                xml_metadata[metadata_root_key].append(
                                    {subindexed_value.attrib["description"]: subindexed_value.attrib["value"]}
                                )
                            else:
                                xml_metadata[metadata_root_key].append(
                                    {indexed_value.attrib["index"]: subindexed_value.attrib["value"]}
                                )

        return xml_metadata

    def _check_consistency_between_imaging_extractors(self):
        """Override the parent class method as none of the properties that are checked are from the sub-imaging extractors."""
        return True

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._height, self._width

    def get_image_size(self) -> Tuple[int, int]:
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
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

        self._num_samples = None
        self._image_size = None
        self._dtype = None

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_num_frames(self) -> int:
        """Get the number of frames in the video.

        Returns
        -------
        num_frames: int
            Number of frames in the video.

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_num_samples() instead.
        """
        warnings.warn(
            "get_num_frames() is deprecated and will be removed in or after September 2025. "
            "Use get_num_samples() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_num_samples()

    def get_num_channels(self) -> int:
        return 1

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._image_size

    def get_image_size(self) -> Tuple[int, int]:
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._image_size

    def get_sampling_frequency(self):
        raise NotImplementedError(self.SAMPLING_FREQ_ERROR.format(self.extractor_name))

    def get_channel_names(self) -> list:
        raise NotImplementedError(self.CHANNEL_NAMES_ERROR.format(self.extractor_name))

    def get_dtype(self):
        raise NotImplementedError(self.DATA_TYPE_ERROR.format(self.extractor_name))

    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
        with self.tifffile.TiffFile(self.file_path, _multifile=False) as tif:
            pages = tif.pages

            if start_sample is not None and end_sample is not None and start_sample == end_sample:
                return pages[start_sample].asarray()

            end_sample = end_sample or self.get_num_samples()
            start_sample = start_sample or 0

            image_shape = (end_sample - start_sample, *self.get_image_shape())
            series = np.zeros(shape=image_shape, dtype=self._dtype)
            for page_ind, page in enumerate(islice(pages, start_sample, end_sample)):
                series[page_ind] = page.asarray()

        return series

    def get_video(
        self, start_frame: Optional[int] = None, end_frame: Optional[int] = None, channel: int = 0
    ) -> np.ndarray:
        """Get the video frames.

        Parameters
        ----------
        start_frame: int, optional
            Start frame index (inclusive).
        end_frame: int, optional
            End frame index (exclusive).
        channel: int, optional
            Channel index. Deprecated: This parameter will be removed in August 2025.

        Returns
        -------
        video: numpy.ndarray
            The video frames.

        Deprecated
        ----------
        This method will be removed in or after September 2025.
        Use get_series() instead.
        """
        warnings.warn(
            "get_video() is deprecated and will be removed in or after September 2025. " "Use get_series() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if channel != 0:
            warnings.warn(
                "The 'channel' parameter in get_video() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.get_series(start_sample=start_frame, end_sample=end_frame)

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # Bruker TIFF data does not have native timestamps in the TIFF files themselves
        # The timestamps are in the XML configuration files which are handled by the parent extractors
        return None
