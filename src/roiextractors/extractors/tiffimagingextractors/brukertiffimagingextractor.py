"""ImagingExtractors for the TIFF image format produced by Bruker.

Classes
-------
BrukerTiffImagingExtractor
    Unified extractor for Bruker OME-TIFF files. Inherits from MultiTIFFMultiPageExtractor.
BrukerTiffSinglePlaneImagingExtractor
    Deprecated. Use BrukerTiffImagingExtractor instead.
BrukerTiffMultiPlaneImagingExtractor
    Deprecated. Use BrukerTiffImagingExtractor instead.
"""

import logging
import re
import warnings
from collections import Counter
from itertools import islice
from pathlib import Path
from types import ModuleType
from xml.etree import ElementTree

import numpy as np
from lxml import etree

from .multitiffmultipageextractor import MultiTIFFMultiPageExtractor
from ...extraction_tools import (
    PathType,
    calculate_regular_series_rate,
    calculate_segmented_series_rate,
    get_package,
)
from ...imagingextractor import ImagingExtractor
from ...multiimagingextractor import MultiImagingExtractor


def filter_read_uic_tag_warnings(record):
    """Filter out the warnings from tifffile.read_uic_tag() that are not relevant to the user."""
    return not record.msg.startswith("<tifffile.read_uic_tag>")


logging.getLogger("tifffile.tifffile").addFilter(filter_read_uic_tag_warnings)


def _get_tiff_reader() -> ModuleType:
    """Return the tifffile module."""
    return get_package(package_name="tifffile", installation_instructions="pip install tifffile")


def _determine_frame_rate(element: etree.Element, file_names: list[str] | None = None) -> float | None:
    """Determine the frame rate from the difference in relative timestamps of the frame elements."""
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
        "TSeries Brightness Over Time Element": False,  # XYT
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


class BrukerTiffImagingExtractor(MultiTIFFMultiPageExtractor):
    """An extractor for Bruker Prairie View OME-TIFF files.

    Inherits from MultiTIFFMultiPageExtractor and reads structural metadata and the
    sampling frequency from the Bruker configuration XML. Supports single-plane,
    volumetric, and multi-channel data.

    The user provides a folder path containing .ome.tif files and a Bruker configuration
    XML. The extractor reads structural metadata (dimensions, channels, planes, file layout)
    from that XML, which is authoritative across all PrairieView versions, and computes
    sampling frequency from its relativeTime attributes.

    Parameters
    ----------
    folder_path : str or Path
        Path to the folder containing Bruker .ome.tif files and the configuration XML.
    channel_name : str or None, optional
        Name of the channel to extract. Required when the data has more than one channel.
        Channel names come from the Bruker XML's ``<File channelName="...">`` attribute.
    """

    extractor_name = "BrukerTiffImagingExtractor"

    def __init__(self, folder_path: PathType, channel_name: str | None = None):
        folder_path = Path(folder_path)

        ome_files = sorted(folder_path.glob("*.ome.tif"))
        if not ome_files:
            tif_files = list(folder_path.glob("*.tif"))
            if tif_files:
                raise ValueError(
                    f"Found plain .tif files but no .ome.tif files in '{folder_path}'. "
                    "This looks like Prairie View 5.0 or earlier data, which does not embed OME-XML metadata. "
                    "No Bruker extractor in roiextractors supports pre-5.1 data."
                )
            raise FileNotFoundError(f"No .ome.tif files found in '{folder_path}'.")

        xml_file_path = folder_path / f"{folder_path.name}.xml"
        if not xml_file_path.is_file():
            raise FileNotFoundError(f"Bruker XML configuration file not found at '{xml_file_path}'.")
        self._xml_root = etree.parse(xml_file_path).getroot()
        self._bruker_xml_metadata = self._parse_bruker_xml_metadata()

        file_positions = self._fetch_filenames_from_bruker_xml()
        # Order files into the CZT layout MultiTIFFMultiPageExtractor expects: channels
        # fastest within a frame, then frames in acquisition order. Sorting by
        # (frame_index, channel) also forces a frame's channels into ascending channel
        # order regardless of how the XML lists its <File> elements.
        ordered_filenames = sorted(file_positions, key=file_positions.get)
        file_paths = [folder_path / file_name for file_name in ordered_filenames]
        num_channels = len(self._get_channel_names())
        num_planes = self._determine_num_planes()

        if num_channels > 1 and num_planes > 1:
            warnings.warn(
                "Multi-channel volumetric Bruker data is not tested due to lack of sample data. "
                "Use with care. If you have this type of data and detect errors please open an issue at "
                "https://github.com/catalystneuro/roiextractors/issues \n"
                "We welcome sample data for improving our test coverage and ensuring correctness.",
                stacklevel=2,
            )

        # Pre-set _num_planes so get_native_timestamps() can be called before init.
        # MultiTIFFMultiPageExtractor.__init__() will set it again to the same value.
        self._num_planes = num_planes
        # Derive the rate from the true per-frame timeline. Burst/cycle recordings (e.g. Brightness
        # Over Time) are not uniformly sampled: their timeline is regular within each burst but has
        # large gaps between bursts, so calculate_segmented_series_rate splits at those gaps and
        # reports the within-burst rate. Uniformly-sampled recordings come back as a single segment.
        timestamps = self.get_native_timestamps()
        sampling_frequency, num_segments = calculate_segmented_series_rate(timestamps)
        if sampling_frequency is None:
            raise ValueError("Could not determine sampling frequency from Bruker configuration XML.")
        if num_segments > 1:
            warnings.warn(
                "This Bruker recording has multiple <Sequence> blocks (bursts/cycles) and is not "
                "uniformly sampled. Reporting the within-burst frame rate as sampling_frequency; "
                "use get_timestamps() for the true (gapped) per-frame timeline.",
                stacklevel=2,
            )

        super().__init__(
            file_paths=file_paths,
            sampling_frequency=sampling_frequency,
            dimension_order=self._get_dimension_order(),
            num_channels=num_channels,
            channel_name=channel_name,
            num_planes=num_planes,
        )

        self.set_times(timestamps)

    def _fetch_filenames_from_bruker_xml(self) -> dict[str, tuple[int, int]]:
        """Map each ``.ome.tif`` filename to its ``(frame_index, channel)`` position.

        Walks ``<Frame>``/``<File>`` once over ``self._xml_root``, recording for each
        file's first appearance the index of its ``<Frame>`` in document order and its
        channel. Reports positions only; the caller decides the file ordering.
        """
        file_positions: dict[str, tuple[int, int]] = {}
        frame_index = -1
        for elem in self._xml_root.iter("Frame", "File"):
            if elem.tag == "Frame":
                frame_index += 1
            else:  # File; the channel is encoded in the filename, so it is constant per file
                filename = elem.attrib["filename"]
                if filename not in file_positions:  # keep the first (earliest) appearance
                    file_positions[filename] = (frame_index, int(elem.attrib["channel"]))

        if not file_positions:
            raise ValueError("No <File> elements found in the Bruker configuration XML.")

        return file_positions

    def _determine_is_volumetric(self) -> bool:
        """Return whether the recording is volumetric, from the first ``<Sequence type="...">``."""
        is_series_type_volumetric = {
            "TSeries ZSeries Element": True,  # XYZT
            "TSeries Timed Element": False,  # XYT
            "ZSeries": True,  # ZT (not a time series)
            "Single": False,  # Single image (not a time series)
            "BrightnessOverTime": False,  # XYT (not a volumetric series)
            "TSeries Brightness Over Time Element": False,  # XYT
        }
        first_sequence = next(self._xml_root.iter("Sequence"), None)
        if first_sequence is None:
            raise ValueError("No <Sequence> elements found in the Bruker configuration XML.")
        series_type = first_sequence.attrib.get("type")
        if series_type not in is_series_type_volumetric:
            raise ValueError(
                f"Unknown series type: {series_type}, please raise an issue in the roiextractor repository"
            )
        return is_series_type_volumetric[series_type]

    def _determine_num_planes(self) -> int:
        """Return the number of depth planes from the Bruker XML.

        Planar recordings have one plane. Volumetric recordings store each volume as one
        ``<Sequence>`` whose ``<Frame>`` children are the Z-planes, so the plane count is
        the number of frames in the first sequence.
        """
        if not self._determine_is_volumetric():
            return 1
        first_sequence = next(self._xml_root.iter("Sequence"))
        return sum(1 for _ in first_sequence.iter("Frame"))

    def _get_dimension_order(self) -> str:
        """Return the recording's physical dimension order, ``"CZT"`` or ``"TZC"``.

        Bruker writes multi-channel data in one of two physical layouts. Per-frame: one
        single-page ``.ome.tif`` per (timepoint, channel), so channels are interleaved and
        vary fastest ("CZT"). Per-channel: one multi-page file per channel with page =
        timepoint, so channel is the slowest dimension and time the fastest ("TZC"); this is
        what dual-color single-plane recordings use. We detect the per-channel case by the
        files being multi-page (a filename referenced by more than one ``<File>`` element).
        Single-channel recordings are unaffected: the channel dimension is trivial and "CZT"
        reduces to the correct planar/volumetric order.
        """
        files = list(self._xml_root.iter("File"))
        num_channels = len({elem.attrib["channelName"] for elem in files})
        distinct_filenames = len({elem.attrib["filename"] for elem in files})
        files_are_multipage = distinct_filenames < len(files)
        if num_channels > 1 and files_are_multipage:
            return "TZC"
        return "CZT"

    def _get_channel_names(self) -> list[str]:
        """Return channel labels in acquisition order, from the Bruker XML's ``<File>`` attributes.

        Overrides ``MultiTIFFMultiPageExtractor._get_channel_names`` to read from Bruker's
        configuration XML instead of OME-XML. PrairieView lets users set custom fluorophore
        labels (e.g. ``"Green"``, ``"Red"``) which the Bruker XML carries via ``channelName``
        but OME-XML's generic ``<Channel Name="Ch1"/>`` does not.

        Order matters: the base resolves a user's ``channel_name`` to a positional index and
        assigns each plane its channel by acquisition position, so the names must be returned
        in acquisition order (by the ``<File channel="...">`` number), not alphabetically.
        Sorting alphabetically would swap channels whenever the labels do not sort in
        acquisition order (e.g. channel 1 = "Red", channel 2 = "Green").
        """
        channel_number_to_name = {
            int(elem.attrib["channel"]): elem.attrib["channelName"] for elem in self._xml_root.iter("File")
        }
        return [channel_number_to_name[number] for number in sorted(channel_number_to_name)]

    def get_native_timestamps(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
        """Extract per-sample timestamps from Frame relativeTime attributes in the Bruker XML.

        For volumetric data, frames alternate between planes; every num_planes-th timestamp
        corresponds to one volume (sample).
        """
        sequences = self._xml_root.findall("Sequence")
        # `relativeTime` resets to 0 at the start of each <Sequence> only for burst/cycle
        # recordings (Brightness Over Time, multi-cycle timed series). Volumetric recordings
        # keep a continuous `relativeTime` across their per-volume sequences, and single-sequence
        # recordings have nothing to offset. Detect the reset case and, only then, offset each
        # burst by its real start (from `absoluteTime`) to recover a monotonic global timeline.
        resets = len(sequences) > 1 and float(sequences[1].findall("Frame")[0].attrib["relativeTime"]) == 0.0
        try:
            if resets:
                seq0_start = float(sequences[0].findall("Frame")[0].attrib["absoluteTime"])
                all_times = []
                for sequence in sequences:
                    frames = sequence.findall("Frame")
                    offset = float(frames[0].attrib["absoluteTime"]) - seq0_start
                    all_times.extend(float(frame.attrib["relativeTime"]) + offset for frame in frames)
                all_times = np.array(all_times)
            else:
                all_times = np.array(
                    [float(frame.attrib["relativeTime"]) for frame in self._xml_root.xpath(".//Frame")]
                )
        except KeyError:
            raise ValueError("One or more Frame elements are missing the 'relativeTime'/'absoluteTime' attribute.")
        if len(all_times) == 0:
            raise ValueError("No Frame elements found in the Bruker configuration XML.")
        timestamps = all_times[:: self._num_planes]
        start_sample = start_sample or 0
        end_sample = end_sample or len(timestamps)
        return timestamps[start_sample:end_sample]

    def _parse_bruker_xml_metadata(self) -> dict[str, str | list[dict[str, str]]]:
        """Parse PVStateValue elements from the Bruker configuration XML into a dictionary.

        Returns
        -------
        dict
            Metadata dictionary with keys from PVStateValue elements.
        """
        xml_metadata = dict()
        xml_metadata.update(self._xml_root.attrib)

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
        stream_name: str | None = None,
    ):
        """Create a BrukerTiffMultiPlaneImagingExtractor instance from a folder path that contains the image files.

        .. deprecated::
            Use :class:`BrukerTiffImagingExtractor` instead.

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
        warnings.warn(
            "BrukerTiffMultiPlaneImagingExtractor is deprecated and will be removed in October 2026 or after. "
            "Use BrukerTiffImagingExtractor instead.",
            FutureWarning,
            stacklevel=2,
        )
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

    def get_image_shape(self) -> tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._image_size[0], self._image_size[1]

    # TODO: fix this method so that it is consistent with base multiimagingextractor method (i.e. num_rows, num_columns)
    def get_num_samples(self) -> int:
        return self._imaging_extractors[0].get_num_samples()

    def get_sampling_frequency(self) -> float:
        return self._imaging_extractors[0].get_sampling_frequency() * self._num_planes_per_channel_stream

    def get_series(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
        start = start_sample if start_sample is not None else 0
        stop = end_sample if end_sample is not None else self.get_num_samples()

        series_shape = (stop - start,) + self.get_sample_shape()
        series = np.empty(shape=series_shape, dtype=self.get_dtype())

        for plane_ind, extractor in enumerate(self._imaging_extractors):
            series[..., plane_ind] = extractor.get_series(start_sample=start, end_sample=stop)

        return series

    def get_num_planes(self) -> int:
        """Get the number of depth planes.

        Returns
        -------
        num_planes: int
            The number of depth planes.
        """
        return self._num_planes_per_channel_stream

    def get_volume_shape(self) -> tuple[int, int, int]:
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

    def __init__(self, folder_path: PathType, stream_name: str | None = None):
        """Create a BrukerTiffSinglePlaneImagingExtractor instance from a folder path that contains the image files.

        .. deprecated::
            Use :class:`BrukerTiffImagingExtractor` instead.

        Parameters
        ----------
        folder_path : PathType
            The path to the folder that contains the Bruker TIF image files (.ome.tif) and configuration files (.xml, .env).
        stream_name: str, optional
            The name of the recording channel (e.g. "Ch2" or "Green").
        """
        warnings.warn(
            "BrukerTiffSinglePlaneImagingExtractor is deprecated and will be removed in October 2026 or after. "
            "Use BrukerTiffImagingExtractor instead.",
            FutureWarning,
            stacklevel=2,
        )
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

    def _get_xml_metadata(self) -> dict[str, str | list[dict[str, str]]]:
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

    def get_image_shape(self) -> tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._height, self._width

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_dtype(self) -> np.dtype:
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

    def get_image_shape(self) -> tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._image_size

    def get_sampling_frequency(self):
        raise NotImplementedError(self.SAMPLING_FREQ_ERROR.format(self.extractor_name))

    def get_dtype(self):
        raise NotImplementedError(self.DATA_TYPE_ERROR.format(self.extractor_name))

    def get_series(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
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

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        # Bruker TIFF data does not have native timestamps in the TIFF files themselves
        # The timestamps are in the XML configuration files which are handled by the parent extractors
        return None
