"""Extractor for OME-TIFF files.

Classes
-------
OMETiffImagingExtractor
    An extractor that reads OME-TIFF files by parsing embedded OME-XML metadata
    and delegating to MultiTIFFMultiPageExtractor.
"""

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .multitiffmultipageextractor import MultiTIFFMultiPageExtractor
from ...extraction_tools import PathType, get_package


class OMETiffImagingExtractor(MultiTIFFMultiPageExtractor):
    """An extractor for OME-TIFF files that automatically parses metadata from embedded OME-XML.

    This extractor reads OME-XML metadata from a TIFF file to determine the dimension order,
    number of channels, number of depth planes, image dimensions, and file references.
    It then delegates all data reading to MultiTIFFMultiPageExtractor.

    For multi-file OME-TIFF datasets, the user provides a single file path. The extractor
    discovers all related files from the TiffData elements in the OME-XML metadata.

    Parameters
    ----------
    file_path : str or Path
        Path to an OME-TIFF file (.ome.tif or .ome.tiff). For multi-file datasets,
        this should be the file containing the full OME-XML metadata (typically the first file).
    sampling_frequency : float or None, optional
        The sampling frequency in Hz. If None, the extractor will attempt to derive it from
        the TimeIncrement attribute on the Pixels element in the OME-XML metadata
        (as 1.0 / TimeIncrement). Per the OME specification (https://www.openmicroscopy.org/Schemas/Documentation/Generated/OME-2016-06/ome.html), TimeIncrement
        stores the global time interval between timepoints in seconds. If neither
        sampling_frequency nor TimeIncrement is available, a ValueError is raised.
        Default is None.
    channel_name : str or None, optional
        Name of the channel to extract (e.g., "0", "1"). Required when the data has
        more than one channel. Default is None.
    """

    extractor_name = "OMETiffImagingExtractor"

    # Conversion factors from OME UnitsTime to seconds. Covers the full enumeration
    # from the OME 2016-06 schema (https://www.openmicroscopy.org/Schemas/Documentation/Generated/OME-2016-06/ome.html).
    _TIME_UNIT_TO_SECONDS = {
        "Ys": 1e24,
        "Zs": 1e21,
        "Es": 1e18,
        "Ps": 1e15,
        "Ts": 1e12,
        "Gs": 1e9,
        "Ms": 1e6,
        "ks": 1e3,
        "hs": 1e2,
        "das": 1e1,
        "s": 1.0,
        "ds": 1e-1,
        "cs": 1e-2,
        "ms": 1e-3,
        "\u00b5s": 1e-6,  # micro sign (U+00B5), used by OME-XML for microseconds
        "ns": 1e-9,
        "ps": 1e-12,
        "fs": 1e-15,
        "as": 1e-18,
        "zs": 1e-21,
        "ys": 1e-24,
        "min": 60.0,
        "h": 3600.0,
        "d": 86400.0,
    }

    def __init__(
        self,
        file_path: PathType,
        sampling_frequency: float | None = None,
        channel_name: str | None = None,
    ):
        metadata = self._parse_ome_metadata(file_path)

        metadata_sampling_frequency = metadata.pop("sampling_frequency", None)
        if sampling_frequency is None:
            if metadata_sampling_frequency is None:
                raise ValueError(
                    "sampling_frequency must be provided when the OME-XML metadata "
                    "does not contain a TimeIncrement attribute on the Pixels element."
                )
            sampling_frequency = metadata_sampling_frequency

        # Store channel names before super().__init__() because the base class
        # calls self._get_channel_names() during init, and Python's MRO dispatches
        # to our override which needs this attribute to be set.
        self._ome_channel_names = metadata.pop("channel_names", None)

        super().__init__(
            file_paths=metadata["file_paths"],
            sampling_frequency=sampling_frequency,
            channel_name=channel_name,
            dimension_order=metadata["dimension_order"],
            num_channels=metadata["num_channels"],
            num_planes=metadata["num_planes"],
        )

    def _get_channel_names(self) -> list[str]:
        """Return channel names from OME-XML Channel/@Name attributes when available."""
        if self._ome_channel_names is not None:
            return self._ome_channel_names
        return super()._get_channel_names()

    @staticmethod
    def get_available_channel_names(file_path: PathType) -> list[str]:
        """Return the available channel names for an OME-TIFF file without constructing an extractor.

        Parameters
        ----------
        file_path : PathType
            Path to an OME-TIFF file.

        Returns
        -------
        list[str]
            Channel names from OME-XML Channel/@Name attributes, or numeric strings
            ("0", "1", ...) if names are not present.
        """
        metadata = OMETiffImagingExtractor._parse_ome_metadata(file_path)
        channel_names = metadata.get("channel_names")
        if channel_names is not None:
            return channel_names
        return [str(i) for i in range(metadata["num_channels"])]

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        """Return per-timepoint timestamps from OME-XML Plane/@DeltaT attributes.

        Parses the OME-XML on demand. Filters to the selected channel and uses Z=0
        planes for volumetric data to get one timestamp per timepoint. Returns None
        if no matching timestamps are found or if the count doesn't match num_samples.
        """
        all_planes = self._parse_ome_native_timestamps(self._file_paths[0])
        if all_planes is None:
            return None

        # Filter planes to get one timestamp per timepoint
        timepoint_to_delta_t: dict[int, float] = {}
        for plane in all_planes:
            if self._num_channels > 1 and plane["the_c"] != self._channel_index:
                continue
            if self.is_volumetric and plane["the_z"] != 0:
                continue
            timepoint_to_delta_t[plane["the_t"]] = plane["delta_t_seconds"]

        if not timepoint_to_delta_t:
            return None

        sorted_timepoints = sorted(timepoint_to_delta_t.keys())
        timestamps = np.array([timepoint_to_delta_t[t] for t in sorted_timepoints])

        if len(timestamps) != self.get_num_samples():
            import warnings

            warnings.warn(
                f"OME-XML contains Plane elements with DeltaT but only {len(timestamps)} "
                f"of {self.get_num_samples()} timepoints have timestamps. "
                f"Native timestamps will not be used. "
                f"Call _parse_ome_native_timestamps() directly to inspect the raw timestamps.",
                stacklevel=2,
            )
            return None

        return timestamps[start_sample:end_sample]

    @staticmethod
    def _parse_ome_metadata(file_path: PathType) -> dict:
        """Parse OME-XML metadata from a TIFF file.

        Extracts structural metadata (dimensions, channels, planes, file references) from the
        OME-XML embedded in a TIFF file. The returned dict contains the parameters needed to
        initialize a MultiTIFFMultiPageExtractor (excluding sampling_frequency and channel_name,
        which are not reliably encoded in OME-XML).

        This method is intended to be called as a utility by vendor-specific extractors that
        produce OME-TIFF files (e.g. Bruker, ThorImage, MicroManager) and need to extract
        structural metadata before initializing MultiTIFFMultiPageExtractor.

        Parameters
        ----------
        file_path : PathType
            Path to an OME-TIFF file.

        Returns
        -------
        dict
            Dictionary with keys: file_paths, dimension_order, num_channels, num_planes,
            and optionally sampling_frequency (float, in Hz) if the Pixels element has
            a TimeIncrement attribute.
        """
        tifffile = get_package(package_name="tifffile")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"OME-TIFF file not found: {file_path}")

        tiff = tifffile.TiffFile(file_path)
        try:
            ome_xml_string = tiff.ome_metadata
            if ome_xml_string is None:
                ome_xml_string = tiff.pages[0].description
            if not ome_xml_string:
                raise ValueError(f"No OME-XML metadata found in {file_path}")
        finally:
            tiff.close()

        # Old OME-TIFF versions wrapped the XML in comments (<!-- ... -->)
        if ome_xml_string.lstrip().startswith("<!--"):
            ome_xml_string = ome_xml_string.replace("<!--", "").replace("-->", "")
        try:
            ome_root = ET.fromstring(ome_xml_string.encode("utf-8"))
        except ValueError:
            ome_root = ET.fromstring(ome_xml_string)
        pixels_element = ome_root.find(".//{*}Pixels")
        if pixels_element is None:
            raise ValueError(f"No Pixels element found in OME-XML metadata of {file_path}")

        ome_dimension_order = pixels_element.get("DimensionOrder")
        if ome_dimension_order is None:
            raise ValueError(f"No DimensionOrder attribute found in OME-XML metadata of {file_path}")

        num_channels = int(pixels_element.get("SizeC", "1"))
        num_planes = int(pixels_element.get("SizeZ", "1"))

        # Extract channel names from Channel/@Name attributes when present
        channel_elements = pixels_element.findall(".//{*}Channel")
        channel_names = None
        if channel_elements:
            names = [ch.get("Name") for ch in channel_elements]
            if all(name is not None for name in names):
                channel_names = names

        # TimeIncrement is a global timing attribute on Pixels (OME spec:
        # https://www.openmicroscopy.org/Schemas/Documentation/Generated/OME-2016-06/ome.html):
        # "used for time series that have a global timing specification instead of
        # per-timepoint timing info". TimeIncrementUnit defaults to seconds ("s").
        time_increment_str = pixels_element.get("TimeIncrement")
        if time_increment_str is not None:
            time_increment = float(time_increment_str)
            unit = pixels_element.get("TimeIncrementUnit", "s")
            time_increment_in_seconds = time_increment * OMETiffImagingExtractor._TIME_UNIT_TO_SECONDS[unit]
            sampling_frequency = 1.0 / time_increment_in_seconds
        else:
            sampling_frequency = None

        # Convert OME dimension order (e.g. "XYCZT") to 3-letter format (e.g. "CZT")
        dimension_order = ome_dimension_order.replace("X", "").replace("Y", "")

        # Discover file paths from TiffData elements
        file_positions = OMETiffImagingExtractor._parse_file_paths_from_ome_metadata(pixels_element, file_path)

        # Sort files to match the dimension order: slowest-varying dimension as primary key
        dimension_to_key = {"C": "first_c", "Z": "first_z", "T": "first_t"}
        sort_keys = [dimension_to_key[d] for d in reversed(dimension_order)]
        file_paths = sorted(
            file_positions.keys(),
            key=lambda fp: tuple(file_positions[fp][k] for k in sort_keys) + (file_positions[fp]["ifd"],),
        )

        result = dict(
            file_paths=file_paths,
            dimension_order=dimension_order,
            num_channels=num_channels,
            num_planes=num_planes,
        )
        if sampling_frequency is not None:
            result["sampling_frequency"] = sampling_frequency
        if channel_names is not None:
            result["channel_names"] = channel_names
        return result

    @staticmethod
    def _parse_file_paths_from_ome_metadata(pixels_element: ET.Element, source_file_path: Path) -> dict[Path, dict]:
        """Discover all TIFF files referenced in the OME-XML TiffData elements.

        Parses TiffData elements to find all referenced files and their earliest
        logical coordinates (FirstC, FirstZ, FirstT, IFD). For single-file datasets
        (no TiffData elements), returns just the source file.

        Parameters
        ----------
        pixels_element : xml.etree.ElementTree.Element
            The Pixels element from the OME-XML.
        source_file_path : Path
            Path to the file that contained the OME-XML metadata. Used to resolve
            relative file references.

        Returns
        -------
        dict[Path, dict]
            Mapping from file path to its earliest logical position, with keys
            first_c, first_z, first_t, ifd.
        """
        tiff_data_elements = pixels_element.findall(".//{*}TiffData")

        if not tiff_data_elements:
            return {source_file_path: dict(first_c=0, first_z=0, first_t=0, ifd=0)}

        folder = source_file_path.parent
        found_files: dict[Path, dict] = {}

        for tiff_data in tiff_data_elements:
            uuid_element = tiff_data.find("{*}UUID")
            if uuid_element is not None:
                file_name = uuid_element.get("FileName")
            else:
                file_name = None

            if file_name is None:
                file_path = source_file_path
            else:
                file_path = folder / file_name

            entry = dict(
                first_c=int(tiff_data.get("FirstC", "0")),
                first_z=int(tiff_data.get("FirstZ", "0")),
                first_t=int(tiff_data.get("FirstT", "0")),
                ifd=int(tiff_data.get("IFD", "0")),
            )

            # Keep the earliest logical position per file
            if file_path not in found_files:
                found_files[file_path] = entry
            else:
                existing = found_files[file_path]
                if (entry["first_t"], entry["first_z"], entry["first_c"], entry["ifd"]) < (
                    existing["first_t"],
                    existing["first_z"],
                    existing["first_c"],
                    existing["ifd"],
                ):
                    found_files[file_path] = entry

        return found_files

    @staticmethod
    def _parse_ome_native_timestamps(file_path: PathType) -> list[dict] | None:
        """Parse DeltaT timestamps from all Plane elements in the OME-XML.

        Returns the raw plane data without any filtering by channel or Z-plane.
        Each entry contains the plane's channel index, Z index, timepoint index,
        and DeltaT value converted to seconds.

        Parameters
        ----------
        file_path : PathType
            Path to an OME-TIFF file containing the OME-XML metadata.

        Returns
        -------
        list[dict] or None
            List of dicts with keys "the_c", "the_z", "the_t", "delta_t_seconds",
            or None if no Plane elements have DeltaT attributes.
        """
        tifffile = get_package(package_name="tifffile")

        tiff = tifffile.TiffFile(file_path)
        try:
            ome_xml_string = tiff.ome_metadata
            if ome_xml_string is None:
                ome_xml_string = tiff.pages[0].description
        finally:
            tiff.close()

        if not ome_xml_string:
            return None

        if ome_xml_string.lstrip().startswith("<!--"):
            ome_xml_string = ome_xml_string.replace("<!--", "").replace("-->", "")
        try:
            ome_root = ET.fromstring(ome_xml_string.encode("utf-8"))
        except ValueError:
            ome_root = ET.fromstring(ome_xml_string)

        pixels_element = ome_root.find(".//{*}Pixels")
        if pixels_element is None:
            return None

        plane_elements = pixels_element.findall(".//{*}Plane")
        if not plane_elements:
            return None

        result = []
        for plane in plane_elements:
            delta_t_str = plane.get("DeltaT")
            if delta_t_str is None:
                continue

            unit = plane.get("DeltaTUnit", "s")
            result.append(
                dict(
                    the_c=int(plane.get("TheC", "0")),
                    the_z=int(plane.get("TheZ", "0")),
                    the_t=int(plane.get("TheT", "0")),
                    delta_t_seconds=float(delta_t_str) * OMETiffImagingExtractor._TIME_UNIT_TO_SECONDS[unit],
                )
            )

        return result if result else None
