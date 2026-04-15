"""Extractor for OME-TIFF files.

Classes
-------
OMETiffImagingExtractor
    An extractor that reads OME-TIFF files by parsing embedded OME-XML metadata
    and delegating to MultiTIFFMultiPageExtractor.
"""

from pathlib import Path
from xml.etree import ElementTree as ET

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
    sampling_frequency : float
        The sampling frequency in Hz.
    channel_name : str or None, optional
        Name of the channel to extract (e.g., "0", "1"). Required when the data has
        more than one channel. Default is None.
    """

    extractor_name = "OMETiffImagingExtractor"

    def __init__(
        self,
        file_path: PathType,
        sampling_frequency: float,
        channel_name: str | None = None,
    ):
        metadata = self._parse_ome_metadata(file_path)

        super().__init__(
            sampling_frequency=sampling_frequency,
            channel_name=channel_name,
            **metadata,
        )

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
            Dictionary with keys: file_paths, dimension_order, num_channels, num_planes.
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

        return dict(
            file_paths=file_paths,
            dimension_order=dimension_order,
            num_channels=num_channels,
            num_planes=num_planes,
        )

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
