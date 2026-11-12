"""An ImagingExtractor for MMStack OME-TIFF files produced by Micro-Manager."""

import logging
import warnings
from datetime import datetime
from functools import cached_property
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .ometiffimagingextractor import OMETiffImagingExtractor
from ..._metadata import Metadata
from ...extraction_tools import PathType, get_package


def filter_tiff_tag_warnings(record):
    """Filter out the warning messages from tifffile package."""
    return not record.msg.startswith("<tifffile.TiffTag 270 @42054>")


logging.getLogger("tifffile.tifffile").addFilter(filter_tiff_tag_warnings)


class MicroManagerTiffImagingExtractor(OMETiffImagingExtractor):
    """Read a selected position and channel from Micro-Manager MMStack OME-TIFF files.

    Parameters
    ----------
    file_path : PathType, optional
        One file in the recording. Related files are discovered from its OME-XML.
    channel_name : str, optional
        Channel to read. Required when the recording has several channels.
    position_index : int, optional
        Position to read, in the order of the OME-XML Image elements. Required when
        the supplied file contains several positions. Otherwise the position is
        selected from the supplied file's name.
    sampling_frequency : float, optional
        Sampling frequency in Hz. By default, the reciprocal of the median interval
        between timestamps of the selected channel's first depth plane. Required
        when fewer than two samples have timestamps.
    folder_path : PathType, optional
        Deprecated alias for a folder containing the MMStack files. Use file_path.
    """

    extractor_name = "MicroManagerTiffImaging"

    def __init__(
        self,
        file_path: PathType | None = None,
        channel_name: str | None = None,
        *,
        position_index: int | None = None,
        sampling_frequency: float | None = None,
        folder_path: PathType | None = None,
    ):
        file_path = self._resolve_file_path(file_path, folder_path)
        tifffile = get_package(package_name="tifffile")
        with tifffile.TiffFile(file_path) as tiff:
            self.micromanager_metadata = tiff.micromanager_metadata
            ome_root = ET.fromstring(tiff.ome_metadata)
        binary_only = ome_root.find("{*}BinaryOnly")
        if binary_only is not None:
            master_path = file_path.parent / binary_only.attrib["MetadataFile"]
            with tifffile.TiffFile(master_path) as tiff:
                ome_root = ET.fromstring(tiff.ome_metadata)

        images = ome_root.findall("{*}Image")
        if position_index is None and len(images) == 1:
            position_index = 0
        if position_index is None:
            matching_positions = [
                index
                for index, image in enumerate(images)
                if any(
                    element.get("FileName") == file_path.name
                    for element in image.findall("{*}Pixels/{*}TiffData/{*}UUID")
                )
            ]
            if len(matching_positions) != 1:
                raise ValueError("position_index must be specified when a file contains several positions.")
            position_index = matching_positions[0]
        if not 0 <= position_index < len(images):
            raise ValueError(f"position_index must be between 0 and {len(images) - 1}.")
        self._ome_root = ome_root
        self._pixels_element = images[position_index].find("{*}Pixels")

        super().__init__(file_path=file_path, sampling_frequency=1.0, channel_name=channel_name)
        if sampling_frequency is None:
            timestamps = self.get_native_timestamps()
            if timestamps is None or len(timestamps) < 2:
                raise ValueError("sampling_frequency must be provided when fewer than two samples have timestamps.")
            sampling_frequency = float(1.0 / np.median(np.diff(timestamps)))
        self._sampling_frequency = sampling_frequency
        self.folder_path = file_path.parent
        self._channel_names = list(self.micromanager_metadata["Summary"]["ChNames"])
        self._kwargs = dict(
            file_path=str(file_path),
            channel_name=channel_name,
            position_index=position_index,
            sampling_frequency=sampling_frequency,
        )

    @staticmethod
    def _resolve_file_path(file_path, folder_path):
        if folder_path is not None:
            if file_path is not None:
                raise ValueError("Pass either file_path or folder_path, not both.")
            file_path = folder_path
        if file_path is None:
            raise ValueError("file_path must be provided.")
        file_path = Path(file_path)
        if folder_path is not None or file_path.is_dir():
            warnings.warn(
                "Passing a folder is deprecated and will be removed on or after May 2027. "
                "Pass an MMStack file as file_path instead.",
                FutureWarning,
                stacklevel=3,
            )
            files = sorted(file_path.glob("*.ome.tif"))
            assert files, f"The TIF image files are missing from '{file_path}'."
            file_path = files[0]
        return file_path

    def _parse_ome_metadata(self, file_path):
        pixels = self._pixels_element
        file_positions = self._parse_file_paths_from_ome_metadata(pixels, file_path)
        self._micromanager_file_paths = list(file_positions)
        missing_files = [path.name for path in self._micromanager_file_paths if not path.is_file()]
        assert not missing_files, (
            f"Some of the TIF image files at '{file_path.parent}' are missing. "
            f"The list of files that are missing: {missing_files}"
        )
        dimension_order = pixels.attrib["DimensionOrder"].replace("X", "").replace("Y", "")
        sizes = {dimension: int(pixels.attrib[f"Size{dimension}"]) for dimension in "CZT"}
        divisors = {}
        divisor = 1
        for dimension in dimension_order:
            divisors[dimension] = divisor
            divisor *= sizes[dimension]
        mapping = []
        for element in pixels.findall("{*}TiffData"):
            uuid = element.find("{*}UUID")
            path = file_path if uuid is None else file_path.parent / uuid.attrib["FileName"]
            file_index = self._micromanager_file_paths.index(path)
            first = sum(int(element.get(f"First{dimension}", "0")) * divisors[dimension] for dimension in "CZT")
            for offset in range(int(element.get("PlaneCount", "1"))):
                coordinates = {
                    dimension: ((first + offset) // divisors[dimension]) % sizes[dimension] for dimension in "CZT"
                }
                mapping.append(
                    (
                        file_index,
                        int(element.get("IFD", "0")) + offset,
                        coordinates["C"],
                        coordinates["Z"],
                        coordinates["T"],
                    )
                )
        self._micromanager_frame_mapping = np.array(
            mapping,
            dtype=[
                (name, np.uint32) for name in ("file_index", "IFD_index", "channel_index", "depth_index", "time_index")
            ],
        )
        physical_frames = self._micromanager_frame_mapping[["file_index", "IFD_index"]]
        if len(np.unique(physical_frames)) != len(physical_frames):
            raise ValueError("The OME-XML contains repeated TIFF page references for different frames.")
        return dict(
            file_paths=self._micromanager_file_paths,
            dimension_order=dimension_order,
            num_channels=sizes["C"],
            num_planes=sizes["Z"],
            channel_names=[channel.attrib["Name"] for channel in pixels.findall("{*}Channel")],
        )

    def _create_frame_to_ifd_table(self, dimension_order, num_channels, num_planes, ifds_per_file):
        return self._micromanager_frame_mapping

    def _get_ifds_per_file(self):
        return [
            int(np.count_nonzero(self._micromanager_frame_mapping["file_index"] == index))
            for index in range(len(self._file_paths))
        ]

    def _get_ome_image_metadata(self):
        return self._ome_root, self._pixels_element

    @cached_property
    def _unified_metadata(self) -> Metadata:
        if self._metadata_parent is not None:
            return super()._unified_metadata
        metadata = super()._unified_metadata
        summary = self.micromanager_metadata["Summary"]
        if summary.get("StartTime"):
            date, time, offset = summary["StartTime"].split()
            metadata.acquisition_start_datetime = datetime.fromisoformat(f"{date}T{time}{offset[:3]}:{offset[3:]}")
        metadata.acquisition_software_name = "Micro-Manager"
        metadata.acquisition_software_version = summary.get("MicroManagerVersion") or None
        return metadata

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        depth_index = self._frames_to_ifd_table[0]["depth_index"]
        time_by_frame = {
            int(plane.attrib["TheT"]): float(plane.attrib["DeltaT"])
            * self._TIME_UNIT_TO_SECONDS[plane.get("DeltaTUnit", "s")]
            for plane in self._pixels_element.findall("{*}Plane")
            if int(plane.attrib["TheC"]) == self._channel_index
            and int(plane.attrib["TheZ"]) == depth_index
            and "DeltaT" in plane.attrib
        }
        time_indices = self._frames_to_ifd_table["time_index"][:: self._num_planes]
        if any(time_index not in time_by_frame for time_index in time_indices):
            return None
        timestamps = np.array([time_by_frame[time_index] for time_index in time_indices])
        return timestamps[start_sample:end_sample]

    @staticmethod
    def get_available_channel_names(
        file_path: PathType | None = None, *, folder_path: PathType | None = None
    ) -> list[str]:
        file_path = MicroManagerTiffImagingExtractor._resolve_file_path(file_path, folder_path)
        tifffile = get_package(package_name="tifffile")
        with tifffile.TiffFile(file_path) as tiff:
            return list(tiff.micromanager_metadata["Summary"]["ChNames"])
