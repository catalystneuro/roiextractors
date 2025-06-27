"""Extractor for Thor TIFF files."""

import math
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...imagingextractor import ImagingExtractor


class ThorTiffImagingExtractor(ImagingExtractor):
    """
    An ImagingExtractor for TIFF files exported files exported with the `ThorImage` software.

    Note that is possible that the data is acquired with a Thor microscope but not with the ThorImage software and
    this extractor will not work in that case.

    The ThorImage software exports TIFF files following (loosely) the OME-TIFF standard. Plus, there is a file
    `Experiment.xml` that contains metadata about the acquisition. This extractor reads the TIFF file and the
    `Experiment.xml` file to extract the image data and metadata.

    This extractor builds a mapping between the T (time) dimension and the corresponding
    pages/IFDs of the TIFF files using a named tuple structure:

    For each time frame (T), we record a list of PageMapping objects that correspond to the
    pages of the TIFF file that contain the image data for that frame.

    Each PageMapping object contains:
      - page_index: The index of the page in the TIFF file (which holds the complete X and Y image data),
      - channel_index: The coordinate along the channel (C) axis (or None if absent),
      - depth_index: The coordinate along the depth (Z) axis (or None if absent).

    When get_frames() is called, the mapping is used to load only the pages for the requested
    frames into a preallocated NumPy array.

    Note: According to the OME specification (see
    https://www.openmicroscopy.org/Schemas/Documentation/Generated/OME-2016-06/ome_xsd.html#Pixels_DimensionOrder),
    the spatial dimensions (X and Y) are always stored on a single page.
    """

    extractor_name = "ThorTiffImaging"

    # Named tuple to hold page mapping details.
    PageMapping = namedtuple("PageMapping", ["page_index", "channel_index", "depth_index"])

    def __init__(self, file_path: Union[str, Path], channel_name: Optional[str] = None):
        """Create a ThorTiffImagingExtractor instance from a TIFF file."""
        super().__init__()
        self.file_path = Path(file_path)
        self.folder_path = self.file_path.parent
        self.channel_name = channel_name

        # Load Experiment.xml metadata if available.
        self._parse_experiment_xml()

        # Open the TIFF file to extract OME metadata and series information.
        # Keep the file reference open instead of using a context manager
        import tifffile

        self._tiff_reader = tifffile.TiffFile(self.file_path)
        self._ome_metadata = self._tiff_reader.ome_metadata
        ome_root = self._parse_ome_metadata(self._ome_metadata)
        pixels_element = ome_root.find(".//{*}Pixels")
        if pixels_element is None:
            raise ValueError("Could not find 'Pixels' element in OME metadata.")

        self._num_channels = int(pixels_element.get("SizeC", "1"))
        self._num_samples = int(pixels_element.get("SizeT", "1"))
        self._num_rows = int(pixels_element.get("SizeY"))
        self._num_columns = int(pixels_element.get("SizeX"))
        self._num_z = int(pixels_element.get("SizeZ", "1"))
        self._dimension_order = pixels_element.get("DimensionOrder")

        # Series is a concept from the tifffile library.
        # It indexes all the data across pages and files
        series = self._tiff_reader.series[0]
        self._dtype = series.dtype
        number_of_pages = len(series)
        series_axes = series.axes
        # "XYZTC" or "XYCZT" Note this is different from OME metadata as this is data layout
        series_shape = series.shape

        # Determine non-spatial axes (remove X and Y).
        non_spatial_axes = [axis for axis in series_axes if axis not in ("X", "Y")]
        non_spatial_shape = [dim for axis, dim in zip(series_axes, series_shape) if axis not in ("X", "Y")]

        if "T" not in non_spatial_axes:
            raise ValueError("The TIFF file must have a T (time) dimension. Image stack mode is not supported.")

        total_expected_pages = math.prod(non_spatial_shape)
        if total_expected_pages != number_of_pages:
            warnings.warn(f"Expected {total_expected_pages} pages but found {number_of_pages} pages in the series.")

        # Identify axis indices.
        self._time_axis_index = non_spatial_axes.index("T")
        self._z_axis_index = non_spatial_axes.index("Z") if "Z" in non_spatial_axes else None
        self._channel_axis_index = non_spatial_axes.index("C") if "C" in non_spatial_axes else None

        self._non_spatial_axes = non_spatial_axes
        self._non_spatial_shape = non_spatial_shape

        # Build the mapping from each time frame (T) to its corresponding pages.
        self._frame_page_mapping: Dict[int, List[ThorTiffImagingExtractor.PageMapping]] = defaultdict(list)
        for page_index in range(number_of_pages):
            page_multi_index = np.unravel_index(page_index, non_spatial_shape, order="C")
            time_index = page_multi_index[self._time_axis_index]
            channel_index = page_multi_index[self._channel_axis_index] if self._channel_axis_index is not None else None
            depth_index = page_multi_index[self._z_axis_index] if self._z_axis_index is not None else None
            mapping_entry = ThorTiffImagingExtractor.PageMapping(
                page_index=page_index, channel_index=channel_index, depth_index=depth_index
            )
            self._frame_page_mapping[time_index].append(mapping_entry)

        self._kwargs = {"file_path": str(file_path)}

    def _parse_experiment_xml(self) -> None:
        """Parse Experiment.xml and extract metadata.

        Extract metadata such as frame rate and channel names from the Experiment.xml file.
        """
        experiment_xml_path = self.folder_path / "Experiment.xml"
        if experiment_xml_path.exists():
            # Parse the XML file into a dictionary
            self._experiment_xml_dict = _read_xml_as_dict(experiment_xml_path)

            # Extract sampling frequency from LSM element
            thor_experiment = self._experiment_xml_dict.get("ThorImageExperiment", {})
            lsm = thor_experiment.get("LSM", {})
            if lsm and "@frameRate" in lsm:
                self._sampling_frequency = float(lsm["@frameRate"])
            else:
                raise ValueError("Could not find 'LSM' element with frameRate attribute in Experiment.xml.")

            # Extract channel names from Wavelength elements
            wavelengths = thor_experiment.get("Wavelengths", {})
            wavelength_list = wavelengths.get("Wavelength", [])

            # Ensure wavelength_list is a list even if there's only one wavelength
            if not isinstance(wavelength_list, list):
                wavelength_list = [wavelength_list]

            self._channel_names = [w.get("@name") for w in wavelength_list if "@name" in w]

            if self.channel_name is not None and self.channel_name not in self._channel_names:
                raise ValueError(
                    f"Channel '{self.channel_name}' not available. Available channels: {self._channel_names}"
                )

            # Set channel filter if a channel name is provided.
            if self.channel_name is not None:
                self._channel_index_for_filter = None
                for index, name in enumerate(self._channel_names):
                    if self.channel_name in name:
                        self._channel_index_for_filter = index
                        break
                if self._channel_index_for_filter is None:
                    raise ValueError(f"Channel '{self.channel_name}' not found in Experiment.xml.")
                # Update channel names and count based on the filter.
                self._channel_names = [self._channel_names[self._channel_index_for_filter]]
                self._num_channels = 1
        else:
            raise ValueError(f"Experiment.xml file not found in {self.folder_path}.")

    @staticmethod
    def _parse_ome_metadata(metadata_string: str):
        """Parse an OME metadata string using lxml.etree.

        Removes XML comments if present and attempts to parse as bytes first.

        In the old version of ome tiff the metadata is stored as a comment. In the new version, the metadata
        is stored as an utf-8 encoded xml string.
        """
        if metadata_string.lstrip().startswith("<!--"):
            metadata_string = metadata_string.replace("<!--", "").replace("-->", "")
        try:
            return ET.fromstring(metadata_string.encode("utf-8"))
        except ValueError:
            return ET.fromstring(metadata_string)

    def get_frames(self, frame_idxs: List[int]) -> np.ndarray:
        """
        Get specific frames by their time indices.

        Parameters
        ----------
        frame_idxs : List[int]
            List of time/frame indices to retrieve.

        Returns
        -------
        np.ndarray
            Array of shape (n_frames, height, width) if no depth, or
            (n_frames, height, width, planes) if a Z dimension exists.
        """
        # Use the stored tiff_reader instead of opening a new one
        series = self._tiff_reader.series[0]
        data_type = series.dtype
        image_height = self._num_rows
        image_width = self._num_columns

        has_z_dimension = self._z_axis_index is not None and self._num_z > 1
        number_of_z_planes = self._num_z if has_z_dimension else 1

        n_frames = len(frame_idxs)
        output_shape = (
            (n_frames, image_height, image_width, number_of_z_planes)
            if has_z_dimension
            else (n_frames, image_height, image_width)
        )
        output_array = np.empty(output_shape, dtype=data_type)

        for frame_counter, frame_idx in enumerate(frame_idxs):
            if frame_idx not in self._frame_page_mapping:
                raise ValueError(f"No pages found for frame {frame_idx}.")
            page_mappings = self._frame_page_mapping[frame_idx]

            # Filter by channel if a channel name was provided.
            if self._channel_axis_index is not None and self.channel_name is not None:
                filter_index = self._channel_index_for_filter
                page_mappings = [m for m in page_mappings if m.channel_index == filter_index]

            if has_z_dimension:
                page_mappings.sort(key=lambda entry: entry.depth_index)
                if len(page_mappings) != number_of_z_planes:
                    raise ValueError(
                        f"Expected {number_of_z_planes} pages for frame {frame_idx} but got {len(page_mappings)}."
                    )
                for depth_counter, mapping_entry in enumerate(page_mappings):
                    page_data = series.pages[mapping_entry.page_index].asarray()
                    output_array[frame_counter, :, :, depth_counter] = page_data
            else:
                if len(page_mappings) != 1:
                    raise ValueError(f"Expected 1 page for frame {frame_idx} but got {len(page_mappings)}.")
                single_page_index = page_mappings[0].page_index
                page_data = series.pages[single_page_index].asarray()
                output_array[frame_counter, :, :] = page_data

        return output_array

    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self._num_samples
        frame_indices = list(range(start_sample, end_sample))
        return self.get_frames(frame_indices)

    def get_video(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> np.ndarray:
        """Get a range of frames.

        Parameters
        ----------
        start_frame: int, optional
            Start frame index (inclusive).
        end_frame: int, optional
            End frame index (exclusive).

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
        return self.get_series(start_sample=start_frame, end_sample=end_frame)

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return self._num_rows, self._num_columns

    def get_image_size(self) -> Tuple[int, int]:
        """Return the image dimensions (height, width)."""
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._num_rows, self._num_columns

    def get_num_samples(self) -> int:
        """Return the number of samples (time points)."""
        return self._num_samples

    def get_num_frames(self) -> int:
        """Return the number of frames (time points).

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

    def get_sampling_frequency(self) -> Optional[float]:
        """Return the sampling frequency, if available."""
        return self._sampling_frequency

    def get_num_channels(self) -> int:
        """Return the number of channels."""
        return self._num_channels

    def get_channel_names(self) -> List[str]:
        """Return the channel names."""
        return self._channel_names

    def get_dtype(self):
        """Return the data type of the video."""
        return self._dtype

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # ThorLabs TIFF imaging data does not have native timestamps
        return None

    def __del__(self):
        """Close the tiff_reader when the object is garbage collected."""
        if hasattr(self, "_tiff_reader"):
            self._tiff_reader.close()


def _xml_element_to_dict(elem):
    """Convert an ElementTree element into a dictionary."""
    dictionary = {elem.tag: {} if elem.attrib else None}
    children = list(elem)
    if children:
        nested_dictionary = {}
        for child in children:
            child_dict = _xml_element_to_dict(child)
            tag = child.tag
            # If the tag is already present, convert to a list
            if tag in nested_dictionary:
                if type(nested_dictionary[tag]) is list:
                    nested_dictionary[tag].append(child_dict[tag])
                else:
                    nested_dictionary[tag] = [nested_dictionary[tag], child_dict[tag]]
            else:
                nested_dictionary.update(child_dict)
        dictionary = {elem.tag: nested_dictionary}
    if elem.attrib:
        dictionary[elem.tag].update({f"@{k}": v for k, v in elem.attrib.items()})
    text = elem.text.strip() if elem.text else ""
    if text:
        if children or elem.attrib:
            dictionary[elem.tag]["#text"] = text
        else:
            dictionary[elem.tag] = text
    return dictionary


def _read_xml_as_dict(file_path: Union[str, Path]) -> dict:
    """Read an XML file and convert it to a dictionary."""
    xml_dict = _xml_element_to_dict(ET.parse(file_path).getroot())
    return xml_dict
