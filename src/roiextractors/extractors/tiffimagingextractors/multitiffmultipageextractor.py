"""Extractor for multiple TIFF files, each with multiple pages.

Classes
-------
MultiTIFFMultiPageExtractor
    An extractor for handling multiple TIFF files, each with multiple pages, organized according to a specified dimension order.
"""

import glob
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ...extraction_tools import PathType, get_package
from ...imagingextractor import ImagingExtractor


class MultiTIFFMultiPageExtractor(ImagingExtractor):
    """
    An extractor for handling multiple TIFF files, each with multiple pages, organized according to a specified dimension order.

    This extractor allows for lazy loading of samples from multiple TIFF files, where each file may contain multiple pages.
    The samples are organized according to a specified dimension order (e.g., ZCT) and the size of each dimension.

    The extractor creates a mapping between each logical sample index and its corresponding file and IFD location.
    This mapping is used to efficiently retrieve samples when requested.
    """

    extractor_name = "MultiTIFFMultiPageExtractor"

    def __init__(
        self,
        file_paths: List[PathType],
        sampling_frequency: float,
        dimension_order: str = "ZCT",
        num_channels: int = 1,
        channel_index: int = 0,
        num_planes: int = 1,
        num_acquisition_cycles: Optional[int] = None,
    ):
        """Initialize the extractor with file paths and dimension information.

        Parameters
        ----------
        file_paths : List[PathType]
            List of paths to TIFF files.
        sampling_frequency : float
            The sampling frequency in Hz.
            Note that if your data is volumetric the sampling rate is the sampling rate of the volume,
            not the individual frames.
        dimension_order : str, optional
            The order of dimensions in the data. Must be one of: ZCT, ZTC, CTZ, TCZ, CZT, TZC.
            This follows the OME-TIFF specification for dimension order, but excludes the XY
            dimensions which are assumed to be the first two dimensions.
            Default is "ZCT".
        num_channels : int, optional
            Number of channels. Default is 1.
        channel_index : int, optional
            Index of the channel to extract. Default is 0 (first channel).
        num_planes : int, optional
            Number of depth planes (Z). Default is 1.
        num_acquisition_cycles : int, optional
            The total number of complete acquisition cycles present in the dataset. An acquisition cycle
            represents one full sweep through the imaging dimensions according to the specified dimension_order.

            In microscopy applications where oversampling occurs (multiple samples per depth/channel):
            - If dimension_order starts with "T" (e.g., "TCZ", "TZC"): This parameter indicates how many
            times each frame is acquired before changing other dimensions.
            - If "T" is the second dimension (e.g., "ZTC", "CZT"): This indicates how many complete
            scans occur at each level of the first dimension before proceeding.
            - If "T" is the last dimension (e.g., "ZCT", "CZT"): This represents distinct timepoints
            in a time series.

            When set to None (default), the extractor automatically calculates this value as:
                num_acquisition_cycles = total_ifds // (num_channels * num_planes)

            This is particularly important for:
            1. Oversampling to improve signal-to-noise ratio
            2. Capturing dynamics at different timescales
            3. Multi-exposure imaging where the same volume is imaged repeatedly

        Notes
        -----
        Dimension Order Notes:

        This class uses a subset of the OME-TIFF dimension order specification, focusing on
        the Z, C, and T dimensions. The XY dimensions are assumed to be the first two
        dimensions and are not included in the dimension_order parameter.
        For more information on OME-TIFF dimension order, see:
        https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/specification.html

        ZCT (Z, then Channel, then Time)
        Acquisition pattern: Complete a Z-stack for one channel, then switch channels and
        repeat the Z-stack, then move to the next timepoint.

        ZTC (Z, then Time, then Channel)
        Acquisition pattern: Complete all Z-stacks for all timepoints in one channel before
        switching to the next channel.

        CZT (Channel, then Z, then Time)
        Acquisition pattern: At each Z position, all channels are imaged before moving to
        the next Z position, and this entire process is repeated at each timepoint.

        CTZ (Channel, then Time, then Z)
        Acquisition pattern: At each Z position, all channels are imaged across all
        timepoints before moving to the next Z position.

        TCZ (Time, then Channel, then Z)
        Acquisition pattern: A full time series is collected for one channel at one Z
        position, then all channels are imaged at that Z before moving to the next Z.

        TZC (Time, then Z, then Channel)
        Acquisition pattern: A full time series is collected for all Z positions in one
        channel before switching channels.
        """
        super().__init__()

        # Validate dimension order
        valid_dimension_orders = ["ZCT", "ZTC", "CZT", "CTZ", "TCZ", "TZC"]
        if dimension_order not in valid_dimension_orders:
            raise ValueError(f"Invalid dimension order: {dimension_order}. Must be one of: {valid_dimension_orders}")

        # Validate num_planes
        if num_planes < 1:
            raise ValueError("num_planes must be at least 1")

        # Validate channel_index
        if channel_index >= num_channels:
            raise ValueError(f"channel_index {channel_index} is out of range (0 to {num_channels-1})")

        self.file_paths = [Path(file_path) for file_path in file_paths]
        self.dimension_order = dimension_order
        self.num_channels = num_channels
        self.channel_index = channel_index
        self.num_planes = num_planes
        self._sampling_frequency = sampling_frequency

        # Get tifffile package
        tifffile = get_package(package_name="tifffile")

        # Open all TIFF files and store file handles for lazy loading
        self._file_handles = []
        total_ifds = 0

        for file_path in self.file_paths:
            try:
                tiff_handle = tifffile.TiffFile(file_path)
                self._file_handles.append(tiff_handle)
                total_ifds += len(tiff_handle.pages)
            except Exception as e:
                # Close any opened file handles before raising the exception
                for handle in self._file_handles:
                    handle.close()
                raise RuntimeError(f"Error opening TIFF file {file_path}: {e}")

        # Get image dimensions from the first IFD of the first file
        if not self._file_handles or not self._file_handles[0].pages:
            raise ValueError("No valid TIFF files or IFDs found")

        first_ifd = self._file_handles[0].pages[0]
        self._num_rows, self._num_columns = first_ifd.shape
        self._dtype = first_ifd.dtype

        # Calculate num_acquisition_cycles if not provided
        ifds_per_file = [len(handle.pages) for handle in self._file_handles]
        total_ifds = sum(ifds_per_file)

        ifds_per_cycle = num_channels * num_planes
        if ifds_per_cycle == 0:
            raise ValueError("Invalid dimension sizes: num_channels and num_planes cannot both be zero")

        self._num_samples = total_ifds // ifds_per_cycle

        if num_acquisition_cycles is None:

            # Calculate num_acquisition_cycles based on total IFDs and other dimensions
            if total_ifds % ifds_per_cycle != 0:
                warnings.warn(
                    f"Total IFDs ({total_ifds}) is not divisible by IFDs per cycle ({ifds_per_cycle}). "
                    f"Some samples may not be accessible."
                )

            num_acquisition_cycles = total_ifds // ifds_per_cycle

        self.num_acquisition_cycles = num_acquisition_cycles

        # Create full mapping for all channels, times, and depths
        full_mapping = self._create_sample_to_ifd_mapping(
            dimension_order=dimension_order,
            num_channels=num_channels,
            num_acquisition_cycles=num_acquisition_cycles,
            num_planes=num_planes,
            ifds_per_file=ifds_per_file,
        )

        # Filter mapping for the specified channel
        channel_mask = full_mapping["channel_index"] == channel_index
        sample_to_ifd_mapping = full_mapping[channel_mask]

        # Sort by time_index and depth_index for easier access
        sorted_indices = np.argsort(sample_to_ifd_mapping["time_index"])
        self._sample_to_ifd_mapping = sample_to_ifd_mapping[sorted_indices]

        # Determine if we're dealing with volumetric data
        self.is_volumetric = self.num_planes > 1

    @staticmethod
    def _create_sample_to_ifd_mapping(
        dimension_order: str, num_channels: int, num_acquisition_cycles: int, num_planes: int, ifds_per_file: List[int]
    ) -> np.ndarray:
        """Create a mapping from sample index to file and IFD indices.

        This function creates a structured numpy array that maps each combination of time,
        channel, and depth to its corresponding physical location in the TIFF files.

        Parameters
        ----------
        dimension_order : str
            The order of dimensions in the data.
        num_channels : int
            Number of channels.
        num_acquisition_cycles : int
            Number of acquisition cycles (timepoints).
        num_planes : int
            Number of depth planes (Z).
        ifds_per_file : List[int]
            Number of IFDs in each file.

        Returns
        -------
        np.ndarray
            A structured array mapping all combinations of time, channel, and depth to file
            and IFD indices.
        """
        # Create structured dtype for mapping
        mapping_dtype = np.dtype(
            [
                ("file_index", np.uint16),
                ("IFD_index", np.uint16),
                ("channel_index", np.uint8),
                ("depth_index", np.uint8),
                ("time_index", np.uint16),
            ]
        )

        # Calculate total number of entries
        total_entries = sum(ifds_per_file)

        # Define the sizes for each dimension
        dimension_sizes = {"Z": num_planes, "T": num_acquisition_cycles, "C": num_channels}

        # Calculate divisors for each dimension
        # In dimension order, the first element changes fastest
        dimension_divisors = {}
        current_divisor = 1
        for dimension in dimension_order:
            dimension_divisors[dimension] = current_divisor
            current_divisor *= dimension_sizes[dimension]

        # Create a linear range of IFD indices
        indices = np.arange(total_entries)

        # Calculate indices for each dimension
        depth_indices = (indices // dimension_divisors["Z"]) % dimension_sizes["Z"]
        time_indices = (indices // dimension_divisors["T"]) % dimension_sizes["T"]
        channel_indices = (indices // dimension_divisors["C"]) % dimension_sizes["C"]

        # Generate file_indices and local_ifd_indices using list comprehensions
        # Create arrays of file indices (repeating each file index for the number of IFDs in that file)
        file_indices = np.concatenate(
            [np.full(num_ifds, file_idx, dtype=np.uint16) for file_idx, num_ifds in enumerate(ifds_per_file)]
        )

        # Create arrays of local IFD indices (starting from 0 for each file)
        ifd_indices = np.concatenate([np.arange(num_ifds, dtype=np.uint16) for num_ifds in ifds_per_file])

        # Ensure we don't exceed total_entries
        file_indices = file_indices[:total_entries]
        ifd_indices = ifd_indices[:total_entries]

        # Create the structured array
        mapping = np.zeros(total_entries, dtype=mapping_dtype)
        mapping["file_index"] = file_indices
        mapping["IFD_index"] = ifd_indices
        mapping["channel_index"] = channel_indices
        mapping["depth_index"] = depth_indices
        mapping["time_index"] = time_indices

        return mapping

    def get_samples(self, sample_indices: np.ndarray) -> np.ndarray:
        """Get specific samples by their indices.

        Parameters
        ----------
        sample_indices : array-like
            Indices of samples to retrieve. Must be an array-like object, not a single integer.

        Returns
        -------
        numpy.ndarray
            Array of frames with shape (num_samples, height, width) if num_planes is 1,
            or (num_samples, height, width, num_planes) if num_planes > 1.

        Raises
        ------
        TypeError
            If frame_idxs is an integer instead of an array-like object.
        ValueError
            If any frame index is out of range.
        """
        if isinstance(sample_indices, (int, np.integer)):
            raise TypeError("sample_indices must be an array-like object, not a single integer")

        sample_indices = np.array(sample_indices)
        if np.any(sample_indices >= self._num_samples) or np.any(sample_indices < 0):
            raise ValueError(f"Frame indices must be between 0 and {self._num_samples - 1}")

        # Always preallocate output array as volumetric
        samples = np.empty((len(sample_indices), self._num_rows, self._num_columns, self.num_planes), dtype=self._dtype)

        # Load each requested frame
        for frame_idx_position, frame_idx in enumerate(sample_indices):
            for depth_position in range(self.num_planes):
                # Calculate the index in the mapping array
                # Each frame has num_planes entries in the mapping
                mapping_idx = frame_idx * self.num_planes + depth_position

                # Get the mapping for this frame and depth
                mapping_entry = self._sample_to_ifd_mapping[mapping_idx]
                file_index = mapping_entry["file_index"]
                file_handle = self._file_handles[file_index]
                ifd_index = mapping_entry["IFD_index"]

                ifd = file_handle.pages[ifd_index]
                samples[frame_idx_position, :, :, depth_position] = ifd.asarray()

        # Squeeze the depth dimension if not volumetric
        if not self.is_volumetric:
            samples = samples.squeeze(axis=3)

        return samples

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the size of the video (num_rows, num_columns).

        Returns
        -------
        tuple
            Size of the video (num_rows, num_columns).
        """
        return (self._num_rows, self._num_columns)

    get_frame_shape = get_image_shape

    def get_num_samples(self) -> int:
        """Get the number of frames in the video.

        Returns
        -------
        int
            Number of frames in the video.
        """
        return self._num_samples

    def get_sampling_frequency(self) -> float:
        """Get the sampling frequency in Hz.

        Returns
        -------
        float
            Sampling frequency in Hz.
        """
        return self._sampling_frequency

    def get_dtype(self):
        """Get the data type of the video.

        Returns
        -------
        dtype
            Data type of the video.
        """
        return self._dtype

    def __del__(self):
        """Close file handles when the extractor is garbage collected."""
        if hasattr(self, "_file_handles"):
            for handle in self._file_handles:
                try:
                    handle.close()
                except Exception:
                    pass

    @staticmethod
    def from_folder(
        folder_path: PathType,
        file_pattern: str,
        sampling_frequency: float,
        dimension_order: str = "ZCT",
        num_channels: int = 1,
        channel_index: int = 0,
        num_acquisition_cycles: Optional[int] = None,
        num_planes: int = 1,
    ) -> "MultiTIFFMultiPageExtractor":
        """Create an extractor from a folder path and file pattern.

        Parameters
        ----------
        folder_path : PathType
            Path to the folder containing TIFF files.
        file_pattern : str
            Glob pattern for identifying TIFF files (e.g., "*.tif").
        sampling_frequency : float
            The sampling frequency in Hz.
        dimension_order : str, optional
            The order of dimensions in the data. Default is "ZCT".
        num_channels : int, optional
            Number of channels. Default is 1.
        channel_index : int, optional
            Index of the channel to extract. Default is 0 (first channel).
        num_acquisition_cycles : int, optional
            Number of acquisition cycles (timepoints). If None, it will be calculated. Default is None.
        num_planes : int, optional
            Number of depth planes (Z). Default is 1.

        Returns
        -------
        MultiTIFFMultiPageExtractor
            The initialized extractor.
        """
        folder_path = Path(folder_path)
        file_paths = sorted(glob.glob(str(folder_path / file_pattern)))

        if not file_paths:
            raise ValueError(f"No files found matching pattern {file_pattern} in folder {folder_path}")

        return MultiTIFFMultiPageExtractor(
            file_paths=file_paths,
            sampling_frequency=sampling_frequency,
            dimension_order=dimension_order,
            num_channels=num_channels,
            channel_index=channel_index,
            num_acquisition_cycles=num_acquisition_cycles,
            num_planes=num_planes,
        )
