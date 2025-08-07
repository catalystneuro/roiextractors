"""Extractor for multiple TIFF files, each with multiple pages.

Classes
-------
MultiTIFFMultiPageExtractor
    An extractor for handling multiple TIFF files, each with multiple pages, organized according to a specified dimension order.
"""

import glob
import warnings
from pathlib import Path
from typing import Optional

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
        file_paths: list[PathType],
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
        file_paths : list[PathType]
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
        num_channels : int, default=1
            Number of channels.
        channel_index : int, default=0
            Index of the channel to extract. Default is 0 (first channel).
        num_planes : int, default=1
            Number of depth planes (Z).
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


        Notes
        -----
        Dimension Order Notes
        ---------------------
        This class follows a subset of the OME-TIFF dimension order specification, focusing on
        the Z (depth), C (channel), and T dimensions. The XY spatial dimensions are
        assumed to be the first two dimensions of each frame and are not included in the
        dimension_order parameter.

        While we use 'T' for compatibility with the OME-TIFF standard, we emphasize that its
        meaning varies significantly based on position:
        - When T is first (TCZ, TZC): Represents oversampling - multiple samples acquired at
            each depth or channel:
            - TCZ: T samples per channel at each depth position
            - TZC: T samples per depth position for each channel
        - When T is middle (ZTC, CTZ): Represents repetitions - repeated acquisitions of
            sub-structures before varying the outer dimension
            - ZTC: T repetitions of each Z-stack before switching channels
            - CTZ: T repetitions of the full channel set at each depth
        - When T is last (ZCT, CZT): Represents acquisition cycles - complete acquisitions
            of the entire multi-channel, multi-plane dataset
            - ZCT: T complete multi-channel volumes where the depth is varied first
            - CZT: T complete multi-channel volumes where the channel is varied first

        For more information on OME-TIFF dimension order, see:
        https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/specification.html

        Acquisition Patterns
        --------------------
        ZCT (Depth → Channel → Acquisition Cycles)
            Acquire a complete Z-stack for the first channel, then switch to the next channel
            and acquire its full Z-stack. After all channels are acquired, this constitutes
            one acquisition cycle. Repeat for T acquisition cycles.

        ZTC (Depth → Repetitions → Channel)
            Acquire full Z-stacks repeated T times for a single channel, then switch to
            the next channel and acquired another T repetitions of the full Z-stack for that
            channel. Repeat the same process for all channels.

        CZT (Channel → Depth → Acquisition Cycles)
            At the first depth position, acquire all channels sequentially. Move to the next
            depth and acquire all channels again. After completing all depths, one acquisition
            cycle is complete. Repeat for T acquisition cycles.

        CTZ (Channel → Repetitions → Depth)
            At a fixed depth position, acquire all channels, then repeat this channel
            acquisition T times. Then move to the next depth position and repeat the
            pattern of T repetitions of all channels.

        TCZ (Oversampling → Channel → Depth)
            At a fixed depth position, acquire T samples for the first channel,
            then acquire T samples for the next channel. After oversampling all channels
            at this depth, move to the next depth position and repeat.

        TZC (Oversampling → Depth → Channel)
            For a fixed channel, acquire T samples at the first depth, then T samples at
            the second depth, continuing through all depths. Switch to the next channel
            and repeat the entire oversampling pattern across depths.
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

        self._file_paths = [Path(file_path) for file_path in file_paths]
        self._dimension_order = dimension_order
        self._num_channels = num_channels
        self._channel_index = channel_index
        self._num_planes = num_planes
        self._sampling_frequency = sampling_frequency

        # Get tifffile package
        tifffile = get_package(package_name="tifffile")

        # Open all TIFF files and store file handles for lazy loading
        self._file_handles = []
        total_ifds = 0

        for file_path in self._file_paths:
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

        self._num_acquisition_cycles = num_acquisition_cycles

        # Create full mapping for all channels, times, and depths
        full_mapping = self._create_frame_to_ifd_table(
            dimension_order=self._dimension_order,
            num_channels=self._num_channels,
            num_acquisition_cycles=self._num_acquisition_cycles,
            num_planes=self._num_planes,
            ifds_per_file=ifds_per_file,
        )

        # Filter mapping for the specified channel
        channel_mask = full_mapping["channel_index"] == self._channel_index
        self._frames_to_ifd_table = full_mapping[channel_mask]

        # Determine if we're dealing with volumetric data
        self.is_volumetric = self._num_planes > 1

    def get_num_planes(self) -> int:
        """Get the number of depth planes."""
        return self._num_planes

    def get_volume_shape(self) -> tuple[int, int, int]:
        """Get the shape of a single volume (num_rows, num_columns, num_planes).

        Returns
        -------
        tuple
            Shape of a single volume (num_rows, num_columns, num_planes).
        """
        return (self._num_rows, self._num_columns, self._num_planes)

    @staticmethod
    def _create_frame_to_ifd_table(
        dimension_order: str, num_channels: int, num_acquisition_cycles: int, num_planes: int, ifds_per_file: list[int]
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
        ifds_per_file : list[int]
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

    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
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
        start_sample = int(start_sample) if start_sample is not None else 0
        end_sample = int(end_sample) if end_sample is not None else self.get_num_samples()

        samples_in_series = end_sample - start_sample

        # Always preallocate output array as volumetric
        num_rows, num_columns, num_planes = self.get_volume_shape()
        dtype = self.get_dtype()
        samples = np.empty((samples_in_series, num_rows, num_columns, num_planes), dtype=dtype)

        for return_index, sample_index in enumerate(range(start_sample, end_sample)):
            for depth_position in range(num_planes):

                # Calculate the index in the mapping table array
                frame_index = sample_index * num_planes + depth_position
                table_row = self._frames_to_ifd_table[frame_index]
                file_index = table_row["file_index"]
                ifd_index = table_row["IFD_index"]

                tiff_handle = self._file_handles[file_index]
                image_file_directory = tiff_handle.pages[ifd_index]
                samples[return_index, :, :, depth_position] = image_file_directory.asarray()

        # Squeeze the depth dimension if not volumetric
        if not self.is_volumetric:
            samples = samples.squeeze(axis=3)

        return samples

    def get_image_shape(self) -> tuple[int, int]:
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

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """No native timestamps for native this extractor."""
        return None

    def get_channel_names(self):
        channel_names = [f"Channel {i}" for i in range(self._num_channels)]
        return channel_names
