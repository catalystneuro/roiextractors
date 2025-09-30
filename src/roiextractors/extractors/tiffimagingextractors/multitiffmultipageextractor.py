"""Extractor for multiple TIFF files, each with multiple pages.

Classes
-------
MultiTIFFMultiPageExtractor
    An extractor for handling multiple TIFF files, each with multiple pages, organized according to a specified dimension order.
"""

import glob
import warnings
from pathlib import Path
from typing import Literal

import numpy as np

from ...extraction_tools import PathType, get_package
from ...imagingextractor import ImagingExtractor


class MultiTIFFMultiPageExtractor(ImagingExtractor):
    """
    An extractor for multi-page TIFF files with flexible organization of channels and depth planes.

    By default, this extractor is configured for simple planar time-series data (single channel, 2D frames)
    where each TIFF page represents a sequential time point. This covers the most common use case and
    requires only file paths (which might be only one) and sampling frequency.

    For more complex data (multi-channel or volumetric), additional parameters must be specified to
    define how channels and Z-planes are organized across TIFF pages. The extractor creates an
    efficient mapping between logical sample indices and their physical locations in the TIFF files.

    For multi-channel or volumetric data, you must explicitly specify:
    - num_channels: Number of channels (required if > 1)
    - channel_name: Which channel to extract (required if num_channels > 1)
    - num_planes: Number of Z-planes per volume (required if > 1)
    - dimension_order: How channels and Z-planes are interleaved across TIFF pages/IFDs
    """

    extractor_name = "MultiTIFFMultiPageExtractor"

    def __init__(
        self,
        file_paths: list[PathType],
        sampling_frequency: float,
        dimension_order: Literal["ZCT", "ZTC", "CZT", "CTZ", "TCZ", "TZC"] = "TZC",
        num_channels: int | None = None,
        channel_name: str | None = None,
        num_planes: int | None = None,
    ):
        """
        Initialize the extractor with file paths and dimension information.

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
            Default is "TZC" (time-first, suitable for planar single-channel data).
            See Notes section for detailed explanations of each dimension order.
        num_channels : int, optional
            Number of channels in the data. Default is 1 (single channel).
        channel_name : str, optional
            Name of the channel to extract (e.g., "0", "1"). Only required when
            num_channels > 1. For single-channel data, this parameter is ignored.
            Default is None.
        num_planes : int, optional
            Number of depth planes (Z-slices) per volume. Default is 1 (planar data).


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

        Special Cases
        ------------------------------
        When data has only a single channel (num_channels=1):

            ZCT, ZTC, CZT → ZT: Volumetric time series (complete volumes acquired sequentially)

            CTZ, TCZ, TZC → TZ: Plane-by-plane time series (T samples at each depth before moving to next)

        When data is planar (num_planes=1):

            ZCT, CZT, CTZ → CT: Channel-first acquisition patterns (one full channel sweep after another)

            ZTC, TCZ, TZC → TC: Time-first acquisition patterns (full data for one channel and then the next)

        When data is both single-channel AND planar:

            All dimension orders → T: Simple planar time series (all orderings equivalent)
        """
        super().__init__()

        valid_dimension_orders = ["ZCT", "ZTC", "CZT", "CTZ", "TCZ", "TZC"]
        if dimension_order not in valid_dimension_orders:
            raise ValueError(f"Invalid dimension order: {dimension_order}. Must be one of: {valid_dimension_orders}")

        # Set default values and validate parameters
        num_channels = 1 if num_channels is None else num_channels
        num_planes = 1 if num_planes is None else num_planes

        if num_channels < 1:
            raise ValueError("num_channels must be at least 1")

        if num_planes < 1:
            raise ValueError("num_planes must be at least 1")

        # Handle channel selection
        if num_channels > 1:
            if channel_name is None:
                raise ValueError("channel_name must be specified when num_channels > 1")
            # Parse channel name to get index (assumes numeric format "0", "1", etc.)
            try:
                channel_index = int(channel_name)
            except ValueError:
                raise ValueError(
                    f"Invalid channel name format: {channel_name}. Expected numeric format: '0', '1', etc."
                )
        else:
            channel_index = 0  # Single channel case

        # Validate channel_index
        if channel_index >= num_channels:
            raise ValueError(f"channel_index {channel_index} is out of range (0 to {num_channels-1})")

        self._file_paths = [Path(file_path) for file_path in file_paths]
        self._dimension_order = dimension_order
        self._num_channels = num_channels
        self._channel_index = channel_index
        self._num_planes = num_planes
        self._sampling_frequency = sampling_frequency

        tifffile = get_package(package_name="tifffile")

        # Open all TIFF files and store file handles for lazy loading
        self._file_handles = []
        total_ifds = 0

        for file_path in self._file_paths:
            # Check if file exists first
            if not Path(file_path).exists():
                # Close any already opened file handles before raising the exception
                for handle in self._file_handles:
                    handle.close()
                raise FileNotFoundError(f"TIFF file not found: {file_path}")

            try:
                tiff_handle = tifffile.TiffFile(file_path)
                self._file_handles.append(tiff_handle)
                total_ifds += len(tiff_handle.pages)
            except Exception as e:
                # Close any opened file handles before raising the exception
                for handle in self._file_handles:
                    handle.close()
                raise RuntimeError(f"Error opening TIFF file {file_path}: {e}")

        first_ifd = self._file_handles[0].pages[0]
        self._num_rows, self._num_columns = first_ifd.shape
        self._dtype = first_ifd.dtype

        # Create mapping table for all available IFDs
        ifds_per_file = [len(handle.pages) for handle in self._file_handles]
        total_ifds = sum(ifds_per_file)

        ifds_per_cycle = num_channels * num_planes

        # Warn if total IFDs doesn't divide evenly into complete cycles
        if total_ifds % ifds_per_cycle != 0:
            warnings.warn(
                f"Total IFDs ({total_ifds}) is not divisible by IFDs per cycle ({ifds_per_cycle}). "
                f"Some samples may not be accessible."
            )

        # Create full mapping for all available IFDs
        full_mapping = self._create_frame_to_ifd_table(
            dimension_order=self._dimension_order,
            num_channels=self._num_channels,
            num_planes=self._num_planes,
            ifds_per_file=ifds_per_file,
        )

        # Filter mapping for the specified channel
        channel_mask = full_mapping["channel_index"] == self._channel_index
        self._frames_to_ifd_table = full_mapping[channel_mask]

        # Sort by time_index first, then depth_index to ensure time-coherent sequential grouping
        # This ensures frames from the same time point are consecutive, regardless of dimension order
        sort_indices = np.lexsort(
            (
                self._frames_to_ifd_table["depth_index"],  # Secondary sort key
                self._frames_to_ifd_table["time_index"],  # Primary sort key
            )
        )
        self._frames_to_ifd_table = self._frames_to_ifd_table[sort_indices]

        # Determine if we're dealing with volumetric data
        self.is_volumetric = self._num_planes > 1

        # Determine number of samples from the filtered mapping table
        # For non-volumetric: each IFD is one sample
        # For volumetric: num_planes IFDs make one volume sample
        if self.is_volumetric:
            self._num_samples = len(self._frames_to_ifd_table) // self._num_planes
        else:
            self._num_samples = len(self._frames_to_ifd_table)

    def get_num_planes(self) -> int:
        """Get the number of depth planes."""
        return self._num_planes

    @staticmethod
    def _create_frame_to_ifd_table(
        dimension_order: str,
        num_channels: int,
        num_planes: int,
        ifds_per_file: list[int],
    ) -> np.ndarray:
        """Create a mapping table from logical dimensions to physical TIFF locations.

        Maps each IFD (Image File Directory) to its logical position within the multi-dimensional
        acquisition based on the specified dimension order. This enables efficient lookup of
        specific time points, channels, and depth planes from the raw TIFF files.

        The dimension order determines how IFDs are interpreted:
        - First dimension changes fastest (varies most rapidly across IFDs)
        - Last dimension changes slowest (varies least frequently across IFDs)

        For example, with dimension_order="CZT":
        - IFDs cycle through Channels first, then depth (Z), then Time
        - IFD sequence: C0Z0T0, C1Z0T0, C0Z1T0, C1Z1T0, C0Z0T1, C1Z0T1, ...

        Parameters
        ----------
        dimension_order : str
            Dimension ordering (e.g., "CZT", "ZTC"). Determines how IFDs map to
            logical coordinates.
        num_channels : int
            Total number of channels in the acquisition.
        num_planes : int
            Number of depth planes (Z-slices) per volume.
        ifds_per_file : list[int]
            Number of IFDs in each TIFF file, used to map global IFD indices
            to specific files and local IFD positions.

        Returns
        -------
        np.ndarray
            Structured array with fields: 'file_index', 'IFD_index', 'channel_index',
            'depth_index', 'time_index'. Each row maps one IFD to its logical coordinates.
        """
        mapping_dtype = np.dtype(
            [
                ("file_index", np.uint16),
                ("IFD_index", np.uint16),
                ("channel_index", np.uint8),
                ("depth_index", np.uint8),
                ("time_index", np.uint16),
            ]
        )

        total_entries = sum(ifds_per_file)
        ifds_per_cycle = num_channels * num_planes
        num_acquisition_cycles = total_entries // ifds_per_cycle

        dimension_sizes = {"Z": num_planes, "T": num_acquisition_cycles, "C": num_channels}

        # Calculate divisors for dimension indexing (first dimension changes fastest)
        dimension_divisors = {}
        current_divisor = 1
        for dimension in dimension_order:
            dimension_divisors[dimension] = current_divisor
            current_divisor *= dimension_sizes[dimension]

        indices = np.arange(total_entries)

        depth_indices = (indices // dimension_divisors["Z"]) % dimension_sizes["Z"]
        time_indices = (indices // dimension_divisors["T"]) % dimension_sizes["T"]
        channel_indices = (indices // dimension_divisors["C"]) % dimension_sizes["C"]

        file_indices = np.concatenate(
            [np.full(num_ifds, file_idx, dtype=np.uint16) for file_idx, num_ifds in enumerate(ifds_per_file)]
        )

        ifd_indices = np.concatenate([np.arange(num_ifds, dtype=np.uint16) for num_ifds in ifds_per_file])

        file_indices = file_indices[:total_entries]
        ifd_indices = ifd_indices[:total_entries]

        mapping = np.zeros(total_entries, dtype=mapping_dtype)
        mapping["file_index"] = file_indices
        mapping["IFD_index"] = ifd_indices
        mapping["channel_index"] = channel_indices
        mapping["depth_index"] = depth_indices
        mapping["time_index"] = time_indices

        return mapping

    def get_series(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
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

        """
        start_sample = int(start_sample) if start_sample is not None else 0
        end_sample = int(end_sample) if end_sample is not None else self.get_num_samples()

        # Clamp end_sample to the actual number of samples
        end_sample = min(end_sample, self.get_num_samples())
        start_sample = max(0, start_sample)

        samples_in_series = end_sample - start_sample

        # Always preallocate output array as volumetric
        num_rows, num_columns = self.get_frame_shape()
        num_planes = self.get_num_planes()
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
        dimension_order: Literal["ZCT", "ZTC", "CZT", "CTZ", "TCZ", "TZC"] = "TZC",
        num_channels: int | None = None,
        channel_name: str | None = None,
        num_planes: int | None = None,
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
            The order of dimensions in the data. Default is "TZC".
        num_channels : int, optional
            Number of channels in the data. Default is 1 (single channel).
        channel_name : str, optional
            Name of the channel to extract (e.g., "0", "1"). Only required when num_channels > 1. Default is None.
        num_planes : int, optional
            Number of depth planes (Z). Default is 1 (planar data).

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
            channel_name=channel_name,
            num_planes=num_planes,
        )

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        """No native timestamps for native this extractor."""
        return None

    def get_channel_names(self):
        channel_names = [str(i) for i in range(self._num_channels)]
        return channel_names
