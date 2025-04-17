"""Specialized extractor for reading TIFF files produced via ScanImage.

Classes
-------
ScanImageTiffImagingExtractor
    Specialized extractor for reading TIFF files produced via ScanImage.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Iterable
import warnings
from warnings import warn
import numpy as np

from ...extraction_tools import PathType, FloatType, ArrayType, DtypeType, get_package
from ...imagingextractor import ImagingExtractor
from ...volumetricimagingextractor import VolumetricImagingExtractor
from ...multiimagingextractor import MultiImagingExtractor
from .scanimagetiff_utils import (
    extract_extra_metadata,
    parse_metadata,
    extract_timestamps_from_file,
    _get_scanimage_reader,
    read_scanimage_metadata,
)


class ScanImageImagingExtractor(ImagingExtractor):
    """
    Specialized extractor for reading TIFF files produced via ScanImage software.

    This extractor is designed to handle the structure of ScanImage TIFF files, which can contain
    multi channel and multi volume data.  It also supports both single-file and multi-file datasets generated
    by ScanImage in various acquisition modes (grab, focus, loop).

    The extractor creates a mapping between each frame in the dataset and its corresponding physical file
    and IFD (Image File Directory) location. This mapping enables efficient retrieval of specific frames
    without loading the entire dataset into memory, making it suitable for large datasets.

    Key features:
    - Handles multi-channel data with channel selection
    - Supports volumetric (multi-plane) imaging data
    - Automatically detects and loads multi-file datasets based on ScanImage naming conventions
    - Extracts and provides access to ScanImage metadata
    - Efficiently retrieves frames using lazy loading

    Current limitations:
    - Does not support datasets with multiple frames per slice (will raise ValueError)
    - Does not support datasets with flyback frames (will raise ValueError)
    """

    extractor_name = "ScanImageImagingExtractor"

    def __init__(
        self,
        file_path: PathType,
        channel_name: Optional[str] = None,
        file_paths: Optional[List[PathType]] = None,
    ):
        """
        Initialize the extractor.
        
        Parameters
        ----------
        file_path : PathType
            Path to the TIFF file. If this is part of a multi-file series, this should be the first file.
        channel_name : str, optional
            Name of the channel to extract. If None and multiple channels are available, the first channel will be used.
        file_paths : List[PathType], optional
            List of file paths to use. If provided, this overrides the automatic
            file detection heuristics. Use this if automatic detection does not work correctly and you know 
            exactly which files should be included.  The file paths should be provided in an order that 
            reflects the temporal order of the frames in the dataset.       
        """
        super().__init__()
        self.file_path = Path(file_path)

        # Validate file suffix
        valid_suffixes = [".tiff", ".tif", ".TIFF", ".TIF"]
        if self.file_path.suffix not in valid_suffixes:
            suffix_string = ", ".join(valid_suffixes[:-1]) + f", or {valid_suffixes[-1]}"
            warn(
                f"Suffix ({self.file_path.suffix}) is not of type {suffix_string}! "
                f"The {self.extractor_name} Extractor may not be appropriate for the file."
            )

        # Open the
        tifffile = get_package(package_name="tifffile")
        tiff_reader = tifffile.TiffReader(file_path)

        self._general_metadata = tiff_reader.scanimage_metadata
        self._metadata = self._general_metadata["FrameData"]

        # This criteria was confirmed by Lawrence Niu, a developer of ScanImage
        self.is_volumetric = self._metadata["SI.hStackManager.enable"]
        if self.is_volumetric:
            self._sampling_frequency = self._metadata["SI.hRoiManager.scanVolumeRate"]
            self._num_planes = self._metadata["SI.hStackManager.numSlices"]

            frames_per_slice = self._metadata["SI.hStackManager.numSlices"]
            if frames_per_slice > 1:
                error_msg = (
                    "Multiple frames per slice detected. "
                    "Please open an issue on GitHub roiextractors to request this feature: "
                )
                raise ValueError(error_msg)
            flyback_frames = (
                self._metadata["SI.hStackManager.numFramesPerVolumeWithFlyback"]
                - self._metadata["SI.hStackManager.numFramesPerVolume"]
            )

            if flyback_frames > 0:
                error_msg(
                    "Flyback frames detected. " "Please open an issue on GitHub roiextractors to request this feature: "
                )

                raise ValueError(error_msg)

        else:
            self._sampling_frequency = self._metadata["SI.hRoiManager.scanFrameRate"]
            self._num_planes = 1
            self._frames_per_slice = 1

        # This piece of the metadata is the indication that the channel is saved on the data
        channels_available = self._metadata["SI.hChannels.channelSave"]
        channels_available = [channels_available] if isinstance(channels_available, int) else channels_available
        self._num_channels = len(channels_available)

        # Determine their name and use matlab 1-indexing
        all_channel_names = self._metadata["SI.hChannels.channelName"]
        channel_names = [all_channel_names[channel_index - 1] for channel_index in channels_available]

        # Channel selection checks
        self._is_multi_channel_data = len(channel_names) > 1
        if self._is_multi_channel_data and channel_name is None:

            error_msg = (
                f"Multiple channels available in the data {channel_names}"
                "Please specify a channel name to extract data from."
            )
            raise ValueError(error_msg)
        elif self._is_multi_channel_data and channel_name is not None:
            if channel_name not in channel_names:
                error_msg = (
                    f"Channel name ({channel_name}) not found in available channels ({channel_names}). "
                    "Please specify a valid channel name."
                )
                raise ValueError(error_msg)

            self.channel_name = channel_name
            self._channel_index = channel_names.index(channel_name)
        else:  # single channel data

            self.channel_name = channel_names[0]
            self._channel_index = 0

        # Check if this is a multi-file dataset
        if file_paths is None:
            self.file_paths = self._find_data_files()
        else:
            self.file_paths = file_paths

        # Open all TIFF files and store only file readers for lazy loading
        total_ifds = 0
        self._tiff_readers = []
        for file_path in self.file_paths:
            try:
                tiff_reader = tifffile.TiffFile(file_path)
                self._tiff_readers.append(tiff_reader)
                total_ifds += len(tiff_reader.pages)
            except Exception as e:
                for tiff_reader in self._tiff_readers:
                    tiff_reader.close()
                raise RuntimeError(f"Error opening TIFF file {file_path}: {e}")

        # Get image dimensions from the first IFD of the first file
        if not self._tiff_readers or not self._tiff_readers[0].pages:
            raise ValueError("No valid TIFF files or IFDs found")

        first_ifd = self._tiff_readers[0].pages[0]
        self._num_rows, self._num_columns = first_ifd.shape
        self._dtype = first_ifd.dtype

        # Calculate total IFDs and samples
        ifds_per_file = [len(tiff_reader.pages) for tiff_reader in self._tiff_readers]
        total_ifds = sum(ifds_per_file)

        # For ScanImage, dimension order is always CZT
        # That is, jump through channels first and then depth and then the pattern is repeated
        dimension_order = "CZT"

        # Calculate number of samples
        ifds_per_cycle = self._num_channels * self._num_planes

        num_acquisition_cycles = total_ifds // ifds_per_cycle
        self._num_samples = num_acquisition_cycles  #  / len(channels_available)

        # Create full mapping for all channels, samples, and depths
        full_frame_to_ifds_table = self._create_frame_to_ifd_table(
            dimension_order=dimension_order,
            num_channels=self._num_channels,
            num_acquisition_cycles=num_acquisition_cycles,
            num_planes=self._num_planes,
            ifds_per_file=ifds_per_file,
        )

        # Filter mapping for the specified channel
        channel_mask = full_frame_to_ifds_table["channel_index"] == self._channel_index
        channel_frames_to_ifd_table = full_frame_to_ifds_table[channel_mask]

        # Sort by time_index and depth_index for easier access
        sorting_tuple = (channel_frames_to_ifd_table["time_index"], channel_frames_to_ifd_table["depth_index"])
        sorted_indices = np.lexsort(sorting_tuple)
        self._frames_to_ifd_table = channel_frames_to_ifd_table[sorted_indices]


    def _find_data_files(self) -> List[PathType]:
        """Find additional files in the series based on the file naming pattern.

        This method determines which files to include in the dataset using one of these approaches:

        1. If file_paths is provided: Uses the provided list of file paths directly
        2. If file_pattern is provided: Uses the provided pattern to glob for files
        3. Otherwise, analyzes the file name and ScanImage metadata to determine if the current file
           is part of a multi-file dataset. It uses different strategies based on the acquisition mode:
           - For 'grab' mode with finite frames per file: Uses base_name_acquisition_* pattern
           - For 'loop' mode: Uses base_name_* pattern
           - For 'slow' stack mode with volumetric data: Uses base_name_* pattern
           - Otherwise: Returns only the current file

        This information about ScanImage file naming was shared in a private conversation with
        Lawrence Niu, who is a developer of ScanImage.

        Returns
        -------
        List[PathType]
            List of paths to all files in the series, sorted naturally (e.g., file_1, file_2, file_10)
        """
        # Parse the file name to extract base name, acquisition number, and file index
        file_stem = self.file_path.stem

        # Can be grab, focus or loop, see
        # https://docs.scanimage.org/Basic+Features/Acquisitions.html
        acquisition_state = self._metadata["SI.acqState"]
        frames_per_file = self._metadata["SI.hScan2D.logFramesPerFile"]
        stack_mode = self._metadata["SI.hStackManager.stackMode"]

        # This is the happy path that is well specified in the documentation
        if acquisition_state == "grab" and frames_per_file != float("inf"):
            base_name, acquisition, file_index = file_stem.split("_")
            pattern = f"{base_name}_{acquisition}_*{self.file_path.suffix}"
        # Looped acquisitions also divides the files according to Lawrence Niu in private conversation
        elif acquisition_state == "loop":  # This also separates the files
            base_name = "_".join(file_stem.split("_")[:-1])  # Everything before the last _
            pattern = f"{base_name}_*{self.file_path.suffix}"
        # This also divided the files according to Lawrence Niu in private conversation
        elif stack_mode == "slow" and self.is_volumetric:
            base_name, acquisition, file_index = file_stem.split("_")
            pattern = f"{base_name}_*{self.file_path.suffix}"
        else:
            files_found = [self.file_path]
            return files_found

        from natsort import natsorted

        files_found = natsorted(self.file_path.parent.glob(pattern))
        return files_found

    @staticmethod
    def _create_frame_to_ifd_table(
        dimension_order: str, num_channels: int, num_acquisition_cycles: int, num_planes: int, ifds_per_file: List[int],
    ) -> np.ndarray:
        """
        Create a table that describes the data layout of the dataset.
        
        Every row in the table corresponds to a frame in the dataset and contains:
        - file_index: The index of the file in the series
        - IFD_index: The index of the IFD in the file
        - channel_index: The index of the channel
        - depth_index: The index of the depth
        - time_index: The index of the time

        The table is represented as a structured numpy array that maps each combination of time,
        channel, and depth to its corresponding physical location in the TIFF files.

        Parameters
        ----------
        dimension_order : str
            The order of dimensions in the data.
        num_channels : int
            Number of channels.
        num_acquisition_cycles : int
            Number of acquisition cycles (samples).
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
        # Create structured dtype for the table
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

        # Generate file_indices and local_ifd_indices
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
    
    def get_series(self, start_sample: Optional[int], end_sample: Optional[int] = None) -> np.ndarray: 
        """
        Get data as a time series from start_sample to end_sample.

        This method retrieves frames at the specified range from the ScanImage TIFF file(s).
        It uses the mapping created during initialization to efficiently locate and load only
        the requested frames, without loading the entire dataset into memory.

        For volumetric data (multiple planes), the returned array will have an additional dimension
        for the planes. For planar data (single plane), the plane dimension is squeezed out.

        Parameters
        ----------
        start_sample : int
        end_sample : int

        Returns
        -------
        numpy.ndarray
            Array of data with shape (num_samples, height, width) if num_planes is 1,
            or (num_samples, height, width, num_planes) if num_planes > 1.

            For example, for a non-volumetric dataset with 512x512 frames, requesting 3 samples
            would return an array with shape (3, 512, 512).

            For a volumetric dataset with 5 planes and 512x512 frames, requesting 3 samples
            would return an array with shape (3, 512, 512, 5).    
        """
        start_sample = int(start_sample) if start_sample is not None else 0
        end_sample = int(end_sample) if end_sample is not None else self._num_samples

        
        samples_in_series = end_sample - start_sample
        
        # Preallocate output array as volumetric and squeeze if not volumetric before returning
        samples = np.empty((samples_in_series, self._num_rows, self._num_columns, self._num_planes), dtype=self._dtype)

        for return_index, sample_index in enumerate(range(start_sample, end_sample)):
            for depth_position in range(self._num_planes):
                
                # Calculate the index in the mapping table array
                frame_index = sample_index * self._num_planes + depth_position
                table_row = self._frames_to_ifd_table[frame_index]
                file_index = table_row["file_index"]
                ifd_index = table_row["IFD_index"]

                tiff_reader = self._tiff_readers[file_index]
                image_file_directory = tiff_reader.pages[ifd_index]
                samples[return_index, :, :, depth_position] = image_file_directory.asarray()

        # Squeeze the depth dimension if not volumetric
        if not self.is_volumetric:
            samples = samples.squeeze(axis=3)

        return samples
        
    def get_video(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> np.ndarray:
        """Here for backwards compatibility, should be removed at some point."""
        return self.get_series(start_sample=start_frame, end_sample=end_frame)

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return (self._num_rows, self._num_columns)
    
    def get_sample_shape(self):
        """
        Get the shape of a single sample

        Returns       
        -------
        tuple of int
            Shape of a single sample. If the data is volumetric, the shape is hape of a single sample (num_rows, num_columns).
            (num_rows, num_columns, num_planes). Otherwise, the shape is 
            (num_rows, num_columns).
        """
        if self.is_volumetric:
            return (self._num_rows, self._num_columns, self._num_planes)
        else:
            return (self._num_rows, self._num_columns)
        
    def get_num_samples(self) -> int:
        """Get the number of samples in the video.

        Returns
        -------
        int
            Number of samples in the video.
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

    @staticmethod
    def get_channel_names(file_path: PathType) -> list:
        """Get the channel names available in a ScanImage TIFF file.

        This static method extracts the channel names from a ScanImage TIFF file
        without needing to create an extractor instance. This is useful for
        determining which channels are available before creating an extractor.

        Parameters
        ----------
        file_path : PathType
            Path to the ScanImage TIFF file.

        Returns
        -------
        list
            List of channel names available in the file.

        Examples
        --------
        >>> channel_names = ScanImageImagingExtractor.get_channel_names('path/to/file.tif')
        >>> print(f"Available channels: {channel_names}")
        """
        from tifffile import read_scanimage_metadata

        with open(file_path, "rb") as fh:
            all_metadata = read_scanimage_metadata(fh)
            non_varying_frame_metadata = all_metadata[0]

        # `channelSave` indicates whether the channel is saved
        # We check `channelSave` first but keep the `channelsActive` check for backward compatibility
        channel_availability_keys = ["SI.hChannels.channelSave", "SI.hChannels.channelsActive"]
        for channel_availability in channel_availability_keys:
            if channel_availability in non_varying_frame_metadata.keys():
                break

        available_channels = non_varying_frame_metadata[channel_availability]
        available_channels = [available_channels] if not isinstance(available_channels, list) else available_channels
        channel_indices = np.array(available_channels) - 1  # Account for MATLAB indexing
        channel_names = non_varying_frame_metadata["SI.hChannels.channelName"]
        channel_names_available = [channel_names[i] for i in channel_indices]

        return channel_names_available

    def get_dtype(self) -> DtypeType:
        """Get the data type of the video.

        Returns
        -------
        dtype
            Data type of the video.
        """
        return self._dtype

    def get_times(self) -> np.ndarray:
        """Get the timestamps for each frame.

        Returns
        -------
        numpy.ndarray
            Array of timestamps in seconds for each frame.

        Notes
        -----
        This method extracts timestamps from the ScanImage TIFF file(s) for the selected channel.
        It uses the mapping created during initialization to efficiently locate and extract
        timestamps for each sample.
        """
        if self._times is not None:
            return self._times

        # Initialize array to store timestamps
        timestamps = np.zeros(self._num_samples, dtype=np.float64)

        # For each sample, extract its timestamp from the corresponding file and IFD
        for sample_index in range(self._num_samples):
            
            # Get the last frame in this sample to get the timestamps
            frame_index = sample_index * self._num_planes + (self._num_planes - 1)
            table_row = self._frames_to_ifd_table[frame_index]
            file_index = table_row["file_index"]
            ifd_index = table_row["IFD_index"]

            tiff_reader = self._tiff_readers[file_index]
            image_file_directory = tiff_reader.pages[ifd_index]      
            
            # Extract timestamp from the IFD description
            description = image_file_directory.description
            description_lines = description.split("\n")

            # Use iterator pattern to find frameTimestamps_sec
            timestamp_line = next((line for line in description_lines if "frameTimestamps_sec" in line), None)
            
            if timestamp_line is not None:
                # Extract the value part after " = "
                _, value_str = timestamp_line.split(" = ", 1)
                try:
                    timestamps[sample_index] = float(value_str.strip())
                except ValueError:
                    # If parsing fails, use sample index / sampling frequency as fallback
                    timestamps[sample_index] = sample_index / self._sampling_frequency
            else:
                # If no timestamp found, throw a warning and use sample index / sampling frequency as fallback
                warnings.warn(
                    f"No frameTimestamps_sec found for sample {sample_index}. Using calculated timestamp instead.",
                    UserWarning,
                )
                timestamps[sample_index] = sample_index / self._sampling_frequency

        # Cache the timestamps
        self._times = timestamps
        return timestamps

    def get_num_planes(self) -> int:
        """Get the number of depth planes.

        For volumetric data, this returns the number of Z-planes in each volume.
        For planar data, this returns 1.

        Returns
        -------
        int
            Number of depth planes.
        """
        return self._num_planes

    def get_plane_extractor(self, plane_index: int) -> "ImagingExtractor":
        """Extract a specific depth plane from volumetric data.

        This method allows for extracting a specific depth plane from volumetric imaging data,
        returning a modified version of the extractor that only returns data for the specified plane.

        Parameters
        ----------
        plane_index: int
            Index of the depth plane to extract (0-indexed).

        Returns
        -------
        extractor: ImagingExtractor
            A modified version of the extractor that only returns data for the specified plane.

        Raises
        ------
        ValueError
            If the data is not volumetric (has only one plane).
            If plane_index is out of range.

        Examples
        --------
        >>> extractor = ScanImageImagingExtractor('path/to/volumetric_file.tif')
        >>> # Get only the first plane
        >>> first_plane = extractor.depth_slice(plane_index=0)
        >>> # Get the second plane
        >>> second_plane = extractor.depth_slice(plane_index=1)
        """
        if not self.is_volumetric:
            raise ValueError("Cannot depth slice non-volumetric data. This data has only one plane.")

        # Validate parameters
        if plane_index < 0 or plane_index >= self._num_planes:
            raise ValueError(f"plane_index ({plane_index}) must be between 0 and {self._num_planes - 1}")

        # Create a copy of the current extractor
        import copy

        sliced_extractor = copy.deepcopy(self)

        # Filter the frames_to_ifd_table to only include entries for the specified depth plane
        depth_mask = sliced_extractor._frames_to_ifd_table["depth_index"] == plane_index
        sliced_extractor._frames_to_ifd_table = sliced_extractor._frames_to_ifd_table[depth_mask]

        # Update the number of samples
        sliced_extractor._num_samples = len(sliced_extractor._frames_to_ifd_table)

        # Override the is_volumetric flag and num_planes
        sliced_extractor.is_volumetric = False
        sliced_extractor._num_planes = 1

        return sliced_extractor

    def __del__(self):
        """Close file handles when the extractor is garbage collected."""
        if hasattr(self, "_tiff_readers"):
            for handle in self._tiff_readers:
                try:
                    handle.close()
                except Exception:
                    pass


class ScanImageTiffMultiPlaneMultiFileImagingExtractor(MultiImagingExtractor):
    """Specialized extractor for reading multi-file (buffered) TIFF files produced via ScanImage."""

    extractor_name = "ScanImageTiffMultiPlaneMultiFileImaging"
    is_writable = True
    mode = "folder"

    def __init__(
        self, folder_path: PathType, file_pattern: str, channel_name: str, extract_all_metadata: bool = True
    ) -> None:
        """Create a ScanImageTiffMultiPlaneMultiFileImagingExtractor instance from a folder of TIFF files produced by ScanImage.

        Parameters
        ----------
        folder_path : PathType
            Path to the folder containing the TIFF files.
        file_pattern : str
            Pattern for the TIFF files to read -- see pathlib.Path.glob for details.
        channel_name : str
            Channel name for this extractor.
        extract_all_metadata : bool
            If True, extract metadata from every file in the folder. If False, only extract metadata from the first
            file in the folder. The default is True.
        """
        self.folder_path = Path(folder_path)
        from natsort import natsorted

        file_paths = natsorted(self.folder_path.glob(file_pattern))
        if len(file_paths) == 0:
            raise ValueError(f"No files found in folder with pattern: {file_pattern}")

        self.metadata = read_scanimage_metadata(file_paths[0])
        if not extract_all_metadata:
            metadata = self.metadata
            parsed_metadata = self.metadata["roiextractors_parsed_metadata"]
        else:
            metadata, parsed_metadata = None, None
        imaging_extractors = []
        for file_path in file_paths:
            imaging_extractor = ScanImageTiffMultiPlaneImagingExtractor(
                file_path=file_path,
                channel_name=channel_name,
                metadata=metadata,
                parsed_metadata=parsed_metadata,
            )
            imaging_extractors.append(imaging_extractor)

        self._num_planes = imaging_extractors[0].get_num_planes()
        super().__init__(imaging_extractors=imaging_extractors)

    def get_num_planes(self) -> int:
        """Get the number of depth planes.

        Returns
        -------
        _num_planes: int
            The number of depth planes.
        """
        return self._num_planes


class ScanImageTiffSinglePlaneMultiFileImagingExtractor(MultiImagingExtractor):
    """Specialized extractor for reading multi-file (buffered) TIFF files produced via ScanImage."""

    extractor_name = "ScanImageTiffSinglePlaneMultiFileImaging"
    is_writable = True
    mode = "folder"

    def __init__(
        self,
        folder_path: PathType,
        file_pattern: str,
        channel_name: str,
        plane_name: str,
        extract_all_metadata: bool = True,
    ) -> None:
        """Create a ScanImageTiffSinglePlaneMultiFileImagingExtractor instance from a folder of TIFF files produced by ScanImage.

        Parameters
        ----------
        folder_path : PathType
            Path to the folder containing the TIFF files.
        file_pattern : str
            Pattern for the TIFF files to read -- see pathlib.Path.glob for details.
        channel_name : str
            Name of the channel for this extractor.
        plane_name : str
            Name of the plane for this extractor.
        extract_all_metadata : bool
            If True, extract metadata from every file in the folder. If False, only extract metadata from the first
            file in the folder. The default is True.
        """
        self.folder_path = Path(folder_path)
        from natsort import natsorted

        file_paths = natsorted(self.folder_path.glob(file_pattern))
        if len(file_paths) == 0:
            raise ValueError(f"No files found in folder with pattern: {file_pattern}")
        if not extract_all_metadata:
            metadata = read_scanimage_metadata(file_paths[0])
            parsed_metadata = metadata["roiextractors_parsed_metadata"]
        else:
            metadata, parsed_metadata = None, None
        imaging_extractors = []
        for file_path in file_paths:
            imaging_extractor = ScanImageTiffSinglePlaneImagingExtractor(
                file_path=file_path,
                channel_name=channel_name,
                plane_name=plane_name,
                metadata=metadata,
                parsed_metadata=parsed_metadata,
            )
            imaging_extractors.append(imaging_extractor)
        super().__init__(imaging_extractors=imaging_extractors)


class ScanImageTiffMultiPlaneImagingExtractor(VolumetricImagingExtractor):
    """Specialized extractor for reading multi-plane (volumetric) TIFF files produced via ScanImage."""

    extractor_name = "ScanImageTiffMultiPlaneImaging"
    is_writable = True
    mode = "file"

    def __init__(
        self,
        file_path: PathType,
        channel_name: Optional[str] = None,
        metadata: Optional[dict] = None,
        parsed_metadata: Optional[dict] = None,
    ) -> None:
        """Create a ScanImageTiffMultPlaneImagingExtractor instance from a volumetric TIFF file produced by ScanImage.

        Parameters
        ----------
        file_path : PathType
            Path to the TIFF file.
        channel_name : str, optional
            Name of the channel for this extractor. If None, the first channel will be used.
        metadata : dict, optional
            Metadata dictionary. If None, metadata will be extracted from the TIFF file.
        parsed_metadata : dict, optional
            Parsed metadata dictionary. If None, metadata must also be None.

        Notes
        -----
            If metadata is provided, it MUST be in the form outputted by extract_extra_metadata in order to be parsed
            correctly.
        """
        self.file_path = Path(file_path)
        if metadata is None:
            self.metadata = read_scanimage_metadata(file_path)
            self.parsed_metadata = self.metadata["roiextractors_parsed_metadata"]
        else:
            self.metadata = metadata
            assert parsed_metadata is not None, "If metadata is provided, parsed_metadata must also be provided."
            self.parsed_metadata = parsed_metadata
        num_planes = self.parsed_metadata["num_planes"]
        channel_names = self.parsed_metadata["channel_names"]
        if channel_name is None:
            channel_name = channel_names[0]
        imaging_extractors = []
        for plane in range(num_planes):
            imaging_extractor = ScanImageTiffSinglePlaneImagingExtractor(
                file_path=file_path,
                channel_name=channel_name,
                plane_name=str(plane),
                metadata=self.metadata,
                parsed_metadata=self.parsed_metadata,
            )
            imaging_extractors.append(imaging_extractor)
        super().__init__(imaging_extractors=imaging_extractors)
        assert all(
            imaging_extractor.get_num_planes() == self._num_planes for imaging_extractor in imaging_extractors
        ), "All imaging extractors must have the same number of planes."


class ScanImageTiffSinglePlaneImagingExtractor(ImagingExtractor):
    """Specialized extractor for reading TIFF files produced via ScanImage."""

    extractor_name = "ScanImageTiffImaging"
    is_writable = True
    mode = "file"

    @classmethod
    def get_available_channels(cls, file_path):
        """Get the available channel names from a TIFF file produced by ScanImage.

        Parameters
        ----------
        file_path : PathType
            Path to the TIFF file.

        Returns
        -------
        channel_names: list
            List of channel names.
        """
        metadata = extract_extra_metadata(file_path)
        parsed_metadata = parse_metadata(metadata)
        channel_names = parsed_metadata["channel_names"]
        return channel_names

    @classmethod
    def get_available_planes(cls, file_path):
        """Get the available plane names from a TIFF file produced by ScanImage.

        Parameters
        ----------
        file_path : PathType
            Path to the TIFF file.

        Returns
        -------
        plane_names: list
            List of plane names.
        """
        metadata = extract_extra_metadata(file_path)
        parsed_metadata = parse_metadata(metadata)
        num_planes = parsed_metadata["num_planes"]
        plane_names = [f"{i}" for i in range(num_planes)]
        return plane_names

    def __init__(
        self,
        file_path: PathType,
        channel_name: str,
        plane_name: str,
        metadata: Optional[dict] = None,
        parsed_metadata: Optional[dict] = None,
    ) -> None:
        """Create a ScanImageTiffImagingExtractor instance from a TIFF file produced by ScanImage.

        The underlying data is stored in a round-robin format collapsed into 3 dimensions (frames, rows, columns).
        I.e. the first frame of each channel and each plane is stored, and then the second frame of each channel and
        each plane, etc.
        If framesPerSlice > 1, then multiple frames are acquired per slice before moving to the next slice.
        Ex. for 2 channels, 2 planes, and 2 framesPerSlice:
        ```
        [channel_1_plane_1_frame_1, channel_2_plane_1_frame_1, channel_1_plane_1_frame_2, channel_2_plane_1_frame_2,
         channel_1_plane_2_frame_1, channel_2_plane_2_frame_1, channel_1_plane_2_frame_2, channel_2_plane_2_frame_2,
         channel_1_plane_1_frame_3, channel_2_plane_1_frame_3, channel_1_plane_1_frame_4, channel_2_plane_1_frame_4,
         channel_1_plane_2_frame_3, channel_2_plane_2_frame_3, channel_1_plane_2_frame_4, channel_2_plane_2_frame_4, ...
         channel_1_plane_1_frame_N, channel_2_plane_1_frame_N, channel_1_plane_2_frame_N, channel_2_plane_2_frame_N]
        ```
        This file structured is accessed by ScanImageTiffImagingExtractor for a single channel and plane.

        Parameters
        ----------
        file_path : PathType
            Path to the TIFF file.
        channel_name : str
            Name of the channel for this extractor (default=None).
        plane_name : str
            Name of the plane for this extractor (default=None).
        metadata : dict, optional
            Metadata dictionary. If None, metadata will be extracted from the TIFF file.
        parsed_metadata : dict, optional
            Parsed metadata dictionary. If None, metadata must also be None.

        Notes
        -----
            If metadata is provided, it MUST be in the form outputted by extract_extra_metadata in order to be parsed
            correctly.
        """
        self.file_path = Path(file_path)
        if metadata is None:
            self.metadata = read_scanimage_metadata(file_path)
            self.parsed_metadata = self.metadata["roiextractors_parsed_metadata"]
        else:
            self.metadata = metadata
            assert parsed_metadata is not None, "If metadata is provided, parsed_metadata must also be provided."
            self.parsed_metadata = parsed_metadata
        self._sampling_frequency = self.parsed_metadata["sampling_frequency"]
        self._num_channels = self.parsed_metadata["num_channels"]
        self._num_planes = self.parsed_metadata["num_planes"]
        self._frames_per_slice = self.parsed_metadata["frames_per_slice"]
        self._channel_names = self.parsed_metadata["channel_names"]
        self._plane_names = [f"{i}" for i in range(self._num_planes)]
        self.channel_name = channel_name
        self.plane_name = plane_name
        if channel_name not in self._channel_names:
            raise ValueError(f"Channel name ({channel_name}) not found in channel names ({self._channel_names}).")
        self.channel = self._channel_names.index(channel_name)
        if plane_name not in self._plane_names:
            raise ValueError(f"Plane name ({plane_name}) not found in plane names ({self._plane_names}).")
        self.plane = self._plane_names.index(plane_name)

        ScanImageTiffReader = _get_scanimage_reader()
        with ScanImageTiffReader(str(self.file_path)) as io:
            shape = io.shape()  # [frames, rows, columns]
        if len(shape) == 2:  # [rows, columns]
            raise NotImplementedError(
                "Extractor cannot handle single-frame ScanImageTiff data. Please raise an issue to request this feature: "
                "https://github.com/catalystneuro/roiextractors/issues "
            )
        elif len(shape) == 3:
            self._total_num_frames, self._num_rows, self._num_columns = shape
            if (
                self._num_planes == 1
            ):  # For single plane data, framesPerSlice sometimes is set to total number of frames
                self._frames_per_slice = 1
            self._num_raw_per_plane = self._frames_per_slice * self._num_channels
            self._num_raw_per_cycle = self._num_raw_per_plane * self._num_planes
            self._num_samples = self._total_num_frames // (self._num_planes * self._num_channels)
            self._num_cycles = self._total_num_frames // self._num_raw_per_cycle
        else:
            raise NotImplementedError(
                "Extractor cannot handle 4D ScanImageTiff data. Please raise an issue to request this feature: "
                "https://github.com/catalystneuro/roiextractors/issues "
            )
        timestamps = extract_timestamps_from_file(file_path)
        index = [self.frame_to_raw_index(iframe) for iframe in range(self._num_samples)]
        self._times = timestamps[index]

    def get_frames(self, frame_idxs: ArrayType) -> np.ndarray:
        """Get specific video frames from indices (not necessarily continuous).

        Parameters
        ----------
        frame_idxs: array-like
            Indices of frames to return.

        Returns
        -------
        frames: numpy.ndarray
            The video frames.
        """
        if isinstance(frame_idxs, int):
            frame_idxs = [frame_idxs]
        self.check_frame_inputs(frame_idxs[-1])

        if not all(np.diff(frame_idxs) == 1):
            return np.concatenate([self._get_single_frame(frame=idx) for idx in frame_idxs])
        else:
            return self.get_video(start_frame=frame_idxs[0], end_frame=frame_idxs[-1] + 1)

    # Data accessed through an open ScanImageTiffReader io gets scrambled if there are multiple calls.
    # Thus, open fresh io in context each time something is needed.
    def _get_single_frame(self, frame: int) -> np.ndarray:
        """Get a single frame of data from the TIFF file.

        Parameters
        ----------
        frame : int
            The index of the frame to retrieve.

        Returns
        -------
        frame: numpy.ndarray
            The frame of data.
        """
        self.check_frame_inputs(frame)
        ScanImageTiffReader = _get_scanimage_reader()
        raw_index = self.frame_to_raw_index(frame)
        with ScanImageTiffReader(str(self.file_path)) as io:
            return io.data(beg=raw_index, end=raw_index + 1)

    def get_video(self, start_frame=None, end_frame=None) -> np.ndarray:
        """Get the video frames.

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
        """
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self._num_samples
        end_frame_inclusive = end_frame - 1
        self.check_frame_inputs(end_frame_inclusive)
        self.check_frame_inputs(start_frame)
        raw_start = self.frame_to_raw_index(start_frame)
        raw_end_inclusive = self.frame_to_raw_index(end_frame_inclusive)  # frame_to_raw_index requires inclusive frame
        raw_end = raw_end_inclusive + 1

        ScanImageTiffReader = _get_scanimage_reader()
        with ScanImageTiffReader(filename=str(self.file_path)) as io:
            raw_video = io.data(beg=raw_start, end=raw_end)

        start_cycle = np.ceil(start_frame / self._frames_per_slice).astype("int")
        end_cycle = end_frame // self._frames_per_slice
        num_cycles = end_cycle - start_cycle
        start_frame_in_cycle = start_frame % self._frames_per_slice
        end_frame_in_cycle = end_frame % self._frames_per_slice
        start_left_in_cycle = (self._frames_per_slice - start_frame_in_cycle) % self._frames_per_slice
        end_left_in_cycle = (self._frames_per_slice - end_frame_in_cycle) % self._frames_per_slice
        index = []
        for j in range(start_left_in_cycle):  # Add remaining frames from first (incomplete) cycle
            index.append(j * self._num_channels)
        for i in range(num_cycles):
            for j in range(self._frames_per_slice):
                index.append(
                    (j - start_frame_in_cycle) * self._num_channels
                    + (i + bool(start_left_in_cycle)) * self._num_raw_per_cycle
                )
        for j in range(end_left_in_cycle):  # Add remaining frames from last (incomplete) cycle)
            index.append((j - start_frame_in_cycle) * self._num_channels + num_cycles * self._num_raw_per_cycle)
        video = raw_video[index]
        return video

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return (self._num_rows, self._num_columns)

    def get_image_size(self) -> Tuple[int, int]:
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        return (self._num_rows, self._num_columns)

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

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_channel_names(self) -> list:
        return self._channel_names

    def get_num_channels(self) -> int:
        return self._num_channels

    def get_num_planes(self) -> int:
        """Get the number of depth planes.

        Returns
        -------
        _num_planes: int
            The number of depth planes.
        """
        return self._num_planes

    def get_dtype(self) -> DtypeType:
        return self.get_frames(0).dtype

    def check_frame_inputs(self, frame) -> None:
        """Check that the frame index is valid. Raise ValueError if not.

        Parameters
        ----------
        frame : int
            The index of the frame to retrieve.

        Raises
        ------
        ValueError
            If the frame index is invalid.
        """
        if frame >= self._num_samples:
            raise ValueError(f"Frame index ({frame}) exceeds number of frames ({self._num_samples}).")
        if frame < 0:
            raise ValueError(f"Frame index ({frame}) must be greater than or equal to 0.")

    def frame_to_raw_index(self, frame: int) -> int:
        """Convert a frame index to the raw index in the TIFF file.

        Parameters
        ----------
        frame : int
            The index of the frame to retrieve.

        Returns
        -------
        raw_index: int
            The raw index of the frame in the TIFF file.

        Notes
        -----
        The underlying data is stored in a round-robin format collapsed into 3 dimensions (frames, rows, columns).
        I.e. the first frame of each channel and each plane is stored, and then the second frame of each channel and
        each plane, etc.
        If framesPerSlice > 1, then multiple frames are acquired per slice before moving to the next slice.
        Ex. for 2 channels, 2 planes, and 2 framesPerSlice:
        ```
        [channel_1_plane_1_frame_1, channel_2_plane_1_frame_1, channel_1_plane_1_frame_2, channel_2_plane_1_frame_2,
         channel_1_plane_2_frame_1, channel_2_plane_2_frame_1, channel_1_plane_2_frame_2, channel_2_plane_2_frame_2,
         channel_1_plane_1_frame_3, channel_2_plane_1_frame_3, channel_1_plane_1_frame_4, channel_2_plane_1_frame_4,
         channel_1_plane_2_frame_3, channel_2_plane_2_frame_3, channel_1_plane_2_frame_4, channel_2_plane_2_frame_4, ...
         channel_1_plane_1_frame_N, channel_2_plane_1_frame_N, channel_1_plane_2_frame_N, channel_2_plane_2_frame_N]
        ```
        """
        cycle = frame // self._frames_per_slice
        frame_in_cycle = frame % self._frames_per_slice
        raw_index = (
            cycle * self._num_raw_per_cycle
            + self.plane * self._num_raw_per_plane
            + frame_in_cycle * self._num_channels
            + self.channel
        )
        return raw_index


class ScanImageTiffImagingExtractor(ImagingExtractor):  # TODO: Remove this extractor on or after December 2023
    """Specialized extractor for reading TIFF files produced via ScanImage.

    This implementation is for legacy purposes and is not recommended for use.
    Please use ScanImageTiffSinglePlaneImagingExtractor or ScanImageTiffMultiPlaneImagingExtractor instead.
    """

    extractor_name = "ScanImageTiffImaging"
    is_writable = True
    mode = "file"

    def __init__(
        self,
        file_path: PathType,
        sampling_frequency: FloatType,
    ):
        """Create a ScanImageTiffImagingExtractor instance from a TIFF file produced by ScanImage.

        This extractor allows for lazy accessing of slices, unlike
        :py:class:`~roiextractors.extractors.tiffimagingextractors.TiffImagingExtractor`.
        However, direct slicing of the underlying data structure is not equivalent to a numpy memory map.

        Parameters
        ----------
        file_path : PathType
            Path to the TIFF file.
        sampling_frequency : float
            The frequency at which the frames were sampled, in Hz.
        """
        deprecation_message = """
        This extractor is being deprecated on or after December 2023 in favor of
        ScanImageTiffMultiPlaneImagingExtractor or ScanImageTiffSinglePlaneImagingExtractor.  Please use one of these
        extractors instead.
        """
        warn(deprecation_message, category=FutureWarning)
        ScanImageTiffReader = _get_scanimage_reader()

        super().__init__()
        self.file_path = Path(file_path)
        self._sampling_frequency = sampling_frequency
        valid_suffixes = [".tiff", ".tif", ".TIFF", ".TIF"]
        if self.file_path.suffix not in valid_suffixes:
            suffix_string = ", ".join(valid_suffixes[:-1]) + f", or {valid_suffixes[-1]}"
            warn(
                f"Suffix ({self.file_path.suffix}) is not of type {suffix_string}! "
                f"The {self.extractor_name}Extractor may not be appropriate for the file."
            )

        with ScanImageTiffReader(str(self.file_path)) as io:
            shape = io.shape()  # [frames, rows, columns]
        if len(shape) == 3:
            self._num_samples, self._num_rows, self._num_columns = shape
            self._num_channels = 1
        else:  # no example file for multiple color channels or depths
            raise NotImplementedError(
                "Extractor cannot handle 4D TIFF data. Please raise an issue to request this feature: "
                "https://github.com/catalystneuro/roiextractors/issues "
            )

    def get_frames(self, frame_idxs: ArrayType, channel: int = 0) -> np.ndarray:
        """Get specific video frames from indices.

        Parameters
        ----------
        frame_idxs: array-like
            Indices of frames to return.
        channel: int, optional
            Channel index. Deprecated: This parameter will be removed in August 2025.

        Returns
        -------
        frames: numpy.ndarray
            The video frames.
        """
        if channel != 0:
            warn(
                "The 'channel' parameter in get_frames() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )

        ScanImageTiffReader = _get_scanimage_reader()
        squeeze_data = False
        if isinstance(frame_idxs, int):
            squeeze_data = True
            frame_idxs = [frame_idxs]

        if not all(np.diff(frame_idxs) == 1):
            return np.concatenate([self._get_single_frame(idx=idx) for idx in frame_idxs])
        else:
            with ScanImageTiffReader(filename=str(self.file_path)) as io:
                frames = io.data(beg=frame_idxs[0], end=frame_idxs[-1] + 1)
                if squeeze_data:
                    frames = frames.squeeze()
            return frames

    # Data accessed through an open ScanImageTiffReader io gets scrambled if there are multiple calls.
    # Thus, open fresh io in context each time something is needed.
    def _get_single_frame(self, idx: int) -> np.ndarray:
        """Get a single frame of data from the TIFF file.

        Parameters
        ----------
        idx : int
            The index of the frame to retrieve.

        Returns
        -------
        frame: numpy.ndarray
            The frame of data.
        """
        ScanImageTiffReader = _get_scanimage_reader()

        with ScanImageTiffReader(str(self.file_path)) as io:
            return io.data(beg=idx, end=idx + 1)

    def get_video(self, start_frame=None, end_frame=None, channel: Optional[int] = 0) -> np.ndarray:
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
        """
        if channel != 0:
            warn(
                "The 'channel' parameter in get_video() is deprecated and will be removed in August 2025.",
                DeprecationWarning,
                stacklevel=2,
            )

        ScanImageTiffReader = _get_scanimage_reader()
        with ScanImageTiffReader(filename=str(self.file_path)) as io:
            return io.data(beg=start_frame, end=end_frame)

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return (self._num_rows, self._num_columns)

    def get_image_size(self) -> Tuple[int, int]:
        warnings.warn(
            "get_image_size() is deprecated and will be removed in or after September 2025. "
            "Use get_image_shape() instead for consistent behavior across all extractors.",
            DeprecationWarning,
            stacklevel=2,
        )
        return (self._num_rows, self._num_columns)

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

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_num_channels(self) -> int:
        return self._num_channels

    def get_channel_names(self) -> list:
        pass
