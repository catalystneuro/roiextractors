"""Specialized extractor for reading TIFF files produced via ScanImage.

Classes
-------
ScanImageLegacyImagingExtractor
    Specialized extractor for reading TIFF files produced via ScanImage.
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple
from warnings import warn

import numpy as np

from .scanimagetiff_utils import (
    _get_scanimage_reader,
    extract_extra_metadata,
    extract_timestamps_from_file,
    parse_metadata,
    read_scanimage_metadata,
)
from ...extraction_tools import ArrayType, DtypeType, FloatType, PathType, get_package
from ...imagingextractor import ImagingExtractor
from ...multiimagingextractor import MultiImagingExtractor
from ...volumetricimagingextractor import VolumetricImagingExtractor


class ScanImageImagingExtractor(ImagingExtractor):
    """
    Specialized extractor for reading TIFF files produced via ScanImage software.

    This extractor is designed to handle the structure of ScanImage TIFF files, which can contain
    multi channel and both planar and volumetric data. It also supports both single-file and multi-file datasets generated
    by ScanImage in various acquisition modes (grab, focus, loop).

    The extractor creates a mapping between each frame in the dataset and its corresponding physical file
    and IFD (Image File Directory) location. This mapping enables efficient retrieval of specific frames
    without loading the entire dataset into memory, making it suitable for large datasets.

    For datasets with multiple frames per slice, either a slice_sample parameter must be provided
    or interleave_slice_samples must be set to True to explicitly opt into interleaving behavior.


    Key features:
    - Handles multi-channel data with channel selection
    - Supports volumetric (multi-plane) imaging data
    - Automatically detects and loads multi-file datasets based on ScanImage naming conventions
    - Extracts and provides access to ScanImage metadata
    - Efficiently retrieves frames using lazy loading
    - Handles flyback frames in volumetric data by ignoring them in the mapping

    """

    extractor_name = "ScanImageImagingExtractor"

    def __init__(
        self,
        file_path: Optional[PathType] = None,
        channel_name: Optional[str] = None,
        file_paths: Optional[list[PathType]] = None,
        slice_sample: Optional[int] = None,
        plane_index: Optional[int] = None,
        interleave_slice_samples: bool = False,
    ):
        """
        Initialize the ScanImageImagingExtractor.

        Parameters
        ----------
        file_path : PathType, optional
            Path to the ScanImage TIFF file. If this is part of a multi-file series, this should be the first file.
            Either `file_path` or `file_paths` must be provided.
        channel_name : str, optional
            Name of the channel to extract (e.g., "Channel 1", "Channel 2").
            - If None and only one channel is available, that channel will be used.
            - If None and multiple channels are available, an error will be raised.
            - Use `get_available_channel_names(file_path)` to see available channels before creating the extractor.
        file_paths : list[PathType], optional
            List of file paths to use. If provided, this overrides the automatic file detection heuristics.
            Use this parameter when:
            - Automatic detection doesn't work correctly
            - You need to specify a custom subset of files
            - You need to control the exact order of files
            The file paths must be provided in the temporal order of the frames in the dataset.
        slice_sample : int, optional
            Controls how to handle multiple frames per slice in volumetric data:
            - If an integer (0 to frames_per_slice-1): Uses only that specific frame for each slice,
              effectively selecting a single sample from each acquisition.
            - If None (default): Requires interleave_slice_samples=True when frames_per_slice > 1.
            - This parameter has no effect when frames_per_slice = 1.
            - Use `get_frames_per_slice(file_path)` to check the number of frames per slice.
        interleave_slice_samples : bool, optional
            Controls whether to interleave all slice samples as separate time points when frames_per_slice > 1:
            - If True: Interleaves all slice samples as separate time points, increasing the effective
              number of samples by frames_per_slice. This treats each slice_sample as a distinct sample.
            - If False (default): Requires a specific slice_sample to be provided when frames_per_slice > 1.
            - This parameter has no effect when frames_per_slice = 1 or when slice_sample is provided.
        plane_index : int, optional
            Must be between 0 and num_planes-1. Used to extract a specific plane from volumetric data.
            When provided:
            - The resulting extractor will be planar (is_volumetric = False)
            - Each sample will contain only data for the specified plane
            - The shape of returned data will be (samples, height, width) instead of (samples, height, width, planes)
            - This parameter has no effect on planar (non-volumetric) data.

        Examples
        --------
        # Basic usage with a single file, single channel
        >>> extractor = ScanImageImagingExtractor(file_path='path/to/file.tif')

        # Multi-channel data, selecting a specific channel
        >>> channel_names = ScanImageImagingExtractor.get_available_channel_names('path/to/file.tif')
        >>> extractor = ScanImageImagingExtractor(file_path='path/to/file.tif', channel_name=channel_names[0])

        # Volumetric data with multiple frames per slice, selecting a specific slice sample
        >>> frames_per_slice = ScanImageImagingExtractor.get_frames_per_slice('path/to/file.tif')
        >>> extractor = ScanImageImagingExtractor(file_path='path/to/file.tif', slice_sample=0)

        # Volumetric data, extracting a specific plane
        >>> extractor = ScanImageImagingExtractor(file_path='path/to/file.tif', plane_index=2)

        # Explicitly specifying multiple files
        >>> extractor = ScanImageImagingExtractor(
        ...     file_paths=['path/to/file1.tif', 'path/to/file2.tif', 'path/to/file3.tif'],
        ...     channel_name='Channel 1'
        ... )
        """
        super().__init__()
        self.file_path = file_paths[0] if file_paths is not None else file_path
        assert self.file_path is not None, "file_path or file_paths must be provided"
        self.file_path = Path(self.file_path)

        # Validate file suffix
        valid_suffixes = [".tiff", ".tif", ".TIFF", ".TIF"]
        if self.file_path.suffix not in valid_suffixes:
            suffix_string = ", ".join(valid_suffixes[:-1]) + f", or {valid_suffixes[-1]}"
            warn(
                f"Suffix ({self.file_path.suffix}) is not of type {suffix_string}! "
                f"The {self.extractor_name} Extractor may not be appropriate for the file."
            )

        # Open the TIFF file
        tifffile = get_package(package_name="tifffile")
        tiff_reader = tifffile.TiffReader(self.file_path)

        self._general_metadata = tiff_reader.scanimage_metadata
        non_valid_metadata = self._general_metadata is None or len(self._general_metadata) == 0
        if non_valid_metadata:
            error_msg = (
                f"Invalid metadata for file with name {file_path.name}. \n"
                "The metadata is either None or empty which probably indicates that the tiff file "
                "Is not a ScanImage file or it could be an older version."
            )
            raise ValueError("Invalid metadata: The metadata is either None or empty.")
        self._metadata = self._general_metadata["FrameData"]

        self._num_rows, self._num_columns = tiff_reader.pages[0].shape
        self._dtype = tiff_reader.pages[0].dtype

        # Check if stack manager is enabled and if there are multiple slices
        # This criteria was confirmed by Lawrence Niu, a developer of ScanImage
        # but we need to also check numSlices > 1 because some planar datasets
        # have SI.hStackManager.enable = True but only one slice
        stack_enabled = self._metadata["SI.hStackManager.enable"]
        num_slices = self._metadata["SI.hStackManager.numSlices"]
        self.is_volumetric = stack_enabled and num_slices > 1
        if self.is_volumetric:
            self._sampling_frequency = self._metadata["SI.hRoiManager.scanVolumeRate"]
            self._num_planes = self._metadata["SI.hStackManager.numSlices"]

            self._frames_per_slice = self._metadata["SI.hStackManager.framesPerSlice"]

            if self._frames_per_slice == 1:
                self._slice_sample = None
            elif slice_sample is not None:
                if not (0 <= slice_sample < self._frames_per_slice):
                    error_msg = f"slice_sample must be between 0 and {self._frames_per_slice - 1} (frames_per_slice - 1), but got {slice_sample}."
                    raise ValueError(error_msg)
                self._slice_sample = slice_sample
            # Case: multiple frames per slice, no slice_sample, but interleaving explicitly enabled
            elif interleave_slice_samples:
                self._slice_sample = None
            # Error case: multiple frames per slice, no slice_sample, interleaving not enabled
            else:
                error_msg = (
                    f"Multiple frames per slice detected ({self._frames_per_slice}), but no slice_sample specified. "
                    f"Either provide a specific slice_sample (0 to {self._frames_per_slice - 1}) or set "
                    f"interleave_slice_samples=True to explicitly opt into interleaving all slice samples as separate time points."
                )
                raise ValueError(error_msg)

            self._frames_per_volume_per_channel = self._metadata["SI.hStackManager.numFramesPerVolume"]
            self._frames_per_volume_with_flyback = self._metadata["SI.hStackManager.numFramesPerVolumeWithFlyback"]

            self.num_flyback_frames_per_channel = (
                self._frames_per_volume_with_flyback - self._frames_per_volume_per_channel
            )
        else:
            self._sampling_frequency = self._metadata["SI.hRoiManager.scanFrameRate"]
            self._num_planes = 1
            self._frames_per_slice = 1
            self.num_flyback_frames_per_channel = 0

        # This piece of the metadata is the indication that the channel is saved on the data
        channels_available = self._metadata["SI.hChannels.channelSave"]
        channels_available = [channels_available] if isinstance(channels_available, int) else channels_available
        self._num_channels = len(channels_available)

        # Determine their name and use matlab 1-indexing
        all_channel_names = self._metadata["SI.hChannels.channelName"]
        self.channel_names = [all_channel_names[channel_index - 1] for channel_index in channels_available]

        # Channel selection checks
        self._is_multi_channel_data = len(self.channel_names) > 1
        if self._is_multi_channel_data and channel_name is None:

            error_msg = (
                f"Multiple channels available in the data {self.channel_names}"
                "Please specify a channel name to extract data from."
            )
            raise ValueError(error_msg)
        elif self._is_multi_channel_data and channel_name is not None:
            if channel_name not in self.channel_names:
                error_msg = (
                    f"Channel name ({channel_name}) not found in available channels ({self.channel_names}). "
                    "Please specify a valid channel name."
                )
                raise ValueError(error_msg)

            self.channel_name = channel_name
            self._channel_index = self.channel_names.index(channel_name)
        else:  # single channel data

            self.channel_name = self.channel_names[0]
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

        # Calculate total IFDs and samples
        self._ifds_per_file = [len(tiff_reader.pages) for tiff_reader in self._tiff_readers]

        # Note that this includes all the frames for all the channels including flyback frames
        self._num_frames_in_dataset = sum(self._ifds_per_file)

        image_frames_per_cycle = self._num_planes * self._num_channels * self._frames_per_slice
        total_frames_per_cycle = image_frames_per_cycle + self.num_flyback_frames_per_channel * self._num_channels

        # Note that the acquisition might end without completing the last cycle and we discard those frames
        num_acquisition_cycles = self._num_frames_in_dataset // (total_frames_per_cycle)

        #  Every cycle is a full channel sample either volume or planar
        self._num_samples = num_acquisition_cycles

        # Map IFDs and files to frames, channel, depth, and acquisition cycle
        full_frames_to_ifds_table = self._create_frame_to_ifd_table(
            num_channels=self._num_channels,
            num_planes=self._num_planes,
            num_acquisition_cycles=num_acquisition_cycles,
            num_frames_per_slice=self._frames_per_slice,
            num_flyback_frames_per_channel=self.num_flyback_frames_per_channel,
            ifds_per_file=self._ifds_per_file,
        )

        # Filter mapping for the specified channel
        channel_mask = full_frames_to_ifds_table["channel_index"] == self._channel_index
        channel_frames_to_ifd_table = full_frames_to_ifds_table[channel_mask]

        self._frames_to_ifd_table = channel_frames_to_ifd_table

        # Filter mapping for the specified slice_sample or reorder for all slice samples
        if self.is_volumetric and interleave_slice_samples:

            # Re-order to interleave samples from different slice_samples
            # For each acquisition cycle, include all slice_samples in sequence
            sorted_indices = np.lexsort(
                (
                    channel_frames_to_ifd_table["depth_index"],
                    channel_frames_to_ifd_table["slice_sample_index"],
                    channel_frames_to_ifd_table["acquisition_cycle_index"],
                )
            )
            self._frames_to_ifd_table = channel_frames_to_ifd_table[sorted_indices]

            # Adjust the number of samples to account for interleaving of slice samples
            # Each acquisition cycle now produces frames_per_slice x samples (one for each slice_sample)
            self._num_samples = self._num_samples * self._frames_per_slice

        if self.is_volumetric and self._slice_sample is not None:
            # Filter for the specified slice_sample
            slice_sample_mask = channel_frames_to_ifd_table["slice_sample_index"] == self._slice_sample
            self._frames_to_ifd_table = channel_frames_to_ifd_table[slice_sample_mask]

        # Finally, if a planar extractor is requested, we filter the samples for that plane
        if self.is_volumetric and plane_index is not None:
            # Validate plane_index
            if plane_index < 0 or plane_index >= self._num_planes:
                raise ValueError(f"plane_index ({plane_index}) must be between 0 and {self._num_planes - 1}")

            # Filter the frames_to_ifd_table to only include entries for the specified depth plane
            depth_mask = self._frames_to_ifd_table["depth_index"] == plane_index
            self._frames_to_ifd_table = self._frames_to_ifd_table[depth_mask]

            # Override the is_volumetric flag and num_planes
            self.is_volumetric = False
            self._num_planes = 1

    @staticmethod
    def _create_frame_to_ifd_table(
        num_channels: int,
        num_planes: int,
        num_acquisition_cycles: int,
        ifds_per_file: list[int],
        num_frames_per_slice: int = 1,
        num_flyback_frames_per_channel: int = 0,
    ) -> np.ndarray:
        """
        Create a table that describes the data layout of the dataset.

        Every row in the table corresponds to a frame in the dataset and contains:
        - file_index: The index of the file in the series
        - IFD_index: The index of the IFD in the file
        - channel_index: The index of the channel
        - depth_index: The index of the depth
        - acquisition_cycle_index: The index of the time

        The table is represented as a structured numpy array that maps each combination of time,
        channel, and depth to its corresponding physical location in the TIFF files.

        Parameters
        ----------
        num_channels : int
            Number of channels.
        num_planes: int
            The number of planes which corresponds to the depth index or the number of frames per volume
            per channel.
        num_acquisition_cycles : int
            Number of acquisition cycles. For ScanImage, this is the number of samples.
        ifds_per_file : list[int]
            Number of IFDs in each file.
        num_frames_per_slice : int
            Number of frames per slice. This is used to determine the slice_sample index.
        num_flyback_frames_per_channel : int
            Number of flyback frames.

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
                ("slice_sample_index", np.uint8),
                ("acquisition_cycle_index", np.uint16),
            ]
        )

        # Calculate total number of entries
        image_frames_per_cycle = num_planes * num_frames_per_slice * num_channels
        flyback_frames = num_flyback_frames_per_channel * num_channels
        total_frames_per_cycle = image_frames_per_cycle + flyback_frames

        # Generate global ifd indices for complete cycles only
        # This ensures we only include frames from complete acquisition cycles
        num_frames_in_complete_cycles = num_acquisition_cycles * total_frames_per_cycle
        global_ifd_indices = np.arange(num_frames_in_complete_cycles, dtype=np.uint32)

        # We need to filter out the flyback frames, we create an index within each acquisition cycle
        # And then filter out the non-image frames (flyback frames)
        index_in_acquisition_cycle = global_ifd_indices % total_frames_per_cycle
        is_imaging_frame = index_in_acquisition_cycle < image_frames_per_cycle

        global_ifd_indices = global_ifd_indices[is_imaging_frame]
        index_in_acquisition_cycle = index_in_acquisition_cycle[is_imaging_frame]

        # To find their file index we need file boundaries
        file_boundaries = np.zeros(len(ifds_per_file) + 1, dtype=np.uint32)
        file_boundaries[1:] = np.cumsum(ifds_per_file)

        # Find which file each global index belongs to
        file_indices = np.searchsorted(file_boundaries, global_ifd_indices, side="right") - 1

        # Now, we offset the global IFD indices by the starting position of the file
        # to get local IFD indices that start at 0 for each file
        ifd_indices = global_ifd_indices - file_boundaries[file_indices]

        # Calculate indices for each dimension based on the frame position within the cycle
        # For ScanImage, the order is always CZT which means that the channel index comes first,
        # followed by the frames per slice, then depth and finally the acquisition cycle
        channel_indices = index_in_acquisition_cycle % num_channels
        slice_sample_indices = (index_in_acquisition_cycle // num_channels) % num_frames_per_slice
        depth_indices = (index_in_acquisition_cycle // (num_channels * num_frames_per_slice)) % num_planes
        acquisition_cycle_indices = global_ifd_indices // total_frames_per_cycle

        # Create the structured array with the correct size (number of imaging frames after filtering)
        mapping = np.zeros(len(global_ifd_indices), dtype=mapping_dtype)
        mapping["file_index"] = file_indices
        mapping["IFD_index"] = ifd_indices
        mapping["channel_index"] = channel_indices
        mapping["slice_sample_index"] = slice_sample_indices
        mapping["depth_index"] = depth_indices
        mapping["acquisition_cycle_index"] = acquisition_cycle_indices

        return mapping

    def _find_data_files(self) -> list[PathType]:
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

        This method also checks for missing files in the sequence and warns the user if any are detected.
        It also identifies and removes files with non-integer indices, warning the user that they can be
        included explicitly using the file_paths parameter.

        This information about ScanImage file naming was shared in a private conversation with
        Lawrence Niu, who is a developer of ScanImage.

        Returns
        -------
        list[PathType]
            list of paths to all files in the series, sorted naturally (e.g., file_1, file_2, file_10)
        """
        # Parse the file name to extract base name, acquisition number, and file index
        file_stem = self.file_path.stem

        # Can be grab, focus or loop, see
        # https://docs.scanimage.org/Basic+Features/Acquisitions.html
        acquisition_state = self._metadata["SI.acqState"]
        frames_per_file = self._metadata["SI.hScan2D.logFramesPerFile"]
        stack_mode = self._metadata["SI.hStackManager.stackMode"]
        extension = self.file_path.suffix
        # This is the happy path that is well specified in the documentation
        if acquisition_state == "grab" and frames_per_file != float("inf"):
            name_parts = file_stem.split("_")
            base_name, acquisition, file_index = "_".join(name_parts[:-2]), name_parts[-2], name_parts[-1]
            pattern_prefix = f"{base_name}_{acquisition}_"
        # Looped acquisitions also divides the files according to Lawrence Niu in private conversation
        elif acquisition_state == "loop":  # This also separates the files
            base_name = "_".join(file_stem.split("_")[:-1])  # Everything before the last _
            pattern_prefix = f"{base_name}_"
        # This also divided the files according to Lawrence Niu in private conversation
        elif stack_mode == "slow" and self.is_volumetric:
            base_name = "_".join(file_stem.split("_")[:-1])  # Everything before the last _
            pattern_prefix = f"{base_name}_"
        else:
            file_paths_found = [self.file_path]
            return file_paths_found

        from natsort import natsorted

        glob_pattern = f"{pattern_prefix}*{extension}"
        file_paths_found = natsorted(self.file_path.parent.glob(glob_pattern))

        # Early return if only one file is found
        if len(file_paths_found) == 1:
            return file_paths_found

        file_paths_found_filtered = self._check_for_missing_and_excess_files(
            file_paths_found,
            pattern_prefix,
        )

        return file_paths_found_filtered

    def _check_for_missing_and_excess_files(
        self,
        file_paths_found: list[PathType],
        pattern_prefix: str,
    ) -> list[PathType]:
        """Check for missing and/or excess files in the sequences of files that was found."""
        # Extract the varying part from each filename using the pattern_prefix
        suffix = self.file_path.suffix
        excess_files = []
        valid_indices = []
        valid_file_paths = []

        # First we exclude excess files that are not part of the sequence
        for file_path in file_paths_found:
            file_name = file_path.name
            # Extract the part between the pattern_prefix and suffix
            varying_part = file_name[len(pattern_prefix) : -len(suffix)]
            if varying_part.isdigit():
                file_index = int(varying_part)
                valid_indices.append(file_index)
                valid_file_paths.append(file_path)
            else:
                excess_files.append(file_name)
                continue

        # Warn about files that don't belong in the sequence
        if excess_files:
            warnings.warn(
                f"Non-sequence files detected: {', '.join(excess_files)}. "
                f"These files will be excluded from the dataset. "
                f"If you need to include these files, use the file_paths parameter.",
                UserWarning,
            )

        # Check for gaps in the sequence
        if len(valid_indices) > 1:
            valid_indices.sort()
            min_index = min(valid_indices)
            max_index = max(valid_indices)
            expected_indices = set(range(min_index, max_index + 1))
            missing_indices = expected_indices - set(valid_indices)

            if missing_indices:
                # Determine the format of the index (e.g., 00001, 01, etc.)
                # by looking at the first file's index format
                first_file = file_paths_found[0]
                varying_part = first_file.name[len(pattern_prefix) : -len(suffix)]

                # Format the missing file names
                missing_files = []
                for index in missing_indices:
                    # Format the index with the same number of digits
                    formatted_index = f"{index:0{len(varying_part)}d}"
                    missing_file = f"{pattern_prefix}{formatted_index}{suffix}"
                    missing_files.append(missing_file)

                warnings.warn(
                    f"Missing files detected in the sequence: {', '.join(missing_files)}. "
                    f"This may affect data integrity and analysis results.",
                    UserWarning,
                )

        return valid_file_paths

    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
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
        end_sample = int(end_sample) if end_sample is not None else self.get_num_samples()

        samples_in_series = end_sample - start_sample

        # Preallocate output array as volumetric and squeeze if not volumetric before returning
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

                tiff_reader = self._tiff_readers[file_index]
                image_file_directory = tiff_reader.pages[ifd_index]
                samples[return_index, :, :, depth_position] = image_file_directory.asarray()

        # Squeeze the depth dimension if not volumetric
        if not self.is_volumetric:
            samples = samples.squeeze(axis=3)

        return samples

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        tuple
            Shape of the video frame (num_rows, num_columns).
        """
        return (self._num_rows, self._num_columns)

    def get_frame_shape(self) -> Tuple[int, int]:
        """Get the shape of a single frame (num_rows, num_columns).

        Returns
        -------
        tuple
            Shape of a single frame (num_rows, num_columns).
        """
        return (self._num_rows, self._num_columns)

    def get_sample_shape(self):
        """
        Get the shape of a sample.

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

    def get_volume_shape(self) -> Tuple[int, int, int]:
        """Get the shape of a single volume (num_rows, num_columns, num_planes).

        Returns
        -------
        tuple
            Shape of a single volume (num_rows, num_columns, num_planes).
        """
        return (self._num_rows, self._num_columns, self._num_planes)

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

    def get_channel_names(self):
        return self.channel_names

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

    @staticmethod
    def get_available_channel_names(file_path: PathType) -> list:
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
            list of channel names available in the file.

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
        num_samples = self.get_num_samples()
        num_planes = self.get_num_planes()
        timestamps = np.zeros(num_samples, dtype=np.float64)

        # For each sample, extract its timestamp from the corresponding file and IFD
        for sample_index in range(num_samples):

            # Get the last frame in this sample to get the timestamps
            frame_index = sample_index * num_planes + (num_planes - 1)
            table_row = self._frames_to_ifd_table[frame_index]
            file_index = table_row["file_index"]
            ifd_index = table_row["IFD_index"]

            tiff_reader = self._tiff_readers[file_index]
            image_file_directory = tiff_reader.pages[ifd_index]

            # Extract timestamp using the static method
            timestamp = self.extract_timestamp_from_page(image_file_directory)

            if timestamp is not None:
                timestamps[sample_index] = timestamp
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

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Retrieve the original unaltered timestamps for the data in this interface.

        Parameters
        ----------
        start_sample : int, optional
            The starting sample index. If None, starts from the beginning.
        end_sample : int, optional
            The ending sample index. If None, goes to the end.

        Returns
        -------
        timestamps: numpy.ndarray or None
            The timestamps for the data stream, or None if native timestamps are not available.
        """
        timestamps = self.get_times()
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = len(timestamps)
        return timestamps[start_sample:end_sample]

    @staticmethod
    def extract_timestamp_from_page(page) -> float:
        """
        Extract timestamp from a ScanImage TIFF page.

        Parameters
        ----------
        page : tifffile.TiffPage
            The TIFF page to extract the timestamp from.

        Returns
        -------
        float
            The timestamp in seconds or None if no timestamp is found.
        """
        if "ImageDescription" not in page.tags:
            return None

        description = page.tags["ImageDescription"].value
        description_lines = description.split("\n")

        # Find the frameTimestamps_sec line
        timestamp_line = next((line for line in description_lines if "frameTimestamps_sec" in line), None)

        if timestamp_line is not None:
            # Extract the value part after " = "
            _, value_str = timestamp_line.split(" = ", 1)
            try:
                timestamp = float(value_str.strip())
                return timestamp
            except ValueError:
                return None

        return None

    @staticmethod
    def get_available_num_planes(file_path: PathType) -> int:
        """
        Get the number of depth planes from a ScanImage TIFF file.

        For volumetric data, this returns the number of Z-planes in each volume.
        For planar data, this returns 1.

        Parameters
        ----------
        file_path : PathType
            Path to the ScanImage TIFF file.

        Returns
        -------
        int
            Number of depth planes.

        """
        from tifffile import read_scanimage_metadata

        with open(file_path, "rb") as fh:
            all_metadata = read_scanimage_metadata(fh)
            non_varying_frame_metadata = all_metadata[0]

        num_planes = non_varying_frame_metadata.get("SI.hStackManager.numSlices", 1)
        return num_planes

    @staticmethod
    def get_frames_per_slice(file_path: PathType) -> int:
        """
        Get the number of frames per slice from a ScanImage TIFF file.

        ScanImage can sample multiple frames per each slice.

        Parameters
        ----------
        file_path : PathType
            Path to the ScanImage TIFF file.

        Returns
        -------
        int
            Number of frames per slice.

        """
        from tifffile import read_scanimage_metadata

        with open(file_path, "rb") as fh:
            all_metadata = read_scanimage_metadata(fh)
            non_varying_frame_metadata = all_metadata[0]

        frames_per_slice = non_varying_frame_metadata.get("SI.hStackManager.framesPerSlice", 1)
        return frames_per_slice

    def get_original_frame_indices(self, plane_index: Optional[int] = None) -> np.ndarray:
        """
        Get the original frame indices for each sample.

        Returns the index of the original frame for each sample, mapping processed samples
        back to their corresponding frames in the raw microscopy data. This accounts for
        any filtering, subsampling, or exclusions (such as flyback frames) performed by
        the extractor.

        Parameters
        ----------
        plane_index : int, optional
            Which plane to use for frame index calculation in volumetric data.
            If None, plane_index is set to the last plane in the volume. This is because the timestamp of the acquisition of the last plane in a volume is typically set as the timestamp of the volume as a whole. It must be less than the total number of planes.

        Returns
        -------
        np.ndarray
            Array of original frame indices (dtype: int64) with length equal to the
            number of samples. Each element represents the index of the original
            microscopy frame that corresponds to that sample.

        Notes
        -----
        **Frame Index Calculation:**

        - **Planar data**: Frame indices are sequential (0, 1, 2, ...)
        - **Multi-channel data**: Accounts for channel interleaving
        - **Volumetric data**: Uses the specified plane (default: last plane)
        - **Multi-file data**: Includes file offsets for global indexing
        - **Flyback frames**: Automatically excluded from indexing

        **Common Use Cases:**

        - Synchronizing with external timing systems
        - Mapping back to original acquisition timestamps
        - Data provenance and traceability
        - Cross-referencing with raw data files

        **Examples:**

        For a 3-sample volumetric dataset with 5 planes per volume:
        - Default behavior returns indices [4, 9, 14] (last plane of each volumetric sample)
        - With plane_index=0 returns indices [0, 5, 10] (first plane of each volumetric sample)
        """
        num_planes = self.get_num_planes()
        if plane_index is not None:
            assert plane_index < num_planes, f"Plane index {plane_index} exceeds number of planes {num_planes}."
        else:
            plane_index = num_planes - 1

        # Initialize array to store timestamps
        num_samples = self.get_num_samples()
        frame_indices = np.zeros(num_samples, dtype=np.int64)

        # For each sample, extract its timestamp from the corresponding file and IFD
        for sample_index in range(num_samples):

            # Get the last frame in this sample to get the timestamps
            frame_index = sample_index * num_planes + plane_index
            table_row = self._frames_to_ifd_table[frame_index]

            file_index = int(table_row["file_index"])
            ifd_index = int(table_row["IFD_index"])

            # The ifds are local within a file, so we need to add and offset
            # equal to the number of IFDs in the previous files
            file_offset = sum(self._ifds_per_file[:file_index]) if file_index > 0 else 0

            frame_indices[sample_index] = ifd_index + file_offset

        return frame_indices

    def __del__(self):
        """Close file handles when the extractor is garbage collected."""
        if hasattr(self, "_tiff_readers"):
            for handle in self._tiff_readers:
                try:
                    handle.close()
                except Exception as e:
                    warnings.warn(f"Error closing TIFF file handle {handle} with error: {e}", UserWarning)
                    pass


class ScanImageTiffMultiPlaneMultiFileImagingExtractor(MultiImagingExtractor):
    """Specialized extractor for reading multi-file (buffered) TIFF files produced via ScanImage."""

    extractor_name = "ScanImageTiffMultiPlaneMultiFileImaging"
    mode = "folder"

    def __init__(
        self, folder_path: PathType, file_pattern: str, channel_name: str, extract_all_metadata: bool = True
    ) -> None:
        warnings.warn(
            f"{self.__class__.__name__} is deprecated and will be removed on or after October 2025.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        self.is_volumetric = True

    def get_volume_shape(self) -> Tuple[int, int, int]:
        """Get the shape of the volumetric video (num_rows, num_columns, num_planes).

        Returns
        -------
        video_shape: tuple
            Shape of the volumetric video (num_rows, num_columns, num_planes).
        """
        image_shape = self.get_image_shape()
        return (image_shape[0], image_shape[1], self.get_num_planes())

    def get_num_planes(self) -> int:
        """Get the number of depth planes.

        Returns
        -------
        _num_planes: int
            The number of depth planes.
        """
        return self._num_planes

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # This is a legacy deprecated extractor - delegate to the first imaging extractor
        return self._imaging_extractors[0].get_native_timestamps(start_sample, end_sample)


class ScanImageTiffSinglePlaneMultiFileImagingExtractor(MultiImagingExtractor):
    """Specialized extractor for reading multi-file (buffered) TIFF files produced via ScanImage."""

    extractor_name = "ScanImageTiffSinglePlaneMultiFileImaging"
    mode = "folder"

    def __init__(
        self,
        folder_path: PathType,
        file_pattern: str,
        channel_name: str,
        plane_name: str,
        extract_all_metadata: bool = True,
    ) -> None:
        warnings.warn(
            f"{self.__class__.__name__} is deprecated and will be removed on or after October 2025.",
            DeprecationWarning,
            stacklevel=2,
        )
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

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # This is a legacy deprecated extractor - delegate to the first imaging extractor
        return self._imaging_extractors[0].get_native_timestamps(start_sample, end_sample)


class ScanImageTiffMultiPlaneImagingExtractor(VolumetricImagingExtractor):
    """Specialized extractor for reading multi-plane (volumetric) TIFF files produced via ScanImage."""

    extractor_name = "ScanImageTiffMultiPlaneImaging"
    mode = "file"

    def __init__(
        self,
        file_path: PathType,
        channel_name: Optional[str] = None,
        metadata: Optional[dict] = None,
        parsed_metadata: Optional[dict] = None,
    ) -> None:
        warnings.warn(
            f"{self.__class__.__name__} is deprecated and will be removed on or after October 2025.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        self.is_volumetric = True

    def get_volume_shape(self) -> Tuple[int, int, int]:
        """Get the shape of the volumetric video (num_rows, num_columns, num_planes).

        Returns
        -------
        video_shape: tuple
            Shape of the volumetric video (num_rows, num_columns, num_planes).
        """
        image_shape = self.get_image_shape()
        return (image_shape[0], image_shape[1], self.get_num_planes())

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # This is a legacy deprecated extractor - delegate to the first imaging extractor
        return self._imaging_extractors[0].get_native_timestamps(start_sample, end_sample)


class ScanImageTiffSinglePlaneImagingExtractor(ImagingExtractor):
    """Specialized extractor for reading TIFF files produced via ScanImage."""

    extractor_name = "ScanImageTiffImaging"
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
        warnings.warn(
            f"{self.__class__.__name__} is deprecated and will be removed on or after October 2025.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        self.is_volumetric = False

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

    def get_series(self, start_sample=None, end_sample=None) -> np.ndarray:
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self._num_samples
        end_sample_inclusive = end_sample - 1
        self.check_frame_inputs(end_sample_inclusive)
        self.check_frame_inputs(start_sample)
        raw_start = self.frame_to_raw_index(start_sample)
        raw_end_inclusive = self.frame_to_raw_index(end_sample_inclusive)  # frame_to_raw_index requires inclusive frame
        raw_end = raw_end_inclusive + 1

        ScanImageTiffReader = _get_scanimage_reader()
        with ScanImageTiffReader(filename=str(self.file_path)) as io:
            raw_video = io.data(beg=raw_start, end=raw_end)

        start_cycle = np.ceil(start_sample / self._frames_per_slice).astype("int")
        end_cycle = end_sample // self._frames_per_slice
        num_cycles = end_cycle - start_cycle
        start_frame_in_cycle = start_sample % self._frames_per_slice
        end_frame_in_cycle = end_sample % self._frames_per_slice
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
        series = raw_video[index]
        return series

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

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # Single plane ScanImage files do not have native timestamps
        return None


class ScanImageLegacyImagingExtractor(ImagingExtractor):
    """Specialized extractor for reading TIFF files produced via ScanImage.

    This implementation is for legacy purposes and is not recommended for use.
    Please use ScanImageTiffSinglePlaneImagingExtractor or ScanImageTiffMultiPlaneImagingExtractor instead.
    """

    extractor_name = "ScanImageLegacyImagingExtractor"

    def __init__(
        self,
        file_path: PathType,
        sampling_frequency: FloatType,
    ):
        """Create a ScanImageLegacyImagingExtractor instance from a TIFF file produced by ScanImage.

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
        ScanImageTiffReader = _get_scanimage_reader()

        super().__init__()
        self.file_path = Path(file_path)
        self._sampling_frequency = sampling_frequency
        valid_suffixes = [".tiff", ".tif", ".TIFF", ".TIF"]
        if self.file_path.suffix not in valid_suffixes:
            suffix_string = ", ".join(valid_suffixes[:-1]) + f", or {valid_suffixes[-1]}"
            warning_message = (
                f"Suffix ({self.file_path.suffix}) is not of type {suffix_string}! "
                f"The {self.extractor_name} may not be appropriate for the file."
            )
            warn(warning_message, UserWarning, stacklevel=2)

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

    def get_series(self, start_sample=None, end_sample=None) -> np.ndarray:
        ScanImageTiffReader = _get_scanimage_reader()
        with ScanImageTiffReader(filename=str(self.file_path)) as io:
            return io.data(beg=start_sample, end=end_sample)

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

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # Legacy ScanImage files do not have native timestamps
        return None
