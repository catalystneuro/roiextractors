"""Extractor for multiple TIFF files, each with multiple pages.

Classes
-------
MultiTIFFMultiPageExtractor
    An extractor for handling multiple TIFF files, each with multiple pages, organized according to a specified dimension order.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, NamedTuple
from collections import defaultdict
import warnings
import numpy as np
import glob

from ...extraction_tools import PathType, FloatType, ArrayType, DtypeType, get_package
from ...imagingextractor import ImagingExtractor


class MultiTIFFMultiPageExtractor(ImagingExtractor):
    """
    An extractor for handling multiple TIFF files, each with multiple pages, organized according to a specified dimension order.

    This extractor allows for lazy loading of frames from multiple TIFF files, where each file may contain multiple pages.
    The frames are organized according to a specified dimension order (e.g., XYZCT) and the size of each dimension.

    The extractor creates a mapping between each logical frame index and its corresponding file and page location.
    This mapping is used to efficiently retrieve frames when requested.
    """

    extractor_name = "MultiTIFFMultiPageExtractor"
    is_writable = False

    # Named tuple for mapping frames to file/page locations
    class FrameMapping(NamedTuple):
        """A named tuple for mapping frames to file and page locations."""

        file_index: int
        page_index: int

    def __init__(
        self,
        file_paths: List[PathType],
        dimension_order: str = "XYZCT",
        channel_size: int = 1,
        time_size: Optional[int] = None,
        depth_size: int = 1,
        sampling_frequency: FloatType = 30.0,
    ):
        """Initialize the extractor with file paths and dimension information.

        Parameters
        ----------
        file_paths : List[PathType]
            List of paths to TIFF files.
        dimension_order : str, optional
            The order of dimensions in the data. Must be one of: XYZCT, XYZTC, XYCTZ, XYTCZ, XYCZT, XYTCZ, XYTZC.
            Default is "XYZCT".
        channel_size : int, optional
            Number of channels. Default is 1.
        time_size : int, optional
            Number of time points. If None, it will be calculated based on the total number of frames and other dimensions.
            Default is None.
        depth_size : int, optional
            Number of depth planes (Z). Default is 1.
        sampling_frequency : float, optional
            The sampling frequency in Hz. Default is 30.0.
        """
        super().__init__()

        # Validate dimension order
        valid_dimension_orders = ["XYZCT", "XYZTC", "XYCTZ", "XYTCZ", "XYCZT", "XYTCZ", "XYTZC"]
        if dimension_order not in valid_dimension_orders:
            raise ValueError(f"Invalid dimension order: {dimension_order}. Must be one of: {valid_dimension_orders}")

        self.file_paths = [Path(file_path) for file_path in file_paths]
        self.dimension_order = dimension_order
        self.channel_size = channel_size
        self.depth_size = depth_size
        self._sampling_frequency = sampling_frequency

        # Get tifffile package
        tifffile = get_package(package_name="tifffile")

        # Open all TIFF files and store file handles for lazy loading
        self._file_handles = []
        total_pages = 0

        for file_path in self.file_paths:
            try:
                tiff_handle = tifffile.TiffFile(file_path)
                self._file_handles.append(tiff_handle)
                total_pages += len(tiff_handle.pages)
            except Exception as e:
                # Close any opened file handles before raising the exception
                for handle in self._file_handles:
                    handle.close()
                raise RuntimeError(f"Error opening TIFF file {file_path}: {e}")

        # Get image dimensions from the first page of the first file
        if not self._file_handles or not self._file_handles[0].pages:
            raise ValueError("No valid TIFF files or pages found")

        first_page = self._file_handles[0].pages[0]
        self._num_rows, self._num_columns = first_page.shape
        self._dtype = first_page.dtype

        # Calculate time_size if not provided
        frames_per_file = [len(handle.pages) for handle in self._file_handles]
        total_frames = sum(frames_per_file)

        if time_size is None:
            # Calculate time_size based on total frames and other dimensions
            frames_per_volume = channel_size * depth_size
            if frames_per_volume == 0:
                raise ValueError("Invalid dimension sizes: channel_size and depth_size cannot both be zero")

            if total_frames % frames_per_volume != 0:
                warnings.warn(
                    f"Total frames ({total_frames}) is not divisible by frames per volume ({frames_per_volume}). "
                    f"Some frames may not be accessible."
                )

            time_size = total_frames // frames_per_volume

        self.time_size = time_size
        self._num_frames = time_size

        # Create mapping from frame index to file and page
        self._frame_mapping = self._create_frame_mapping(
            dimension_order=dimension_order,
            channel_size=channel_size,
            time_size=time_size,
            depth_size=depth_size,
            frames_per_file=frames_per_file,
        )

        # Store initialization parameters for potential reconstruction
        self._kwargs = {
            "file_paths": [str(path) for path in self.file_paths],
            "dimension_order": dimension_order,
            "channel_size": channel_size,
            "time_size": time_size,
            "depth_size": depth_size,
            "sampling_frequency": sampling_frequency,
        }

    def _create_frame_mapping(
        self, dimension_order: str, channel_size: int, time_size: int, depth_size: int, frames_per_file: List[int]
    ) -> Dict[int, FrameMapping]:
        """Create a mapping from frame index to file and page.

        Parameters
        ----------
        dimension_order : str
            The order of dimensions in the data.
        channel_size : int
            Number of channels.
        time_size : int
            Number of time points.
        depth_size : int
            Number of depth planes (Z).
        frames_per_file : List[int]
            Number of frames in each file.

        Returns
        -------
        Dict[int, FrameMapping]
            A dictionary mapping frame indices to file and page locations.
        """
        frame_mapping = {}

        # Extract dimension indices from dimension_order
        # X and Y are always together in a single page
        t_idx = dimension_order.index("T")  # TCZ
        c_idx = dimension_order.index("C")
        z_idx = dimension_order.index("Z")

        # Remove X and Y from consideration for indexing
        non_spatial_order = dimension_order.replace("X", "").replace("Y", "")
        non_spatial_sizes = []

        for dim in non_spatial_order:
            if dim == "T":
                non_spatial_sizes.append(time_size)
            elif dim == "C":
                non_spatial_sizes.append(channel_size)
            elif dim == "Z":
                non_spatial_sizes.append(depth_size)

        # Calculate total number of pages needed
        total_pages_needed = time_size * channel_size * depth_size

        # Create mapping for each frame
        frame_idx = 0
        file_idx = 0
        page_offset = 0

        for t in range(time_size):
            for c in range(channel_size):
                for z in range(depth_size):
                    # Calculate the page index based on dimension order
                    coords = [0, 0, 0]  # [T, C, Z]
                    coords[non_spatial_order.index("T")] = t
                    coords[non_spatial_order.index("C")] = c
                    coords[non_spatial_order.index("Z")] = z

                    # Calculate linear index
                    linear_idx = 0
                    stride = 1
                    for i in range(len(non_spatial_order) - 1, -1, -1):
                        linear_idx += coords[i] * stride
                        stride *= non_spatial_sizes[i]

                    # Find which file and page this frame is in
                    global_page_idx = linear_idx

                    # Find the file that contains this page
                    current_file_idx = 0
                    current_page_offset = 0

                    while current_file_idx < len(frames_per_file):
                        if global_page_idx < current_page_offset + frames_per_file[current_file_idx]:
                            # Found the file
                            local_page_idx = global_page_idx - current_page_offset
                            frame_mapping[frame_idx] = self.FrameMapping(
                                file_index=current_file_idx, page_index=local_page_idx
                            )
                            break

                        current_page_offset += frames_per_file[current_file_idx]
                        current_file_idx += 1

                    frame_idx += 1

        return frame_mapping

    def get_frames(self, frame_idxs: ArrayType) -> np.ndarray:
        """Get specific frames by their indices.

        Parameters
        ----------
        frame_idxs : array-like
            Indices of frames to retrieve.

        Returns
        -------
        numpy.ndarray
            Array of frames with shape (n_frames, height, width) if depth_size is 1,
            or (n_frames, height, width, depth_size) if depth_size > 1.
        """
        if isinstance(frame_idxs, (int, np.integer)):
            frame_idxs = [frame_idxs]
            single_frame = True
        else:
            single_frame = False

        frame_idxs = np.array(frame_idxs)
        if np.any(frame_idxs >= self._num_frames) or np.any(frame_idxs < 0):
            raise ValueError(f"Frame indices must be between 0 and {self._num_frames - 1}")

        # Check if we need to return volumes (depth_size > 1)
        is_volumetric = self.depth_size > 1

        if is_volumetric:
            # Preallocate output array for volumetric data
            frames = np.empty((len(frame_idxs), self._num_rows, self._num_columns, self.depth_size), dtype=self._dtype)

            # Load each requested frame with all depth planes
            for i, frame_idx in enumerate(frame_idxs):
                # For each frame, we need to get all depth planes
                for z in range(self.depth_size):
                    # Calculate the frame index for this time point and depth
                    # This depends on the dimension order
                    if "Z" in self.dimension_order and "T" in self.dimension_order:
                        z_idx = self.dimension_order.index("Z")
                        t_idx = self.dimension_order.index("T")

                        # Determine which comes first in the dimension order, Z or T
                        if z_idx < t_idx:
                            # Z varies faster than T (e.g., XYZCT)
                            volume_frame_idx = frame_idx * self.depth_size + z
                        else:
                            # T varies faster than Z (e.g., XYTCZ)
                            volume_frame_idx = z * self._num_frames + frame_idx
                    else:
                        # Default to Z varying faster than T
                        volume_frame_idx = frame_idx * self.depth_size + z

                    if volume_frame_idx not in self._frame_mapping:
                        raise ValueError(f"No mapping found for volume frame {volume_frame_idx}")

                    mapping = self._frame_mapping[volume_frame_idx]
                    file_handle = self._file_handles[mapping.file_index]
                    page = file_handle.pages[mapping.page_index]
                    frames[i, :, :, z] = page.asarray()
        else:
            # Preallocate output array for non-volumetric data
            frames = np.empty((len(frame_idxs), self._num_rows, self._num_columns), dtype=self._dtype)

            # Load each requested frame
            for i, frame_idx in enumerate(frame_idxs):
                if frame_idx not in self._frame_mapping:
                    raise ValueError(f"No mapping found for frame {frame_idx}")

                mapping = self._frame_mapping[frame_idx]
                file_handle = self._file_handles[mapping.file_index]
                page = file_handle.pages[mapping.page_index]
                frames[i] = page.asarray()

        if single_frame and not is_volumetric:
            return frames[0]
        elif single_frame and is_volumetric:
            # For volumetric data, we still return a 3D array (height, width, depth) for a single frame
            return frames[0]
        return frames

    def get_video(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> np.ndarray:
        """Get a range of frames.

        Parameters
        ----------
        start_frame : int, optional
            Start frame index (inclusive). Default is 0.
        end_frame : int, optional
            End frame index (exclusive). Default is the total number of frames.

        Returns
        -------
        numpy.ndarray
            Array of frames with shape (n_frames, height, width) if depth_size is 1,
            or (n_frames, height, width, depth_size) if depth_size > 1.
        """
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self._num_frames

        frame_idxs = np.arange(start_frame, end_frame)
        return self.get_frames(frame_idxs)

    def get_image_size(self) -> Tuple[int, int]:
        """Get the size of the video (num_rows, num_columns).

        Returns
        -------
        tuple
            Size of the video (num_rows, num_columns).
        """
        return (self._num_rows, self._num_columns)

    def get_num_frames(self) -> int:
        """Get the number of frames in the video.

        Returns
        -------
        int
            Number of frames in the video.
        """
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        """Get the sampling frequency in Hz.

        Returns
        -------
        float
            Sampling frequency in Hz.
        """
        return self._sampling_frequency

    def get_channel_names(self) -> list:
        """Get the channel names in the recording.

        Returns
        -------
        list
            List of strings of channel names.
        """
        return [f"Channel{i}" for i in range(self.channel_size)]

    def get_dtype(self) -> DtypeType:
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
        dimension_order: str = "XYZCT",
        channel_size: int = 1,
        time_size: Optional[int] = None,
        depth_size: int = 1,
        sampling_frequency: FloatType = 30.0,
    ) -> "MultiTIFFMultiPageExtractor":
        """Create an extractor from a folder path and file pattern.

        Parameters
        ----------
        folder_path : PathType
            Path to the folder containing TIFF files.
        file_pattern : str
            Glob pattern for identifying TIFF files (e.g., "*.tif").
        dimension_order : str, optional
            The order of dimensions in the data. Default is "XYZCT".
        channel_size : int, optional
            Number of channels. Default is 1.
        time_size : int, optional
            Number of time points. If None, it will be calculated. Default is None.
        depth_size : int, optional
            Number of depth planes (Z). Default is 1.
        sampling_frequency : float, optional
            The sampling frequency in Hz. Default is 30.0.

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
            dimension_order=dimension_order,
            channel_size=channel_size,
            time_size=time_size,
            depth_size=depth_size,
            sampling_frequency=sampling_frequency,
        )
