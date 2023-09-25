"""Specialized extractor for reading TIFF files produced via ScanImage.

Classes
-------
ScanImageTiffImagingExtractor
    Specialized extractor for reading TIFF files produced via ScanImage.
"""
from pathlib import Path
from typing import Optional, Tuple, List, Iterable
from warnings import warn
import numpy as np
from pprint import pprint

from roiextractors.extraction_tools import DtypeType

from ...extraction_tools import PathType, FloatType, ArrayType, DtypeType, get_package
from ...imagingextractor import ImagingExtractor


def _get_scanimage_reader() -> type:
    """Import the scanimage-tiff-reader package and return the ScanImageTiffReader class."""
    return get_package(
        package_name="ScanImageTiffReader", installation_instructions="pip install scanimage-tiff-reader"
    ).ScanImageTiffReader


def extract_extra_metadata(
    file_path,
) -> dict:  # TODO: Refactor neuroconv to reference this implementation to avoid duplication
    """Extract metadata from a ScanImage TIFF file.

    Parameters
    ----------
    file_path : PathType
        Path to the TIFF file.

    Returns
    -------
    extra_metadata: dict
        Dictionary of metadata extracted from the TIFF file.

    Notes
    -----
    Known to work on SI versions v3.8.0, v2019bR0, and v2022.0.0.
    """
    ScanImageTiffReader = _get_scanimage_reader()
    io = ScanImageTiffReader(str(file_path))
    extra_metadata = {}
    for metadata_string in (io.description(iframe=0), io.metadata()):
        metadata_dict = {
            x.split("=")[0].strip(): x.split("=")[1].strip()
            for x in metadata_string.replace("\n", "\r").split("\r")
            if "=" in x
        }
        extra_metadata = dict(**extra_metadata, **metadata_dict)
    return extra_metadata


def parse_matlab_vector(matlab_vector: str) -> list:
    """Parse a MATLAB vector string into a list of integer values.

    Parameters
    ----------
    matlab_vector : str
        MATLAB vector string.

    Returns
    -------
    vector: list of int
        List of integer values.

    Raises
    ------
    ValueError
        If the MATLAB vector string cannot be parsed.

    Notes
    -----
    MATLAB vector string is of the form "[1 2 3 ... N]" or "[1,2,3,...,N]" or "[1;2;3;...;N]".
    There may or may not be whitespace between the values. Ex. "[1, 2, 3]" or "[1,2,3]".
    """
    vector = matlab_vector.strip("[]")
    if ";" in vector:
        vector = vector.split(";")
    elif "," in vector:
        vector = vector.split(",")
    elif " " in vector:
        vector = vector.split(" ")
    elif len(vector) == 1:
        pass
    else:
        raise ValueError(f"Could not parse vector from {matlab_vector}.")
    vector = [int(x.strip()) for x in vector if x != ""]
    return vector


def parse_metadata(metadata):
    """Parse metadata dictionary to extract relevant information and store it standard keys for ImagingExtractors.

    Currently supports
    - sampling_frequency
    - num_planes
    - frames_per_slice
    - channel_names
    - num_channels

    Parameters
    ----------
    metadata : dict
        Dictionary of metadata extracted from the TIFF file.

    Returns
    -------
    metadata_parsed: dict
        Dictionary of parsed metadata.

    Notes
    -----
    Known to work on SI versions v2019bR0 and v2022.0.0.
    SI.hChannels.channelsActive = string of MATLAB-style vector with channel integers (see parse_matlab_vector).
    SI.hChannels.channelName = "{'channel_name_1' 'channel_name_2' ... 'channel_name_M'}"
        where M is the number of channels (active or not).
    """
    sampling_frequency = float(metadata["SI.hRoiManager.scanVolumeRate"])
    num_planes = int(metadata["SI.hStackManager.numSlices"])
    frames_per_slice = int(metadata["SI.hStackManager.framesPerSlice"])
    active_channels = parse_matlab_vector(metadata["SI.hChannels.channelsActive"])
    channel_indices = np.array(active_channels) - 1  # Account for MATLAB indexing
    channel_names = np.array(metadata["SI.hChannels.channelName"].split("'")[1::2])
    channel_names = channel_names[channel_indices].tolist()
    num_channels = len(channel_names)
    metadata_parsed = dict(
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        num_planes=num_planes,
        frames_per_slice=frames_per_slice,
        channel_names=channel_names,
    )
    return metadata_parsed


def parse_metadata_v3_8(metadata):
    """Parse metadata dictionary to extract relevant information and store it standard keys for ImagingExtractors.

    Requires old version of metadata (v3.8).
    Currently supports
    - sampling frequency
    - num_channels
    - num_planes

    Parameters
    ----------
    metadata : dict
        Dictionary of metadata extracted from the TIFF file.

    Returns
    -------
    metadata_parsed: dict
        Dictionary of parsed metadata.
    """
    sampling_frequency = float(metadata["state.acq.frameRate"])
    num_channels = int(metadata["state.acq.numberOfChannelsSave"])
    num_planes = int(metadata["state.acq.numberOfZSlices"])
    metadata_parsed = dict(
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        num_planes=num_planes,
    )
    return metadata_parsed


class MultiPlaneImagingExtractor(ImagingExtractor):
    """Class to combine multiple ImagingExtractor objects by depth plane."""

    extractor_name = "MultiPlaneImaging"
    installed = True
    installatiuon_mesage = ""

    def __init__(self, imaging_extractors: List[ImagingExtractor]):
        """Initialize a MultiPlaneImagingExtractor object from a list of ImagingExtractors.

        Parameters
        ----------
        imaging_extractors: list of ImagingExtractor
            list of imaging extractor objects
        """
        super().__init__()
        assert isinstance(imaging_extractors, list), "Enter a list of ImagingExtractor objects as argument"
        assert all(isinstance(imaging_extractor, ImagingExtractor) for imaging_extractor in imaging_extractors)
        self._check_consistency_between_imaging_extractors(imaging_extractors)
        self._imaging_extractors = imaging_extractors
        self._num_planes = len(imaging_extractors)

    # TODO: Add consistency check for channel_names when API is standardized
    def _check_consistency_between_imaging_extractors(self, imaging_extractors: List[ImagingExtractor]):
        """Check that essential properties are consistent between extractors so that they can be combined appropriately.

        Parameters
        ----------
        imaging_extractors: list of ImagingExtractor
            list of imaging extractor objects

        Raises
        ------
        AssertionError
            If any of the properties are not consistent between extractors.

        Notes
        -----
        This method checks the following properties:
            - sampling frequency
            - image size
            - number of channels
            - channel names
            - data type
            - num_frames
        """
        properties_to_check = dict(
            get_sampling_frequency="The sampling frequency",
            get_image_size="The size of a frame",
            get_num_channels="The number of channels",
            get_dtype="The data type",
            get_num_frames="The number of frames",
        )
        for method, property_message in properties_to_check.items():
            values = [getattr(extractor, method)() for extractor in imaging_extractors]
            unique_values = set(tuple(v) if isinstance(v, Iterable) else v for v in values)
            assert (
                len(unique_values) == 1
            ), f"{property_message} is not consistent over the files (found {unique_values})."

    def get_video(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> np.ndarray:
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
            The 3D video frames (num_rows, num_columns, num_planes).
        """
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else self.get_num_frames()

        video = np.zeros((end_frame - start_frame, *self.get_image_size()), self.get_dtype())
        for i, imaging_extractor in enumerate(self._imaging_extractors):
            video[..., i] = imaging_extractor.get_video(start_frame, end_frame)
        return video

    def get_frames(self, frame_idxs: ArrayType) -> np.ndarray:
        """Get specific video frames from indices (not necessarily continuous).

        Parameters
        ----------
        frame_idxs: array-like
            Indices of frames to return.

        Returns
        -------
        frames: numpy.ndarray
            The 3D video frames (num_rows, num_columns, num_planes).
        """
        if isinstance(frame_idxs, int):
            frame_idxs = [frame_idxs]

        if not all(np.diff(frame_idxs) == 1):
            frames = np.zeros((len(frame_idxs), *self.get_image_size()), self.get_dtype())
            for i, imaging_extractor in enumerate(self._imaging_extractors):
                frames[..., i] = imaging_extractor.get_frames(frame_idxs)
        else:
            return self.get_video(start_frame=frame_idxs[0], end_frame=frame_idxs[-1] + 1)

    def get_image_size(self) -> Tuple:
        """Get the size of a single frame.

        Returns
        -------
        image_size: tuple
            The size of a single frame (num_rows, num_columns, num_planes).
        """
        image_size = (*self._imaging_extractors[0].get_image_size(), self.get_num_planes())
        return image_size

    def get_num_planes(self) -> int:
        """Get the number of depth planes.

        Returns
        -------
        _num_planes: int
            The number of depth planes.
        """
        return self._num_planes

    def get_num_frames(self) -> int:
        return self._imaging_extractors[0].get_num_frames()

    def get_sampling_frequency(self) -> float:
        return self._imaging_extractors[0].get_sampling_frequency()

    def get_channel_names(self) -> list:
        return self._imaging_extractors[0].get_channel_names()

    def get_num_channels(self) -> int:
        return self._imaging_extractors[0].get_num_channels()

    def get_dtype(self) -> DtypeType:
        return self._imaging_extractors[0].get_dtype()


class ScanImageTiffMultiPlaneImagingExtractor(MultiPlaneImagingExtractor):
    """Specialized extractor for reading multi-plane (volumetric) TIFF files produced via ScanImage."""

    extractor_name = "ScanImageTiffMultiPlaneImaging"
    is_writable = True
    mode = "file"

    def __init__(
        self,
        file_path: PathType,
        channel_name: Optional[str] = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.metadata = extract_extra_metadata(file_path)
        parsed_metadata = parse_metadata(self.metadata)
        num_planes = parsed_metadata["num_planes"]
        channel_names = parsed_metadata["channel_names"]
        if channel_name is None:
            channel_name = channel_names[0]
        imaging_extractors = []
        for plane in range(num_planes):
            imaging_extractor = ScanImageTiffSinglePlaneImagingExtractor(
                file_path=file_path, channel_name=channel_name, plane_name=str(plane)
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
    def get_channel_names(cls, file_path):
        """Get the channel names from a TIFF file produced by ScanImage.

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
    def get_plane_names(cls, file_path):
        """Get the plane names from a TIFF file produced by ScanImage.

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
        """
        super().__init__()
        self.file_path = Path(file_path)
        self.metadata = extract_extra_metadata(file_path)
        parsed_metadata = parse_metadata(self.metadata)
        self._sampling_frequency = parsed_metadata["sampling_frequency"]
        self._num_channels = parsed_metadata["num_channels"]
        self._num_planes = parsed_metadata["num_planes"]
        self._frames_per_slice = parsed_metadata["frames_per_slice"]
        self._channel_names = parsed_metadata["channel_names"]
        self._plane_names = [f"{i}" for i in range(self._num_planes)]
        self.channel_name = channel_name
        self.plane_name = plane_name
        if channel_name not in self._channel_names:
            raise ValueError(f"Channel name ({channel_name}) not found in channel names ({self._channel_names}).")
        self.channel = self._channel_names.index(channel_name)
        if plane_name not in self._plane_names:
            raise ValueError(f"Plane name ({plane_name}) not found in plane names ({self._plane_names}).")
        self.plane = self._plane_names.index(plane_name)

        valid_suffixes = [".tiff", ".tif", ".TIFF", ".TIF"]
        if self.file_path.suffix not in valid_suffixes:
            suffix_string = ", ".join(valid_suffixes[:-1]) + f", or {valid_suffixes[-1]}"
            warn(
                f"Suffix ({self.file_path.suffix}) is not of type {suffix_string}! "
                f"The {self.extractor_name}Extractor may not be appropriate for the file."
            )
        ScanImageTiffReader = _get_scanimage_reader()
        with ScanImageTiffReader(str(self.file_path)) as io:
            shape = io.shape()  # [frames, rows, columns]
        if len(shape) == 3:
            self._total_num_frames, self._num_rows, self._num_columns = shape
            self._num_raw_per_plane = self._frames_per_slice * self._num_channels
            self._num_raw_per_cycle = self._num_raw_per_plane * self._num_planes
            self._num_frames = self._total_num_frames // (self._num_planes * self._num_channels)
            self._num_cycles = self._total_num_frames // self._num_raw_per_cycle
        else:
            raise NotImplementedError(
                "Extractor cannot handle 4D ScanImageTiff data. Please raise an issue to request this feature: "
                "https://github.com/catalystneuro/roiextractors/issues "
            )

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
            end_frame = self._num_frames
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

    def get_image_size(self) -> Tuple[int, int]:
        return (self._num_rows, self._num_columns)

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_num_channels(self) -> int:
        return self._num_channels

    def get_num_planes(self) -> int:
        return self._num_planes

    def get_dtype(self) -> DtypeType:
        return self.get_frames(0).dtype

    def check_frame_inputs(self, frame) -> None:
        if frame >= self._num_frames:
            raise ValueError(f"Frame index ({frame}) exceeds number of frames ({self._num_frames}).")
        if frame < 0:
            raise ValueError(f"Frame index ({frame}) must be greater than or equal to 0.")

    def frame_to_raw_index(self, frame):
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


class ScanImageTiffImagingExtractor(ImagingExtractor):  # TODO: Remove this extractor on/after December 2023
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
            self._num_frames, self._num_rows, self._num_columns = shape
            self._num_channels = 1
        else:  # no example file for multiple color channels or depths
            raise NotImplementedError(
                "Extractor cannot handle 4D TIFF data. Please raise an issue to request this feature: "
                "https://github.com/catalystneuro/roiextractors/issues "
            )

    def get_frames(self, frame_idxs: ArrayType, channel: int = 0) -> np.ndarray:
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
        ScanImageTiffReader = _get_scanimage_reader()

        with ScanImageTiffReader(filename=str(self.file_path)) as io:
            return io.data(beg=start_frame, end=end_frame)

    def get_image_size(self) -> Tuple[int, int]:
        return (self._num_rows, self._num_columns)

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_num_channels(self) -> int:
        return self._num_channels

    def get_channel_names(self) -> list:
        pass
