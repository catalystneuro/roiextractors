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


def parse_metadata(metadata):
    """Parse metadata dictionary to extract relevant information and store it standard keys for ImagingExtractors.

    Currently supports
    - sampling_frequency
    - num_channels
    - num_planes
    - frames_per_slice
    - channel_names

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
    SI.hChannels.channelsActive = '[1;2;...;N]' where N is the number of active channels.
    SI.hChannels.channelName = "{'channel_name_1' 'channel_name_2' ... 'channel_name_M'}"
        where M is the number of channels (active or not).
    """
    sampling_frequency = float(metadata["SI.hRoiManager.scanVolumeRate"])
    num_channels = len(metadata["SI.hChannels.channelsActive"].split(";"))
    num_planes = int(metadata["SI.hStackManager.numSlices"])
    frames_per_slice = int(metadata["SI.hStackManager.framesPerSlice"])
    channel_names = metadata["SI.hChannels.channelName"].split("'")[1::2][:num_channels]
    metadata_parsed = dict(
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        num_planes=num_planes,
        frames_per_slice=frames_per_slice,
        channel_names=channel_names,
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
        """
        properties_to_check = dict(
            get_sampling_frequency="The sampling frequency",
            get_image_size="The size of a frame",
            get_num_channels="The number of channels",
            get_channel_names="The name of the channels",
            get_dtype="The data type.",
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

        video = np.zeros((end_frame - start_frame, *self.get_image_size(), self.get_num_planes()), self.get_dtype())
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
            frames = np.zeros((len(frame_idxs), *self.get_image_size(), self.get_num_planes()), self.get_dtype())
            for i, imaging_extractor in enumerate(self._imaging_extractors):
                frames[..., i] = imaging_extractor.get_frames(frame_idxs)
        else:
            return self.get_video(start_frame=frame_idxs[0], end_frame=frame_idxs[-1] + 1)

    def get_image_size(self) -> Tuple:
        return self._imaging_extractors[0].get_image_size()

    def get_num_planes(self) -> int:
        """Get the number of depth planes.

        Returns
        -------
        _num_planes: int
            The number of depth planes.
        """
        return self._num_planes

    def get_num_frames(self) -> int:
        return self._num_frames

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
        sampling_frequency: float,
        channel: Optional[int] = 0,
        num_channels: Optional[int] = 1,
        num_planes: Optional[int] = 1,
        frames_per_slice: Optional[int] = 1,
        channel_names: Optional[list] = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.metadata = extract_extra_metadata(file_path)
        self.channel = channel
        if channel >= num_channels:
            raise ValueError(f"Channel index ({channel}) exceeds number of channels ({num_channels}).")
        if frames_per_slice != 1:
            raise NotImplementedError(
                "Extractor cannot handle multiple frames per slice. Please raise an issue to request this feature: "
                "https://github.com/catalystneuro/roiextractors/issues "
            )
        imaging_extractors = []
        for plane in range(num_planes):
            imaging_extractor = ScanImageTiffImagingExtractor(
                file_path=file_path,
                sampling_frequency=sampling_frequency,
                channel=channel,
                num_channels=num_channels,
                plane=plane,
                num_planes=num_planes,
                channel_names=channel_names,
            )
            imaging_extractors.append(imaging_extractor)
        super().__init__(imaging_extractors=imaging_extractors)
        assert all(
            imaging_extractor.get_num_planes() == self._num_planes for imaging_extractor in imaging_extractors
        ), "All imaging extractors must have the same number of planes."


class ScanImageTiffImagingExtractor(ImagingExtractor):
    """Specialized extractor for reading TIFF files produced via ScanImage."""

    extractor_name = "ScanImageTiffImaging"
    is_writable = True
    mode = "file"

    def __init__(
        self,
        file_path: PathType,
        sampling_frequency: float,
        channel: Optional[int] = 0,
        num_channels: Optional[int] = 1,
        plane: Optional[int] = 0,
        num_planes: Optional[int] = 1,
        frames_per_slice: Optional[int] = 1,
        channel_names: Optional[list] = None,
    ) -> None:
        """Create a ScanImageTiffImagingExtractor instance from a TIFF file produced by ScanImage.

        The underlying data is stored in a round-robin format collapsed into 3 dimensions (frames, rows, columns).
        I.e. the first frame of each channel and each plane is stored, and then the second frame of each channel and
        each plane, etc.
        Ex. for 2 channels and 2 planes:
        [channel_1_plane_1_frame_1, channel_2_plane_1_frame_1, channel_1_plane_2_frame_1, channel_2_plane_2_frame_1,
        channel_1_plane_1_frame_2, channel_2_plane_1_frame_2, channel_1_plane_2_frame_2, channel_2_plane_2_frame_2, ...
        channel_1_plane_1_frame_N, channel_2_plane_1_frame_N, channel_1_plane_2_frame_N, channel_2_plane_2_frame_N]
        This file structured is accessed by ScanImageTiffImagingExtractor for a single channel and plane.

        Parameters
        ----------
        file_path : PathType
            Path to the TIFF file.
        sampling_frequency : float
            Sampling frequency of each plane (scanVolumeRate) in Hz.
        channel : int, optional
            Index of the optical channel for this extractor (default=0).
        num_channels : int, optional
            Number of active channels that were acquired (default=1).
        plane : int, optional
            Index of the depth plane for this extractor (default=0).
        num_planes : int, optional
            Number of depth planes that were acquired (default=1).
        frames_per_slice : int, optional
            Number of frames per depth plane that were acquired (default=1).
        channel_names : list, optional
            Names of the channels (default=None).
        """
        super().__init__()
        self.file_path = Path(file_path)
        self._sampling_frequency = sampling_frequency
        self.metadata = extract_extra_metadata(file_path)
        self.channel = channel
        self._num_channels = num_channels
        self.plane = plane
        self._num_planes = num_planes
        self._channel_names = channel_names
        if channel >= num_channels:
            raise ValueError(f"Channel index ({channel}) exceeds number of channels ({num_channels}).")
        if plane >= num_planes:
            raise ValueError(f"Plane index ({plane}) exceeds number of planes ({num_planes}).")
        if frames_per_slice != 1:
            raise NotImplementedError(
                "Extractor cannot handle multiple frames per slice. Please raise an issue to request this feature: "
                "https://github.com/catalystneuro/roiextractors/issues "
            )

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
            self._num_frames = self._total_num_frames // (self._num_planes * self._num_channels)
        else:
            raise NotImplementedError(
                "Extractor cannot handle 4D TIFF data. Please raise an issue to request this feature: "
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
        self.check_frame_inputs(end_frame - 1)
        ScanImageTiffReader = _get_scanimage_reader()
        raw_start = self.frame_to_raw_index(start_frame)
        raw_end = self.frame_to_raw_index(end_frame)
        raw_end = np.min([raw_end, self._total_num_frames])
        with ScanImageTiffReader(filename=str(self.file_path)) as io:
            raw_video = io.data(beg=raw_start, end=raw_end)
        video = raw_video[self.channel :: self._num_channels]
        video = video[self.plane :: self._num_planes]
        return video

    def get_image_size(self) -> Tuple[int, int]:
        return (self._num_rows, self._num_columns)

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_num_channels(self) -> int:
        return self._num_channels

    def get_channel_names(self) -> list:
        return self._channel_names

    def get_num_planes(self) -> int:
        return self._num_planes

    def get_dtype(self) -> DtypeType:
        return self.get_frames(0).dtype

    def check_frame_inputs(self, frame) -> None:
        if frame >= self._num_frames:
            raise ValueError(f"Frame index ({frame}) exceeds number of frames ({self._num_frames}).")

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
        Ex. for 2 channels and 2 planes:
        [channel_1_plane_1_frame_1, channel_2_plane_1_frame_1, channel_1_plane_2_frame_1, channel_2_plane_2_frame_1,
        channel_1_plane_1_frame_2, channel_2_plane_1_frame_2, channel_1_plane_2_frame_2, channel_2_plane_2_frame_2, ...
        channel_1_plane_1_frame_N, channel_2_plane_1_frame_N, channel_1_plane_2_frame_N, channel_2_plane_2_frame_N]
        """
        raw_index = (frame * self._num_planes * self._num_channels) + (self.plane * self._num_channels) + self.channel
        return raw_index
