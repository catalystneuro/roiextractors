"""Specialized extractor for reading TIFF files produced via ScanImage.

Classes
-------
ScanImageTiffImagingExtractor
    Specialized extractor for reading TIFF files produced via ScanImage.
"""
from pathlib import Path
from typing import Optional, Tuple
from warnings import warn
import numpy as np
from pprint import pprint

from ...extraction_tools import PathType, FloatType, ArrayType, get_package
from ...imagingextractor import ImagingExtractor


def _get_scanimage_reader() -> type:
    """Import the scanimage-tiff-reader package and return the ScanImageTiffReader class."""
    return get_package(
        package_name="ScanImageTiffReader", installation_instructions="pip install scanimage-tiff-reader"
    ).ScanImageTiffReader


def extract_extra_metadata(
    file_path,
) -> dict:  # TODO: Refactor neuroconv to reference this implementation to avoid duplication
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


class ScanImageTiffImagingExtractor(ImagingExtractor):
    """Specialized extractor for reading TIFF files produced via ScanImage."""

    extractor_name = "ScanImageTiffImaging"
    is_writable = True
    mode = "file"

    def __init__(self, file_path: PathType) -> None:
        """Create a ScanImageTiffImagingExtractor instance from a TIFF file produced by ScanImage.

        The underlying data is stored in a round-robin format collapsed into 3 dimensions (frames, rows, columns).
        I.e. the first frame of each channel and each plane is stored, and then the second frame of each channel and
        each plane, etc.
        Ex. for 2 channels and 2 planes:
        [channel_1_plane_1_frame_1, channel_2_plane_1_frame_1, channel_1_plane_2_frame_1, channel_2_plane_2_frame_1,
        channel_1_plane_1_frame_2, channel_2_plane_1_frame_2, channel_1_plane_2_frame_2, channel_2_plane_2_frame_2, ...
        channel_1_plane_1_frame_N, channel_2_plane_1_frame_N, channel_1_plane_2_frame_N, channel_2_plane_2_frame_N]
        This file structure is sliced lazily using ScanImageTiffReader with the appropriate logic for specified
        channels/frames.

        Parameters
        ----------
        file_path : PathType
            Path to the TIFF file.
        """
        super().__init__()
        self.file_path = Path(file_path)
        ScanImageTiffReader = _get_scanimage_reader()
        extra_metadata = extract_extra_metadata(file_path)
        # SI.hChannels.channelsActive = '[1;2;...;N]' where N is the number of active channels
        self._num_channels = len(extra_metadata["SI.hChannels.channelsActive"].split(";"))
        self._num_planes = int(extra_metadata["SI.hStackManager.numSlices"])
        frames_per_slice = int(extra_metadata["SI.hStackManager.framesPerSlice"])
        if frames_per_slice != 1:
            raise NotImplementedError(
                "Extractor cannot handle multiple frames per slice. Please raise an issue to request this feature: "
                "https://github.com/catalystneuro/roiextractors/issues "
            )
        self._sampling_frequency = float(extra_metadata["SI.hRoiManager.scanVolumeRate"])
        # SI.hChannels.channelName = "{'channel_name_1' 'channel_name_2' ... 'channel_name_N'}"
        #   where N is the number of channels (active or not)
        self._channel_names = extra_metadata["SI.hChannels.channelName"].split("'")[1::2][: self._num_channels]

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
            self._total_num_frames, self._num_rows, self._num_columns = shape
            self._num_frames = self._total_num_frames // (self._num_planes * self._num_channels)
        else:
            raise NotImplementedError(
                "Extractor cannot handle 4D TIFF data. Please raise an issue to request this feature: "
                "https://github.com/catalystneuro/roiextractors/issues "
            )

    def get_frames(self, frame_idxs: ArrayType, channel: int = 0, plane: int = 0) -> np.ndarray:
        """Get specific video frames from indices (not necessarily continuous).

        Parameters
        ----------
        frame_idxs: array-like
            Indices of frames to return.
        channel: int, optional
            Channel index.
        plane: int, optional
            Plane index.

        Returns
        -------
        frames: numpy.ndarray
            The video frames.
        """
        self.check_frame_inputs(frame_idxs[-1], channel, plane)
        if isinstance(frame_idxs, int):
            frame_idxs = [frame_idxs]

        if not all(np.diff(frame_idxs) == 1):
            return np.concatenate([self._get_single_frame(frame=idx) for idx in frame_idxs])
        else:
            return self.get_video(start_frame=frame_idxs[0], end_frame=frame_idxs[-1] + 1, channel=channel)

    # Data accessed through an open ScanImageTiffReader io gets scrambled if there are multiple calls.
    # Thus, open fresh io in context each time something is needed.
    def _get_single_frame(self, frame: int, channel: Optional[int] = 0, plane: Optional[int] = 0) -> np.ndarray:
        """Get a single frame of data from the TIFF file.

        Parameters
        ----------
        frame : int
            The index of the frame to retrieve.
        channel : int, optional
            The index of the channel to retrieve.
        plane : int, optional
            The index of the plane to retrieve.

        Returns
        -------
        frame: numpy.ndarray
            The frame of data.
        """
        self.check_frame_inputs(frame, channel, plane)
        ScanImageTiffReader = _get_scanimage_reader()
        raw_index = (frame * self._num_planes * self._num_channels) + (plane * self._num_channels) + channel
        with ScanImageTiffReader(str(self.file_path)) as io:
            return io.data(beg=raw_index, end=raw_index + 1)

    def get_video(
        self, start_frame=None, end_frame=None, channel: Optional[int] = 0, plane: Optional[int] = 0
    ) -> np.ndarray:
        """Get the video frames.

        Parameters
        ----------
        start_frame: int, optional
            Start frame index (inclusive).
        end_frame: int, optional
            End frame index (exclusive).
        channel: int, optional
            Channel index.
        plane: int, optional
            Plane index.

        Returns
        -------
        video: numpy.ndarray
            The video frames.
        """
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self._num_frames
        self.check_frame_inputs(end_frame - 1, channel, plane)
        ScanImageTiffReader = _get_scanimage_reader()
        raw_start = (start_frame * self._num_planes * self._num_channels) + (plane * self._num_channels) + channel
        raw_end = (end_frame * self._num_planes * self._num_channels) + (plane * self._num_channels) + channel
        raw_end = np.min([raw_end, self._total_num_frames])
        with ScanImageTiffReader(filename=str(self.file_path)) as io:
            raw_video = io.data(beg=raw_start, end=raw_end)
        video = raw_video[channel :: self._num_channels]
        video = video[plane :: self._num_planes]
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

    def check_frame_inputs(self, frame, channel, plane) -> None:
        if frame >= self._num_frames:
            raise ValueError(f"Frame index ({frame}) exceeds number of frames ({self._num_frames}).")
        if channel >= self._num_channels:
            raise ValueError(f"Channel index ({channel}) exceeds number of channels ({self._num_channels}).")
        if plane >= self._num_planes:
            raise ValueError(f"Plane index ({plane}) exceeds number of planes ({self._num_planes}).")
