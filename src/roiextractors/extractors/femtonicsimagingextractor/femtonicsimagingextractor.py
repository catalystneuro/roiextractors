"""A Femtonics imaging extractor with simplified initialization."""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from warnings import warn
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import numpy as np
import h5py
from lazy_ops import DatasetView

from ...imagingextractor import ImagingExtractor
from ...extraction_tools import PathType, FloatType, ArrayType


class FemtonicsImagingExtractor(ImagingExtractor):
    """A Femtonics imaging extractor with simplified initialization."""

    extractor_name = "FemtonicsImaging"
    is_writable = False
    mode = "file"

    def __init__(
        self,
        file_path: PathType,
        munit: int = 0,
        channel_name: Optional[str] = None,
        channel_index: Optional[int] = None,
    ):
        """Create a FemtonicsImagingExtractor from a .mesc file.

        Parameters
        ----------
        file_path : str or Path
            Path to the .mesc file.
        munit : int, optional
            Index of the measurement unit to extract. The default is 0.
        channel_name : str, optional
            Name of the channel to extract (e.g., 'UG', 'UR').
            If provided, takes precedence over channel_index.
        channel_index : int, optional
            Index of the channel to extract (0 or 1). The default is 0.
        """
        super().__init__(file_path=file_path)

        self.file_path = Path(file_path)
        self._munit = munit
        self._channel_name = channel_name
        self._channel_index = channel_index or 0

        if self.file_path.suffix != ".mesc":
            warn("File is not a .mesc file")

        # Open file and setup basic access
        self._file = h5py.File(file_path, "r")
        self._setup_channel_selection()
        self._setup_video_data()

    def _setup_channel_selection(self):
        """Determine which channel to extract using either name or index."""
        available_channels = self.get_available_channels_from_file()

        if self._channel_name is not None:
            if self._channel_name not in available_channels:
                raise ValueError(f"Channel '{self._channel_name}' not found. Available: {available_channels}")
            self._selected_channel_index = available_channels.index(self._channel_name)
            self._selected_channel_name = self._channel_name
        else:
            if self._channel_index >= len(available_channels):
                raise ValueError(f"Channel index {self._channel_index} out of range. Available: {available_channels}")
            self._selected_channel_index = self._channel_index
            self._selected_channel_name = available_channels[self._channel_index]

    def _setup_video_data(self):
        """Setup access to the actual imagon data  for the selected measurement unit and chanel."""
        session_key = f"MSession_{self._munit}"
        munit_key = f"MUnit_{self._munit}"
        channel_key = f"Channel_{self._selected_channel_index}"

        if (
            session_key in self._file
            and munit_key in self._file[session_key]
            and channel_key in self._file[session_key][munit_key]
        ):
            self._video = DatasetView(self._file[session_key][munit_key][channel_key])
        else:
            raise Exception(f"Cannot find data at {session_key}/{munit_key}/{channel_key}")

    def get_available_channels_from_file(self) -> List[str]:
        """Get available channels from the current file."""
        session_key = f"MSession_{self._munit}"
        munit_key = f"MUnit_{self._munit}"

        if session_key not in self._file or munit_key not in self._file[session_key]:
            raise ValueError(f"MUnit {self._munit} not found in file")

        attrs = dict(self._file[session_key][munit_key].attrs)
        num_channels = attrs.get("VecChannelsSize", 2)

        channels = []
        for i in range(num_channels):
            name_arr = attrs.get(f"Channel_{i}_Name", [])
            name = self._decode_string(name_arr) if len(name_arr) > 0 else f"Channel_{i}"
            channels.append(name)
        return channels

    def _decode_string(self, arr) -> str:
        """Convert int16 array to string."""
        if len(arr) == 0:
            return ""
        return "".join(chr(x) for x in arr if x != 0)

    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns)."""
        return self._video.shape[1], self._video.shape[2]  # (height, width)

    def get_num_samples(self) -> int:
        """Get the number of samples (frames) in the video."""
        return self._video.shape[0]
    
    def get_channel_names(self) -> List[str]:
        """Get the channel names."""
        return [self._selected_channel_name]

    def get_sampling_frequency(self) -> float:
        """Get the sampling frequency in Hz."""
        time_per_frame_ms = self._get_time_per_frame()
        if time_per_frame_ms is None:
            raise ValueError("Sampling frequency could not be determined from metadata.")
        return 1000.0 / time_per_frame_ms

    def _get_time_per_frame(self) -> Optional[float]:
        """Get time per frame from metadata."""
        session_key = f"MSession_{self._munit}"
        munit_key = f"MUnit_{self._munit}"
        attrs = dict(self._file[session_key][munit_key].attrs)
        return attrs.get("ZAxisConversionConversionLinearScale") #ZAxisConversionConversionLinearScale gives the duration of one frame in milliseconds.

    def get_dtype(self) -> np.dtype:
        """Get the data type of the video."""
        return self._video.dtype

    def get_frames(self, frame_idxs: ArrayType, channel: Optional[int] = 0):
        """Get specific video frames from indices."""
        if channel != 0:
            warn("Femtonics extractor extracts one channel at a time. Channel parameter ignored.")

        squeeze_data = False
        if isinstance(frame_idxs, int): # Single frame index
            squeeze_data = True
            frame_idxs = [frame_idxs]
        elif isinstance(frame_idxs, np.ndarray): # Numpy array of indices
            frame_idxs = frame_idxs.tolist()

        frames = self._video.lazy_slice[frame_idxs, :, :].dsetread()
        if squeeze_data:
            frames = frames.squeeze()
        return frames

    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
        """Get a series of frames."""
        return self._video.lazy_slice[start_sample:end_sample, :, :].dsetread()

    # Femtonics-specific getter methods
    def get_session_uuid(self):
        """
        Get the session UUID from the root attributes.

        Returns
        -------
        uuid : array or value
            The session UUID.
        """
        attrs = dict(self._file.attrs)
        return attrs.get("Uuid")

    def get_pixel_size(self) -> Tuple[float, float]:
        """Get pixel size in micrometers."""
        session_key = f"MSession_{self._munit}"
        munit_key = f"MUnit_{self._munit}"
        attrs = dict(self._file[session_key][munit_key].attrs)

        x_size = attrs.get("XAxisConversionConversionLinearScale", 1.0)
        y_size = attrs.get("YAxisConversionConversionLinearScale", 1.0)
        return x_size, y_size

    def get_image_shape_metadata(self) -> Tuple[int, int, int]:
        """
        Get the image shape metadata (X, Y, Z dimensions) from the measurement unit attributes.

        Returns
        -------
        shape : tuple
            (XDim, YDim, ZDim) as (num_columns, num_rows, num_frames)
        """
        session_key = f"MSession_{self._munit}"
        munit_key = f"MUnit_{self._munit}"
        munit_attrs = dict(self._file[session_key][munit_key].attrs)
        x_dim = munit_attrs.get("XDim")
        y_dim = munit_attrs.get("YDim")
        z_dim = munit_attrs.get("ZDim")
        return (x_dim, y_dim, z_dim)

    def get_measurement_date(self) -> Optional[datetime]:
        """Get measurement date as a timezone-aware UTC datetime."""
        session_key = f"MSession_{self._munit}"
        munit_key = f"MUnit_{self._munit}"
        attrs = dict(self._file[session_key][munit_key].attrs)

        posix_time = attrs.get("MeasurementDatePosix")
        nano_secs = attrs.get("MeasurementDateNanoSecs", 0)

        if posix_time is not None:
            # Return as UTC
            return datetime.fromtimestamp(posix_time + nano_secs / 1e9, tz=timezone.utc)
        return None

    def get_experimenter_info(self) -> Dict[str, str]:
        """Get experimenter information."""
        session_key = f"MSession_{self._munit}"
        munit_key = f"MUnit_{self._munit}"
        attrs = dict(self._file[session_key][munit_key].attrs)

        return {
            "username": self._decode_string(attrs.get("ExperimenterUsername", [])),
            "setup_id": self._decode_string(attrs.get("ExperimenterSetupID", [])),
            "hostname": self._decode_string(attrs.get("ExperimenterHostname", [])),
        }

    def get_geometric_transformations(self) -> Dict[str, np.ndarray]:
        """
        Get geometric transformations: translation, rotation, and labeling origin.

        Returns
        -------
        dict with keys 'translation', 'rotation', 'labeling_origin'
        """
        session_key = f"MSession_{self._munit}"
        munit_key = f"MUnit_{self._munit}"
        attrs = dict(self._file[session_key][munit_key].attrs)
        return {
            "translation": attrs.get("GeomTransTransl"),
            "rotation": attrs.get("GeomTransRot"),
            "labeling_origin": attrs.get("LabelingOriginTransl"),
        }

    def get_mesc_version_info(self) -> Dict[str, Any]:
        """Get MESc software version information."""
        session_key = f"MSession_{self._munit}"
        munit_key = f"MUnit_{self._munit}"
        attrs = dict(self._file[session_key][munit_key].attrs)

        return {
            "version": self._decode_string(attrs.get("CreatingMEScVersion", [])),
            "revision": attrs.get("CreatingMEScRevision"),
        }

    def get_pmt_settings(self) -> Dict[str, Dict[str, float]]:
        """Get PMT settings if available from XML metadata."""
        try:
            session_key = f"MSession_{self._munit}"
            munit_key = f"MUnit_{self._munit}"
            attrs = dict(self._file[session_key][munit_key].attrs)

            xml_data = attrs.get("MeasurementParamsXML")
            if xml_data is None:
                return {}

            xml_str = xml_data.decode("latin-1")

            root = ET.fromstring(xml_str)

            pmt_settings = {}
            for gear in root.findall('.//Gear[@type="PMT"]'):
                try:
                    channel_name = gear.find('.//param[@name="channel_name"]').get("value")
                    voltage = float(gear.find('.//param[@name="reference_voltage"]').get("value"))
                    warmup = float(gear.find('.//param[@name="warmup_time"]').get("value"))
                    pmt_settings[channel_name] = {"voltage": voltage, "warmup_time": warmup}
                except (AttributeError, ValueError):
                    continue
            return pmt_settings
        except Exception:
            return {}

    def get_scan_parameters(self) -> Dict[str, Any]:
        """Get scan parameters from XML metadata."""
        try:
            session_key = f"MSession_{self._munit}"
            munit_key = f"MUnit_{self._munit}"
            attrs = dict(self._file[session_key][munit_key].attrs)

            xml_data = attrs.get("MeasurementParamsXML")
            if xml_data is None:
                return {}

            xml_str = xml_data.decode("latin-1")

            root = ET.fromstring(xml_str)

            scan_params = {}
            for param in root.findall(".//param"):
                name = param.get("name")
                value = param.get("value")
                if name in ["SizeX", "SizeY", "PixelSizeX", "PixelSizeY", "Pixelclock"]:
                    try:
                        scan_params[name] = float(value)
                    except ValueError:
                        scan_params[name] = value
            return scan_params
        except Exception:
            return {}

    @staticmethod
    def get_available_channels(file_path: PathType, munit: int = 0) -> List[str]:
        """Get available channels in the file."""
        with h5py.File(file_path, "r") as f:
            session_key = f"MSession_{munit}"
            munit_key = f"MUnit_{munit}"

            if session_key not in f or munit_key not in f[session_key]:
                raise ValueError(f"MUnit {munit} not found in file")

            attrs = dict(f[session_key][munit_key].attrs)
            num_channels = attrs.get("VecChannelsSize", 2)

            def decode_string(arr):
                return "".join(chr(x) for x in arr if x != 0) if len(arr) > 0 else ""

            channels = []
            for i in range(num_channels):
                name_arr = attrs.get(f"Channel_{i}_Name", [])
                name = decode_string(name_arr) or f"Channel_{i}"
                channels.append(name)
            return channels

    def __del__(self):
        """Close the HDF5 file."""
        if hasattr(self, "_file"):
            self._file.close()
