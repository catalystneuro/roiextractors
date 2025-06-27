"""A Femtonics imaging extractor with corrected MSession/MUnit hierarchy handling."""

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from warnings import warn

import h5py
import numpy as np
from lazy_ops import DatasetView

from ...extraction_tools import PathType
from ...imagingextractor import ImagingExtractor


class FemtonicsImagingExtractor(ImagingExtractor):
    """A Femtonics imaging extractor with corrected MSession/MUnit hierarchy handling."""

    extractor_name = "FemtonicsImaging"

    def __init__(
        self,
        file_path: PathType,
        session_index: int = 0,
        munit_index: int = 0,
        channel_name: Optional[str] = None,
    ):
        """Create a FemtonicsImagingExtractor from a MESc (.mesc) file.

        MESc (Measurement Session Container) is the HDF5-based file format used by Femtonics two-photon microscopes.
        A MESc file (.mesc) contains imaging data, experiment metadata, scan parameters, and hardware configuration
        for one or more imaging runs ("Measurement Units", or MUnits).

        Parameters
        ----------
        file_path : str or Path
            Path to the .mesc file.
        session_index : int, optional
            Index of the MSession to use (0-based). Default is 0.
        munit_index : int, optional
            Index of the MUnit within the specified session (0-based). Default is 0.

            In Femtonics MESc files, an MUnit ("Measurement Unit") represents a single imaging acquisition or experiment,
            including all associated imaging data and metadata. A single MSession can contain multiple MUnits,
            each corresponding to a separate imaging run/experiment performed during the session.
            MUnits are indexed (0, 1, 2, ...) within each session.

        channel_name : str, optional
            Name of the channel to extract (e.g., 'UG', 'UR').
            If multiple channels are available and no channel is specified, an error will be raised.
            If only one channel is available, it will be used automatically.
        """
        super().__init__(file_path=file_path)

        self.file_path = Path(file_path)
        self._session_index = session_index
        self._munit_index = munit_index
        self._channel_name = channel_name

        if self.file_path.suffix != ".mesc":
            warn("File is not a .mesc file")

        # Open file and setup basic access
        self._file_handle = h5py.File(file_path, "r")
        self._setup_hierarchy_access()
        self._setup_channel_selection()
        self._setup_video_data()
        self._setup_sampling_frequency()

    def _setup_hierarchy_access(self):
        """Set up proper MSession/MUnit hierarchy access."""
        # Validate session exists
        self._session_key = f"MSession_{self._session_index}"
        if self._session_key not in self._file_handle:
            available_sessions = [k for k in self._file_handle.keys() if k.startswith("MSession_")]
            raise ValueError(
                f"Session index {self._session_index} not found. " f"Available sessions: {available_sessions}"
            )

        # Get available units in this session
        session_group = self._file_handle[self._session_key]
        available_units = [k for k in session_group.keys() if k.startswith("MUnit_")]

        if self._munit_index >= len(available_units):
            raise ValueError(
                f"Unit index {self._munit_index} not found in {self._session_key}. "
                f"Available units: {available_units}"
            )

        # Set the actual unit key (not assuming index matches)
        self._munit_key = available_units[self._munit_index]

    def _setup_channel_selection(self):
        """
        Select the channel to extract with proper hierarchy handling.

        - Gets available channels for the selected MSession/MUnit combination.
        - If no channels, raises an error.
        - If channel_name is not given and multiple channels exist, raises an error listing options.
        - If channel_name is given, checks it exists; otherwise, raises an error.
        - If only one channel, selects it automatically.
        """
        available_channels = self._get_available_channels_from_file()
        if not available_channels:
            raise ValueError(f"No channels found in {self._session_key}/{self._munit_key}.")

        if self._channel_name is None:
            if len(available_channels) > 1:
                raise ValueError(
                    f"Multiple channels found in {self._session_key}/{self._munit_key}: {available_channels}. "
                    "Please specify 'channel_name' to select one."
                )
            self._selected_channel_name = available_channels[0]
        else:
            if self._channel_name not in available_channels:
                raise ValueError(
                    f"Channel '{self._channel_name}' not found in {self._session_key}/{self._munit_key}. "
                    f"Available: {available_channels}"
                )
            self._selected_channel_name = self._channel_name

    def _setup_video_data(self):
        """Set up access to the actual imaging data for the selected session, unit, and channel."""
        # Find the channel index by looking for the channel name in the HDF5 structure
        available_channels = self._get_available_channels_from_file()
        channel_index = available_channels.index(self._selected_channel_name)
        channel_key = f"Channel_{channel_index}"

        channel_path = f"{self._session_key}/{self._munit_key}/{channel_key}"

        if channel_key in self._file_handle[self._session_key][self._munit_key]:
            self._video = DatasetView(self._file_handle[self._session_key][self._munit_key][channel_key])
        else:
            raise Exception(f"Cannot find data at {channel_path}")

    def _setup_sampling_frequency(self):
        """Cache the sampling frequency during initialization to avoid repeated file access."""
        time_per_frame_ms = self._get_frame_duration_in_milliseconds()
        if time_per_frame_ms is None:
            raise ValueError(
                f"Sampling frequency could not be determined from metadata in "
                f"{self._session_key}/{self._munit_key}."
            )
        self._sampling_frequency = 1000.0 / time_per_frame_ms

    def _get_available_channels_from_file(self) -> list[str]:
        """Get available channels from the current session/unit combination."""
        if self._session_key not in self._file_handle or self._munit_key not in self._file_handle[self._session_key]:
            raise ValueError(f"Session {self._session_index}/Unit {self._munit_index} not found in file")

        attrs = dict(self._file_handle[self._session_key][self._munit_key].attrs)
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

    def get_image_shape(self) -> tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns)."""
        return self._video.shape[1], self._video.shape[2]  # (height, width)

    def get_num_samples(self) -> int:
        """Get the number of samples (frames) in the video."""
        return self._video.shape[0]

    def get_channel_names(self) -> list[str]:
        """Get the channel names."""
        return [self._selected_channel_name]

    def get_sampling_frequency(self) -> float:
        """Get the sampling frequency in Hz."""
        return self._sampling_frequency

    def _get_metadata(self) -> dict[str, Any]:
        """
        Get all available metadata in a single dictionary.

        This method consolidates all metadata extraction methods into one call
        for convenience when extracting metadata in neuroconv.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all extractable metadata including:
            - session_uuid: Session UUID
            - pixel_size_micrometers: (x_size, y_size) in micrometers
            - image_shape_metadata: (XDim, YDim, ZDim) from metadata
            - session_start_time: Measurement start time as datetime
            - experimenter_info: Dict with username, setup_id, hostname
            - geometric_transformations: Dict with translation, rotation, labeling_origin
            - mesc_version_info: Dict with version and revision
            - pmt_settings: Dict with PMT settings per channel
            - scan_parameters: Dict with scan parameters from XML
            - frame_duration_ms: Time per frame in milliseconds
            - sampling_frequency_hz: Sampling frequency in Hz
            - selected_channel: Currently selected channel name
            - available_channels: List of all available channels
            - session_index: Current session index
            - munit_index: Current MUnit index
        """
        metadata = {}

        # Basic identifiers
        metadata["session_index"] = self._session_index
        metadata["munit_index"] = self._munit_index
        metadata["selected_channel"] = self._selected_channel_name
        metadata["available_channels"] = self._get_available_channels_from_file()

        # Session UUID
        try:
            metadata["session_uuid"] = self._get_session_uuid()
        except Exception:
            metadata["session_uuid"] = None

        # Pixel size
        try:
            metadata["pixel_size_micrometers"] = self._get_pixels_sizes_and_units()
        except Exception:
            metadata["pixel_size_micrometers"] = None

        # Image shape from metadata
        try:
            metadata["image_shape_metadata"] = self._get_image_shape_metadata()
        except Exception:
            metadata["image_shape_metadata"] = None

        # Session start time
        try:
            metadata["session_start_time"] = self._get_session_start_time()
        except Exception:
            metadata["session_start_time"] = None

        # Experimenter info
        try:
            metadata["experimenter_info"] = self._get_experimenter_info()
        except Exception:
            metadata["experimenter_info"] = {}

        # Geometric transformations
        try:
            metadata["geometric_transformations"] = self._get_geometric_transformations()
        except Exception:
            metadata["geometric_transformations"] = {}

        # MESc version info
        try:
            metadata["mesc_version_info"] = self._get_mesc_version_info()
        except Exception:
            metadata["mesc_version_info"] = {}

        # PMT settings
        try:
            metadata["pmt_settings"] = self._get_pmt_settings()
        except Exception:
            metadata["pmt_settings"] = {}

        # Scan parameters
        try:
            metadata["scan_parameters"] = self._get_scan_parameters()
        except Exception:
            metadata["scan_parameters"] = {}

        # Timing information
        try:
            metadata["frame_duration_ms"] = self._get_frame_duration_in_milliseconds()
        except Exception:
            metadata["frame_duration_ms"] = None

        metadata["sampling_frequency_hz"] = self._sampling_frequency

        return metadata

    # Femtonics-specific getter methods
    def _get_session_uuid(self) -> Optional[str]:
        """Get the session UUID from the root attributes as a hex string."""
        attrs = dict(self._file_handle.attrs)
        uuid_arr = attrs.get("Uuid")
        if uuid_arr is not None:
            # Convert to hex format
            hex_str = "".join(f"{x:02x}" for x in uuid_arr)
            return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"
        return None

    def _get_frame_duration_in_milliseconds(self) -> Optional[float]:
        """Get time per frame from metadata, ensuring units are in milliseconds."""
        attrs = dict(self._file_handle[self._session_key][self._munit_key].attrs)

        frame_duration = attrs.get("ZAxisConversionConversionLinearScale")
        if frame_duration is None:
            return None

        # Check units to ensure they are milliseconds
        z_units = self._decode_string(attrs.get("ZAxisConversionUnitName", []))

        if z_units:
            z_units_lower = z_units.lower()
            is_milliseconds = any(
                ms_variant in z_units_lower for ms_variant in ["ms", "millisec", "millisecond", "msec"]
            )

            if not is_milliseconds:
                raise ValueError(
                    f"Z-axis (time) units are '{z_units}', expected milliseconds. "
                    f"Cannot reliably calculate sampling frequency. "
                    f"Frame duration value: {frame_duration}"
                )
        else:
            # If no units specified, warn but assume milliseconds for backward compatibility
            warn(
                "No Z-axis (time) unit specified in metadata; assuming milliseconds (ms) for sampling frequency calculation."
            )

        return frame_duration

    def _get_pixels_sizes_and_units(self) -> dict[str, Any]:
        """Get pixel size and units from metadata."""
        attrs = dict(self._file_handle[self._session_key][self._munit_key].attrs)

        x_size = attrs.get("XAxisConversionConversionLinearScale", 1.0)
        y_size = attrs.get("YAxisConversionConversionLinearScale", 1.0)
        x_units = self._decode_string(attrs.get("XAxisConversionUnitName", []))
        y_units = self._decode_string(attrs.get("YAxisConversionUnitName", []))

        return {"x_size": x_size, "y_size": y_size, "x_units": x_units, "y_units": y_units}

    def _get_image_shape_metadata(self) -> tuple[int, int, int]:
        """
        Get the image shape metadata (X, Y, Z dimensions) from the measurement unit attributes.

        Returns
        -------
        shape : tuple
            (XDim, YDim, ZDim) as (num_columns, num_rows, num_frames)
        """
        munit_attrs = dict(self._file_handle[self._session_key][self._munit_key].attrs)
        x_dim = int(munit_attrs.get("XDim"))
        y_dim = int(munit_attrs.get("YDim"))
        z_dim = int(munit_attrs.get("ZDim"))
        return (x_dim, y_dim, z_dim)

    def _get_session_start_time(self) -> Optional[datetime]:
        """Get measurement date as a timezone-aware UTC datetime."""
        attrs = dict(self._file_handle[self._session_key][self._munit_key].attrs)

        posix_time = attrs.get("MeasurementDatePosix")
        nano_secs = attrs.get("MeasurementDateNanoSecs", 0)

        if posix_time is not None:
            # Return as UTC
            return datetime.fromtimestamp(posix_time + nano_secs / 1e9, tz=timezone.utc)
        return None

    def _get_experimenter_info(self) -> dict[str, str]:
        """Get experimenter information."""
        attrs = dict(self._file_handle[self._session_key][self._munit_key].attrs)

        return {
            "username": self._decode_string(attrs.get("ExperimenterUsername", [])),
            "setup_id": self._decode_string(attrs.get("ExperimenterSetupID", [])),
            "hostname": self._decode_string(attrs.get("ExperimenterHostname", [])),
        }

    def _get_geometric_transformations(self) -> dict[str, np.ndarray]:
        """
        Get geometric transformations: translation, rotation, and labeling origin.

        Returns
        -------
        dict with keys 'translation', 'rotation', 'labeling_origin'
        """
        attrs = dict(self._file_handle[self._session_key][self._munit_key].attrs)
        return {
            "translation": attrs.get("GeomTransTransl"),
            "rotation": attrs.get("GeomTransRot"),
            "labeling_origin": attrs.get("LabelingOriginTransl"),
        }

    def _get_mesc_version_info(self) -> dict[str, Any]:
        """Get MESc (Measurement Session Container) software version information."""
        attrs = dict(self._file_handle[self._session_key][self._munit_key].attrs)

        return {
            "version": self._decode_string(attrs.get("CreatingMEScVersion", [])),
            "revision": attrs.get("CreatingMEScRevision"),
        }

    def _get_pmt_settings(self) -> dict[str, dict[str, float]]:
        """Get photomultiplier tube (PMT) settings if available from XML metadata."""
        try:
            attrs = dict(self._file_handle[self._session_key][self._munit_key].attrs)

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

    def _get_scan_parameters(self) -> dict[str, Any]:
        """Get scan parameters from XML metadata."""
        try:
            attrs = dict(self._file_handle[self._session_key][self._munit_key].attrs)

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
    def get_available_channels(file_path: PathType, session_index: int = 0, munit_index: int = 0) -> list[str]:
        """
        Get available channels in the specified session/unit combination.

        Parameters
        ----------
        file_path : str or Path
            Path to the .mesc file.
        session_index : int, optional
            Index of the MSession to use. Default is 0.
        munit_index : int, optional
            Index of the MUnit within the session. Default is 0.

        Returns
        -------
        list of str
            List of available channel names.
        """
        with h5py.File(file_path, "r") as f:
            session_key = f"MSession_{session_index}"

            if session_key not in f:
                raise ValueError(f"Session index {session_index} not found in file")

            # Get available units in this session
            session_group = f[session_key]
            available_units = [k for k in session_group.keys() if k.startswith("MUnit_")]

            if munit_index >= len(available_units):
                raise ValueError(
                    f"Unit index {munit_index} not found in {session_key}. " f"Available units: {available_units}"
                )

            munit_key = available_units[munit_index]

            attrs = dict(f[session_key][munit_key].attrs)
            num_channels = attrs.get("VecChannelsSize", 2)

            decode_string = lambda arr: "".join(chr(x) for x in arr if x != 0) if len(arr) > 0 else ""

            channels = []
            for i in range(num_channels):
                name_arr = attrs.get(f"Channel_{i}_Name", [])
                name = decode_string(name_arr) or f"Channel_{i}"
                channels.append(name)
            return channels

    @staticmethod
    def get_available_sessions(file_path: PathType) -> list[str]:
        """Get list of available session keys in the file."""
        with h5py.File(file_path, "r") as f:
            return [k for k in f.keys() if k.startswith("MSession_")]

    @staticmethod
    def get_available_units(file_path: PathType, session_index: int = 0) -> list[str]:
        """Get list of available unit keys in the specified session."""
        with h5py.File(file_path, "r") as f:
            session_key = f"MSession_{session_index}"
            if session_key not in f:
                return []
            return [k for k in f[session_key].keys() if k.startswith("MUnit_")]

    def get_dtype(self) -> np.dtype:
        """Get the data type of the samples."""
        return self._video.dtype

    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
        """Get a series of samples."""
        return self._video.lazy_slice[start_sample:end_sample, :, :].dsetread()

    def get_samples(self, sample_indices) -> np.ndarray:
        """Get specific samples by indices."""
        return self._video.lazy_slice[sample_indices, :, :].dsetread()

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # Femtonics data does not have native timestamps
        return None

    def __del__(self):
        """Close the HDF5 file."""
        if hasattr(self, "_file_handle"):
            self._file_handle.close()
