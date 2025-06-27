"""Inscopix Imaging Extractor."""

import platform
import sys
import warnings
from datetime import datetime
from typing import Any, Optional

import numpy as np

from ...extraction_tools import PathType
from ...imagingextractor import ImagingExtractor


class InscopixImagingExtractor(ImagingExtractor):
    """Extracts imaging data from Inscopix recordings."""

    extractor_name = "InscopixImaging"

    def __init__(self, file_path: PathType):
        """
        Create an InscopixImagingExtractor instance from a single .isx file.

        Parameters
        ----------
        file_path : PathType
            Path to the Inscopix file.
        """
        python_version = sys.version_info
        if python_version >= (3, 13):
            raise ImportError(
                "The isx package only supports Python versions 3.9 to 3.13. "
                f"Your Python version is {python_version.major}.{python_version.minor}. "
                "See https://github.com/inscopix/pyisx for details."
            )
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            raise ImportError(
                "The isx package is currently not natively supported on macOS with Apple Silicon. "
                "Installation instructions can be found at: "
                "https://github.com/inscopix/pyisx?tab=readme-ov-file#install"
            )

        import isx

        super().__init__(file_path=file_path)
        self.movie = isx.Movie.read(str(file_path))

    def get_image_shape(self) -> tuple[int, int]:
        """Get the shape of the video frame (num_rows, num_columns).

        Returns
        -------
        image_shape: tuple
            Shape of the video frame (num_rows, num_columns).
        """
        num_pixels = self.movie.spacing.num_pixels
        return num_pixels

    def get_num_samples(self) -> int:
        return self.movie.timing.num_samples

    def get_sampling_frequency(self) -> float:
        return 1 / self.movie.timing.period.secs_float

    def get_channel_names(self) -> list[str]:
        warnings.warn("isx only supports single channel videos.")
        return ["channel_0"]

    def get_num_channels(self) -> int:
        warnings.warn("isx only supports single channel videos.")
        return 1

    def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> np.ndarray:
        start_sample = start_sample or 0
        end_sample = end_sample or self.get_num_samples()
        return np.array([self.movie.get_frame_data(i) for i in range(start_sample, end_sample)])

    def get_dtype(self) -> np.dtype:
        return np.dtype(self.movie.data_type)

    def _get_session_start_time(self) -> datetime | None:
        """
        Get the session start time as a datetime object.

        Returns
        -------
        Optional[datetime]
            The session start time if available, otherwise None.
        """
        timing = getattr(self.movie, "timing", None)
        start_time = getattr(timing, "start", None) if timing else None

        if not start_time:
            return None

        return datetime.fromisoformat(str(start_time))

    def _get_device_info(self) -> dict:
        """
        Get device-specific information including hardware settings and imaging parameters.

        Returns
        -------
        dict
            Dictionary containing device information such as microscope type, serial number,
            acquisition software version, field of view, exposure time, focus, gain, and
            LED power settings.
        """
        acq_info = self.movie.get_acquisition_info()
        device_info = {}

        # Always include field of view info
        if hasattr(self.movie, "spacing") and self.movie.spacing:
            device_info["field_of_view_pixels"] = self.get_image_shape()

        # Basic device identification
        if acq_info is not None:
            if acq_info.get("Microscope Type"):
                device_info["device_name"] = acq_info.get("Microscope Type")
            if acq_info.get("Microscope Serial Number"):
                device_info["device_serial_number"] = acq_info.get("Microscope Serial Number")
            if acq_info.get("Acquisition SW Version"):
                device_info["acquisition_software_version"] = acq_info.get("Acquisition SW Version")

            # Hardware/optical settings
            if acq_info.get("Exposure Time (ms)"):
                device_info["exposure_time_ms"] = acq_info.get("Exposure Time (ms)")
            if acq_info.get("Microscope Focus"):
                device_info["microscope_focus"] = acq_info.get("Microscope Focus")
            if acq_info.get("Microscope Gain"):
                device_info["microscope_gain"] = acq_info.get("Microscope Gain")
            if acq_info.get("efocus"):
                device_info["efocus"] = acq_info.get("efocus")

            # LED power settings - imaging has different field names than segmentation
            if acq_info.get("Microscope EX LED Power (mw/mm^2)"):
                device_info["led_power_ex_mw_per_mm2"] = acq_info.get("Microscope EX LED Power (mw/mm^2)")
            if acq_info.get("Microscope OG LED Power (mw/mm^2)"):
                device_info["led_power_og_mw_per_mm2"] = acq_info.get("Microscope OG LED Power (mw/mm^2)")

        return device_info

    def _get_subject_info(self) -> dict:
        """
        Get subject/animal information from the acquisition metadata.

        Returns
        -------
        dict
            Dictionary containing subject information such as animal ID, species, sex, weight,
            date of birth, and description.
        """
        acq_info = self.movie.get_acquisition_info()
        subject_info = {}

        if acq_info is not None:
            if acq_info.get("Animal ID"):
                subject_info["animal_id"] = acq_info.get("Animal ID")
            if acq_info.get("Animal Species"):
                subject_info["species"] = acq_info.get("Animal Species")
            if acq_info.get("Animal Sex"):
                subject_info["sex"] = acq_info.get("Animal Sex")
            if acq_info.get("Animal Weight"):
                subject_info["weight"] = acq_info.get("Animal Weight")
            if acq_info.get("Animal Date of Birth"):
                subject_info["date_of_birth"] = acq_info.get("Animal Date of Birth")
            if acq_info.get("Animal Description"):
                subject_info["description"] = acq_info.get("Animal Description")

        return subject_info

    def _get_session_info(self) -> dict:
        """
        Get session information from the acquisition metadata.

        Returns
        -------
        dict
            Dictionary containing session information : session name, and experimenter name.
        """
        info = {}

        acq_info = self.movie.get_acquisition_info()
        if acq_info is not None:
            if acq_info.get("Session Name"):
                info["session_name"] = acq_info.get("Session Name")
            if acq_info.get("Experimenter Name"):
                info["experimenter_name"] = acq_info.get("Experimenter Name")

        return info

    def _get_probe_info(self) -> dict:
        """
        Get probe information from the acquisition metadata.

        Returns
        -------
        dict
            Dictionary containing probe information such as diameter, flip, length, pitch, rotation, and type.
            Only includes fields with non-empty, non-zero, and non-'none' values.
        """
        probe_info = {}
        acq_info = self.movie.get_acquisition_info()

        if acq_info is not None:
            probe_fields = [
                "Probe Diameter (mm)",
                "Probe Flip",
                "Probe Length (mm)",
                "Probe Pitch",
                "Probe Rotation (degrees)",
                "Probe Type",
            ]

            for field in probe_fields:
                value = acq_info.get(field)
                # Include value if it's not None, empty string, 0, or string variations of "none"
                if value is not None and value != "" and value != 0 and str(value).lower() != "none":
                    probe_info[field] = value

        return probe_info

    def _get_metadata(self) -> dict[str, Any]:
        """
        Get all available metadata in a single dictionary.

        This method consolidates all metadata extraction methods into one call
        for convenience when extracting metadata in neuroconv.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all extractable metadata including:
            - device: Device/hardware information
            - subject: Subject/animal information
            - session: Session information
            - probe: Probe information
            - session_start_time: Measurement start time as datetime
        """
        metadata = {}

        # Device info
        try:
            metadata["device"] = self._get_device_info()
        except Exception:
            metadata["device"] = {}

        # Subject info
        try:
            metadata["subject"] = self._get_subject_info()
        except Exception:
            metadata["subject"] = {}

        # Analysis info
        try:
            metadata["analysis"] = self._get_analysis_info()
        except Exception:
            metadata["analysis"] = {}

        # Session info
        try:
            metadata["session"] = self._get_session_info()
        except Exception:
            metadata["session"] = {}

        # Probe info
        try:
            metadata["probe"] = self._get_probe_info()
        except Exception:
            metadata["probe"] = {}

        # Session start time
        try:
            metadata["session_start_time"] = self._get_session_start_time()
        except Exception:
            metadata["session_start_time"] = None

        return metadata

    def get_native_timestamps(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> Optional[np.ndarray]:
        # Inscopix imaging data does not have native timestamps
        return None
