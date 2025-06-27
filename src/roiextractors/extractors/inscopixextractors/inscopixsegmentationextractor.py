"""Inscopix Segmentation Extractor."""

import platform
import sys
import warnings
from datetime import datetime
from typing import Any, Optional

import numpy as np

from ...extraction_tools import ArrayType, PathType
from ...segmentationextractor import SegmentationExtractor


class InscopixSegmentationExtractor(SegmentationExtractor):
    """A segmentation extractor for Inscopix."""

    extractor_name = "InscopixSegmentationExtractor"

    def __init__(self, file_path: PathType):
        """Initialize a InscopixSegmentationExtractor instance.

        Main class for extracting segmentation data from Inscopix format.

        Parameters
        ----------
        file_path: str or Path
            The location of the folder containing Inscopix *.mat output file.
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

        SegmentationExtractor.__init__(self)
        self._file_path = file_path
        file_path_str = str(file_path)

        self.cell_set = isx.CellSet.read(file_path_str, read_only=True)

        # Get original IDs directly from CellSet
        self._roi_ids = [self.cell_set.get_cell_name(x) for x in range(self.cell_set.num_cells)]

        # Cache for metadata to avoid repeated extraction
        self._metadata_cache = None

        # Set sampling frequency
        self._sampling_frequency = 1 / self.cell_set.timing.period.secs_float

    def get_num_rois(self) -> int:
        return self.cell_set.num_cells

    def get_roi_image_masks(self, roi_ids: Optional[list] = None) -> np.ndarray:
        """Get image masks for the specified ROIs.

        Parameters
        ----------
        roi_ids : list or None
            List of ROI IDs (can be integers or original string IDs)
        If None, all ROIs will be returned.

        Returns
        -------
        np.ndarray
            Image masks for the specified ROIs
        """
        if roi_ids is None:
            roi_indices = list(range(self.get_num_rois()))
        else:
            all_roi_ids = self.get_roi_ids()
            roi_indices = [all_roi_ids.index(roi_id) for roi_id in roi_ids]

        masks = [self.cell_set.get_cell_image_data(roi_idx) for roi_idx in roi_indices]
        if len(masks) == 1:
            return masks[0]
        return np.stack(masks)

    def get_roi_pixel_masks(self, roi_ids: Optional[list] = None) -> list[np.ndarray]:
        """Get pixel masks for the specified ROIs.

        This converts the image masks to pixel masks with the format expected by the NWB standard.

        Parameters
        ----------
        roi_ids : list or None
            List of ROI IDs (can be integers or original string IDs)

        Returns
        -------
        list
            List of pixel masks, each with shape (N, 3) where N is the number of pixels in the ROI.
            Each row is (x, y, weight).
        """
        if roi_ids is None:
            roi_ids = self.get_roi_ids()

        # Get image masks
        image_masks = self.get_roi_image_masks(roi_ids=roi_ids)

        # Handle case when only one ROI ID is specified
        if len(roi_ids) == 1:
            image_masks = [image_masks]

        # Convert image masks to pixel masks
        pixel_masks = []
        for mask in image_masks:
            # Find non-zero pixels in the mask
            y_indices, x_indices = np.where(mask > 0)

            if len(x_indices) > 0:
                # Use the mask values as weights
                weights = mask[y_indices, x_indices]
                # Create pixel mask with (x, y, weight) format
                pixel_mask = np.column_stack((x_indices, y_indices, weights))
            else:
                # For empty ROIs, create a dummy pixel mask with correct shape
                pixel_mask = np.array([[0, 0, 1.0]])

            pixel_masks.append(pixel_mask)

        return pixel_masks

    def get_roi_ids(self) -> list:
        """Get ROI IDs as original string IDs from the CellSet."""
        return self._roi_ids.copy()

    def get_frame_shape(self) -> tuple[int, int]:
        if hasattr(self.cell_set, "spacing"):
            # Swap dimensions to return (width, height)
            pixels = self.cell_set.spacing.num_pixels
            return (pixels[1], pixels[0])
        else:
            if self.get_num_rois() > 0:
                shape = self.cell_set.get_cell_image_data(0).shape
                # Swap dimensions to return (width, height)
                return (shape[1], shape[0])
            raise ValueError("No ROIs found in the segmentation. Unable to determine image size.")

    def get_image_size(self) -> ArrayType:
        warnings.warn(
            "get_image_size is deprecated and will be removed on or after January 2026. "
            "Use get_frame_shape instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_frame_shape()

    def get_accepted_list(self) -> list:
        """Get list of accepted ROI IDs (as string IDs)."""
        accepted = []
        for i, original_id in enumerate(self._roi_ids):
            if self.cell_set.get_cell_status(i) == "accepted":
                accepted.append(original_id)  # Return string IDs
        return accepted

    def get_rejected_list(self) -> list:
        """Get list of rejected ROI IDs (as string IDs)."""
        rejected = []
        for i, original_id in enumerate(self._roi_ids):
            if self.cell_set.get_cell_status(i) == "rejected":
                rejected.append(original_id)  # Return string IDs
        return rejected

    def get_traces(self, roi_ids=None, start_frame=None, end_frame=None, name="raw") -> ArrayType:
        """Get traces for the specified ROIs.

        Parameters
        ----------
        roi_ids : list or None
            List of ROI IDs (can be integers or string IDs)
        start_frame : int or None
            Start frame index
        end_frame : int or None
            End frame index
        name : str
            Name of the trace type

        Returns
        -------
        np.ndarray
            Traces for the specified ROIs
        """
        if roi_ids is None:
            roi_indices = list(range(self.get_num_rois()))
        else:
            all_roi_ids = self.get_roi_ids()
            roi_indices = [all_roi_ids.index(roi_id) for roi_id in roi_ids]

        return np.vstack([self.cell_set.get_cell_trace_data(roi_idx)[start_frame:end_frame] for roi_idx in roi_indices])

    def get_num_samples(self) -> int:
        """Get the number of samples in the recording (duration of recording).

        Returns
        -------
        num_samples: int
            Number of samples in the recording.
        """
        try:
            return self.cell_set.timing.num_samples
        except AttributeError:
            if self.get_num_rois() > 0:
                return len(self.cell_set.get_cell_trace_data(0))
            return 0

    def get_num_frames(self) -> int:
        warnings.warn(
            "get_num_frames is deprecated and will be removed on or after January 2026. "
            "Use get_num_samples instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_num_samples()

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def _get_session_start_time(self) -> datetime | None:
        """
        Get the session start time as a datetime object.

        Returns
        -------
        Optional[datetime]
            The session start time if available, otherwise None.
        """
        timing = getattr(self.cell_set, "timing", None)
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
            acquisition software version, field of view, exposure time, focus, gain, channel,
            efocus, and LED power settings.
        """
        acq_info = self.cell_set.get_acquisition_info()
        device_info = {}

        # Handle case where acquisition info is None (empty cell sets)
        if acq_info is None:
            return device_info

        # Basic device identification
        if acq_info.get("Microscope Type"):
            device_info["device_name"] = acq_info.get("Microscope Type")
        if acq_info.get("Microscope Serial Number"):
            device_info["device_serial_number"] = acq_info.get("Microscope Serial Number")
        if acq_info.get("Acquisition SW Version"):
            device_info["acquisition_software_version"] = acq_info.get("Acquisition SW Version")

        # Imaging/acquisition parameters
        if hasattr(self.cell_set, "spacing") and self.cell_set.spacing:
            device_info["field_of_view_pixels"] = self.cell_set.spacing.num_pixels

        # Hardware/optical settings
        if acq_info.get("Exposure Time (ms)"):
            device_info["exposure_time_ms"] = acq_info.get("Exposure Time (ms)")
        if acq_info.get("Microscope Focus"):
            device_info["microscope_focus"] = acq_info.get("Microscope Focus")
        if acq_info.get("Microscope Gain"):
            device_info["microscope_gain"] = acq_info.get("Microscope Gain")
        if acq_info.get("channel"):
            device_info["channel"] = acq_info.get("channel")
        if acq_info.get("efocus"):
            device_info["efocus"] = acq_info.get("efocus")
        if acq_info.get("Microscope EX LED 1 Power (mw/mm^2)"):
            device_info["led_power_1_mw_per_mm2"] = acq_info.get("Microscope EX LED 1 Power (mw/mm^2)")
        if acq_info.get("Microscope EX LED 2 Power (mw/mm^2)"):
            device_info["led_power_2_mw_per_mm2"] = acq_info.get("Microscope EX LED 2 Power (mw/mm^2)")

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
        acq_info = self.cell_set.get_acquisition_info()
        subject_info = {}

        # Handle case where acquisition info is None (empty cell sets)
        if acq_info is None:
            return subject_info

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

    def _get_analysis_info(self) -> dict:
        """
        Get analysis method information specific to Inscopix Segmentation.

        Returns
        -------
        dict
            Dictionary containing analysis information such as cell identification method and trace units.
        """
        acq_info = self.cell_set.get_acquisition_info()
        analysis_info = {}

        # Handle case where acquisition info is None (empty cell sets)
        if acq_info is None:
            return analysis_info

        if acq_info.get("Cell Identification Method"):
            analysis_info["cell_identification_method"] = acq_info.get("Cell Identification Method")
        if acq_info.get("Trace Units"):
            analysis_info["trace_units"] = acq_info.get("Trace Units")

        return analysis_info

    def _get_session_info(self) -> dict:
        """
        Get session information from the acquisition metadata.

        Returns
        -------
        dict
            Dictionary containing session information such as session name, and experimenter name.
        """
        info = {}

        acq_info = self.cell_set.get_acquisition_info()

        # Handle case where acquisition info is None (empty cell sets)
        if acq_info is None:
            return info

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
        acq_info = self.cell_set.get_acquisition_info()

        # Handle case where acquisition info is None (empty cell sets)
        if acq_info is None:
            return probe_info

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
            - analysis: Analysis method information
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
        # Inscopix segmentation data does not have native timestamps
        return None
