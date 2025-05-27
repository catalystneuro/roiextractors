"""Inscopix Segmentation Extractor."""

from typing import Optional, List
import platform
import numpy as np
from datetime import datetime

from ...extraction_tools import PathType, ArrayType
from ...segmentationextractor import SegmentationExtractor


class InscopixSegmentationExtractor(SegmentationExtractor):
    """A segmentation extractor for Inscopix."""

    extractor_name = "InscopixSegmentationExtractor"

    def __init__(self, file_path: PathType, verbose: bool = True):
        """Initialize a InscopixSegmentationExtractor instance.

        Main class for extracting segmentation data from Inscopix format.

        Parameters
        ----------
        file_path: str or PathType
            The location of the folder containing Inscopix *.mat output file.
        verbose: bool, default True
            Whether to print verbose output for warnings and errors.
        """
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            raise ImportError(
                "For macOS ARM64, please use a special conda environment setup. " "See README for instructions."
            )

        import isx

        SegmentationExtractor.__init__(self)
        self.file_path = file_path
        self.verbose = verbose
        file_path_str = str(file_path)

        self.cell_set = isx.CellSet.read(file_path_str)

        # Create mappings between original IDs and integer IDs
        self._original_ids = [self.cell_set.get_cell_name(x) for x in range(self.cell_set.num_cells)]
        self._id_to_index = {id: i for i, id in enumerate(self._original_ids)}
        self._index_to_id = {i: id for id, i in self._id_to_index.items()}

        # Cache for metadata to avoid repeated extraction
        self._metadata_cache = None

    def get_num_rois(self) -> int:
        return self.cell_set.num_cells

    def _get_roi_indices(self, roi_ids=None) -> List[int]:
        """Convert ROI IDs to indices (positions in the original CellSet).

        Handle both string IDs (e.g., 'C0') and integer IDs (e.g., 0).
    
        Parameters
        ----------
        roi_ids : list or None
            List of ROI IDs (can be integers or original string IDs)
        
        Returns
        -------
        List[int]
            List of indices corresponding to the ROI IDs
        

        """
        if roi_ids is None:
            return list(range(self.get_num_rois()))

        indices = []
        max_rois = self.get_num_rois()

        for roi_id in roi_ids:
            if isinstance(roi_id, int):
                if 0 <= roi_id < max_rois:
                    indices.append(roi_id)
                else:
                    raise ValueError(f"ROI index {roi_id} out of range [0, {max_rois-1}]")
            elif isinstance(roi_id, str):
                if roi_id in self._id_to_index:
                    indices.append(self._id_to_index[roi_id])
                else:
                    raise ValueError(f"ROI ID '{roi_id}' not found. Available IDs: {list(self._id_to_index.keys())[:5]}...")
            else:
                raise ValueError(f"ROI ID must be int or str, got {type(roi_id)}: {roi_id}")

        return indices

    def get_roi_image_masks(self, roi_ids=None) -> np.ndarray:
        """Get image masks for the specified ROIs.

        Parameters
        ----------
        roi_ids : list or None
            List of ROI IDs (can be integers or original string IDs)

        Returns
        -------
        np.ndarray
            Image masks for the specified ROIs
        """
        roi_indices = self._get_roi_indices(roi_ids)

        masks = [self.cell_set.get_cell_image_data(roi_idx) for roi_idx in roi_indices]
        if len(masks) == 1:
            return masks[0]
        return np.stack(masks)

    def get_roi_pixel_masks(self, roi_ids=None) -> List[np.ndarray]:
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
        # Get image masks
        image_masks = self.get_roi_image_masks(roi_ids=roi_ids)

        # Handle case when only one ROI ID is specified
        if roi_ids is not None and (not isinstance(roi_ids, list) or len(roi_ids) == 1):
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
        """Get ROI IDs as integers (0, 1, 2, ...)."""
        return list(range(self.get_num_rois()))

    def get_original_roi_ids(self) -> list:
        """Get original ROI IDs from the CellSet."""
        return self._original_ids.copy()

    def get_image_size(self) -> ArrayType:
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

    def get_accepted_list(self) -> list:
        """Get list of accepted ROI IDs (as integers)."""
        accepted = []
        for i, original_id in enumerate(self._original_ids):
            idx = self._original_ids.index(original_id)
            if self.cell_set.get_cell_status(idx) == "accepted":
                accepted.append(i)  # Return integer IDs
        return accepted

    def get_rejected_list(self) -> list:
        """Get list of rejected ROI IDs (as integers)."""
        rejected = []
        for i, original_id in enumerate(self._original_ids):
            idx = self._original_ids.index(original_id)
            if self.cell_set.get_cell_status(idx) == "rejected":
                rejected.append(i)  # Return integer IDs
        return rejected

    def get_traces(self, roi_ids=None, start_frame=None, end_frame=None, name="raw") -> ArrayType:
        """Get traces for the specified ROIs.

        Parameters
        ----------
        roi_ids : list or None
            List of ROI IDs (can be integers or original string IDs)
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
        roi_indices = self._get_roi_indices(roi_ids)

        return np.vstack([self.cell_set.get_cell_trace_data(roi_idx)[start_frame:end_frame] for roi_idx in roi_indices])

    def get_num_frames(self) -> int:
        try:
            return self.cell_set.timing.num_samples
        except AttributeError:
            if self.get_num_rois() > 0:
                return len(self.cell_set.get_cell_trace_data(0))
            return 0

    def get_sampling_frequency(self) -> float:
        try:
            return 1 / self.cell_set.timing.period.secs_float
        except AttributeError:
            return None

    def _get_inscopix_metadata(self) -> dict:
        """Extract comprehensive metadata from the Inscopix file.

        Returns
        -------
        dict
            Dictionary containing all available metadata from the Inscopix file.
        """

        try:
            metadata = {}

            # Session timing info
            if self.cell_set.timing:
                metadata["session"] = {
                    "start_time": self.cell_set.timing.start,
                    "duration_seconds": self.cell_set.timing.num_samples
                    * self.cell_set.timing.period.to_msecs()
                    / 1000,
                    "num_samples": self.cell_set.timing.num_samples,
                    "sampling_period_ms": self.cell_set.timing.period.to_msecs(),
                    "sampling_rate_hz": self.get_sampling_frequency(),
                }

            # Imaging info
            if self.cell_set.spacing:
                metadata["imaging"] = {
                    "field_of_view_pixels": self.cell_set.spacing.num_pixels,
                    "num_cells": self.cell_set.num_cells,
                }

            # Acquisition info
            try:
                acq_info = self.cell_set.get_acquisition_info()

                # Device info
                metadata["device"] = {
                    "device_name": acq_info.get("Microscope Type"),
                    "device_serial_number": acq_info.get("Microscope Serial Number"),
                    "acquisition_software_version": acq_info.get("Acquisition SW Version"),
                }

                # update imaging info
                if "imaging" not in metadata:
                    metadata["imaging"] = {}
                metadata["imaging"].update(
                    {
                        "exposure_time_ms": acq_info.get("Exposure Time (ms)"),
                        "microscope_focus": acq_info.get("Microscope Focus"),
                        "microscope_gain": acq_info.get("Microscope Gain"),
                        "channel": acq_info.get("channel"),
                        "efocus": acq_info.get("efocus"),
                        "led_power_1_mw_per_mm2": acq_info.get("Microscope EX LED 1 Power (mw/mm^2)"),
                        "led_power_2_mw_per_mm2": acq_info.get("Microscope EX LED 2 Power (mw/mm^2)"),
                    }
                )

                # update session info
                if "session" not in metadata:
                    metadata["session"] = {}
                metadata["session"].update(
                    {
                        "session_name": acq_info.get("Session Name"),
                        "experimenter_name": acq_info.get("Experimenter Name"),
                    }
                )

                # Subject info
                metadata["subject"] = {
                    "animal_id": acq_info.get("Animal ID"),
                    "species_strain": acq_info.get("Animal Species"),
                    "sex": acq_info.get("Animal Sex"),
                    "weight": acq_info.get("Animal Weight"),
                    "date_of_birth": acq_info.get("Animal Date of Birth"),
                    "description": acq_info.get("Animal Description"),
                }

                # Analysis info
                metadata["analysis"] = {
                    "cell_identification_method": acq_info.get("Cell Identification Method"),
                    "trace_units": acq_info.get("Trace Units"),
                }

                # Probe info
                probe_fields = [
                    "Probe Diameter (mm)",
                    "Probe Flip",
                    "Probe Length (mm)",
                    "Probe Pitch",
                    "Probe Rotation (degrees)",
                    "Probe Type",
                ]
                probe_info = {}
                for field in probe_fields:
                    value = acq_info.get(field)
                    if value not in (None, "", 0, "None", "none"):
                        probe_info[field] = value
                if probe_info:
                    metadata["probe"] = probe_info

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not extract acquisition info: {e}")
                # empty dicts for missing sections
                metadata.update(
                    {
                        "device": {},
                        "subject": {},
                        "analysis": {},
                    }
                )

            self._metadata_cache = metadata
            return metadata

        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not extract Inscopix metadata: {e}")
            return {}

    def get_session_start_time(self) -> Optional[datetime]:
        """Get the session start time as a datetime object.

        Returns
        -------
        datetime or None
            Session start time, or None if not available or cannot be parsed.
        """
        metadata = self._get_inscopix_metadata()
        session_info = metadata.get("session", {})
        start_time = session_info.get("start_time")

        if not start_time:
            return None

        try:
            if isinstance(start_time, str):
                if "T" in start_time:
                    return datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                return datetime.fromisoformat(start_time)
            elif hasattr(start_time, "year"):
                return start_time
            elif hasattr(start_time, "to_datetime"):
                return start_time.to_datetime()
            return datetime.fromisoformat(str(start_time))
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not parse session start time '{start_time}' of type {type(start_time)}: {e}")
        return None

    def get_device_info(self) -> dict:
        """Get device-specific information.

        Returns
        -------
        dict
            Device information including microscope details.
        """
        return self._get_inscopix_metadata().get("device", {})

    def get_imaging_info(self) -> dict:
        """Get imaging parameters.

        Returns
        -------
        dict
            Imaging parameters including field of view, exposure, etc.
        """
        return self._get_inscopix_metadata().get("imaging", {})

    def get_subject_info(self) -> dict:
        """Get subject/animal information.

        Returns
        -------
        dict
            Subject information including animal ID, species, etc.
        """
        return self._get_inscopix_metadata().get("subject", {})

    def get_analysis_info(self) -> dict:
        """Get analysis method information.

        Returns
        -------
        dict
            Analysis information including cell identification method.
        """
        return self._get_inscopix_metadata().get("analysis", {})

    def get_session_info(self) -> dict:
        """Get session information.

        Returns
        -------
        dict
            Session information including timing, experimenter, etc.
        """
        return self._get_inscopix_metadata().get("session", {})

    def get_probe_info(self) -> dict:
        """Get probe information.

        Returns
        -------
        dict
            Probe information including diameter, length, etc.
        """
        return self._get_inscopix_metadata().get("probe", {})
