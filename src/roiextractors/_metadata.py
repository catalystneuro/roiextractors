"""The metadata of an imaging extractor.

Classes
-------
Metadata
    The metadata of the recording that an imaging extractor reads, with NWB field names.
"""

from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any


@dataclass
class Metadata:
    """The metadata of the recording that an imaging extractor reads.

    Field names follow the NWB ophys conventions where they apply, so a consumer like NeuroConv can pass them to NWB
    with little renaming. Every field is optional: an extractor fills the fields its format records and leaves the
    others as `None`. Format-specific values that have no field go in `extra_metadata`.

    A field with a unit has the unit at the end of its name, in a size that keeps the values readable: `_um` for
    micrometers, `_nm` for nanometers, `_mm` for millimeters, `_g` for grams, `_mW` for milliwatts and `_mW_per_mm2`
    for milliwatts per square millimeter. Times are in seconds and rates in Hz, as in the rest of roiextractors, and
    their names have no unit.

    Some fields are always `None` from an extractor because no format records them: `excitation_lambda_nm`,
    `indicator`, `location`, `subject_genotype` and `subject_age`. The consumer supplies them.
    """

    # Acquisition timing
    acquisition_datetime: datetime | None = None  # NWB `session_start_time`

    # Spatial calibration
    grid_spacing_um: list[float] | None = None  # [x, y] or [x, y, z]
    field_of_view_um: list[float] | None = None  # [width, height] or [width, height, depth]
    origin_coords_um: list[float] | None = None
    reference_frame: str | None = None

    # Imaging plane (NWB ImagingPlane)
    imaging_plane_name: str | None = None
    imaging_plane_description: str | None = None
    excitation_lambda_nm: float | None = None
    indicator: str | None = None  # for example "GCaMP6s"
    location: str | None = None  # brain region, for example "V1"

    # Channel, the one the extractor reads
    channel_name: str | None = None
    channel_description: str | None = None
    emission_lambda_nm: float | None = None

    # Photon series (NWB OnePhotonSeries or TwoPhotonSeries)
    photon_series_type: str | None = None  # "OnePhotonSeries" or "TwoPhotonSeries"
    pmt_gain: float | None = None
    scan_line_rate: float | None = None  # Hz
    exposure_time: float | None = None  # seconds
    power_mW: float | None = None
    binning: int | None = None
    frames_per_slice: int | None = None

    # Device (microscope)
    device_name: str | None = None
    device_manufacturer: str | None = None
    device_model: str | None = None
    device_serial_number: str | None = None
    device_description: str | None = None
    device_software_name: str | None = None  # for example "Prairie View", "ScanImage", "ThorImageLS"
    device_software_version: str | None = None

    # One-photon miniscopes
    microscope_focus_um: float | None = None
    microscope_gain: float | None = None  # LED or camera gain, distinct from `pmt_gain`
    led_power_excitation_mW_per_mm2: float | None = None
    led_power_orange_green_mW_per_mm2: float | None = None

    # Objective
    objective_magnification: float | None = None
    objective_numerical_aperture: float | None = None
    objective_immersion: str | None = None  # for example "Air", "Water", "Oil"
    objective_correction: str | None = None  # for example "Achromatic", "PlanFluor"
    objective_working_distance_mm: float | None = None

    # Subject
    subject_id: str | None = None
    subject_species: str | None = None
    subject_strain: str | None = None
    subject_sex: str | None = None  # "M", "F", "U" or "O", as in NWB
    subject_age: str | None = None  # ISO 8601 duration, for example "P60D"
    subject_genotype: str | None = None
    subject_weight_g: float | None = None
    subject_date_of_birth: datetime | None = None
    subject_description: str | None = None

    # Session
    session_id: str | None = None
    session_description: str | None = None
    experimenter: list[str] | None = None

    # Format-specific values that have no field
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_populated_dict(self, *, include_extra: bool = True) -> dict[str, Any]:
        """Return the fields that are not `None`.

        Parameters
        ----------
        include_extra : bool, default True
            Whether to include `extra_metadata` when it is not empty.

        Returns
        -------
        populated_metadata : dict
            The fields that are set, by name.
        """
        populated_metadata = {}
        for metadata_field in fields(self):
            if metadata_field.name == "extra_metadata":
                continue
            value = getattr(self, metadata_field.name)
            if value is None:
                continue
            populated_metadata[metadata_field.name] = value
        if include_extra and self.extra_metadata:
            populated_metadata["extra_metadata"] = dict(self.extra_metadata)
        return populated_metadata
