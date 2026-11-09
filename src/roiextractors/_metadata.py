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

    Some fields are always `None` from an extractor because no format records them: `excitation_lambda`,
    `indicator`, `location`, `subject_genotype` and `subject_age`. The consumer supplies them.
    """

    # Acquisition timing
    acquisition_datetime: datetime | None = None  # NWB `session_start_time`

    # Spatial calibration
    grid_spacing: list[float] | None = None  # [x, y] or [x, y, z] in `grid_spacing_unit`
    grid_spacing_unit: str = "micrometers"
    field_of_view: list[float] | None = None  # [width, height] or [width, height, depth] in meters
    origin_coords: list[float] | None = None
    origin_coords_unit: str = "meters"
    reference_frame: str | None = None

    # Imaging plane (NWB ImagingPlane)
    imaging_plane_name: str | None = None
    imaging_plane_description: str | None = None
    excitation_lambda: float | None = None  # nm
    indicator: str | None = None  # for example "GCaMP6s"
    location: str | None = None  # brain region, for example "V1"

    # Channel, the one the extractor reads
    channel_name: str | None = None
    channel_description: str | None = None
    emission_lambda: float | None = None  # nm

    # Photon series (NWB OnePhotonSeries or TwoPhotonSeries)
    photon_series_type: str | None = None  # "OnePhotonSeries" or "TwoPhotonSeries"
    pmt_gain: float | None = None
    scan_line_rate: float | None = None  # Hz
    exposure_time: float | None = None  # seconds
    power: float | None = None  # mW
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
    microscope_focus: float | None = None  # micrometers
    microscope_gain: float | None = None  # LED or camera gain, distinct from `pmt_gain`
    led_power_excitation: float | None = None  # mW/mm^2
    led_power_orange_green: float | None = None  # mW/mm^2

    # Objective
    objective_magnification: float | None = None
    objective_numerical_aperture: float | None = None
    objective_immersion: str | None = None  # for example "Air", "Water", "Oil"
    objective_correction: str | None = None  # for example "Achromatic", "PlanFluor"
    objective_working_distance: float | None = None  # mm

    # Subject
    subject_id: str | None = None
    subject_species: str | None = None
    subject_strain: str | None = None
    subject_sex: str | None = None  # "M", "F", "U" or "O", as in NWB
    subject_age: str | None = None  # ISO 8601 duration, for example "P60D"
    subject_genotype: str | None = None
    subject_weight: float | None = None  # grams
    subject_date_of_birth: datetime | None = None
    subject_description: str | None = None

    # Session
    session_id: str | None = None
    session_description: str | None = None
    experimenter: list[str] | None = None

    # Format-specific values that have no field
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_populated_dict(self, *, include_extra: bool = True) -> dict[str, Any]:
        """Return the fields that are set, skipping those left at their default.

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
            # The units default to a string
            if metadata_field.default is not None and value == metadata_field.default:
                continue
            populated_metadata[metadata_field.name] = value
        if include_extra and self.extra_metadata:
            populated_metadata["extra_metadata"] = dict(self.extra_metadata)
        return populated_metadata
