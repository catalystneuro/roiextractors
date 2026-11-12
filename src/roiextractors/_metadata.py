"""The metadata of an imaging extractor.

Classes
-------
Metadata
    The metadata of the recording that an imaging extractor reads, with NWB field names.
"""

from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from typing import Any


@dataclass
class Metadata:
    """The metadata of the recording that an imaging extractor reads.

    Field names follow the NWB ophys conventions where they apply, so a consumer like NeuroConv can pass them to NWB
    with little renaming. Every field is optional: an extractor fills the fields its format records and leaves the
    others as `None`. Format-specific values that have no field go in `extra_metadata`.

    A field with a unit has the unit at the end of its name, in a size that keeps the values readable: `_um` for
    micrometers, `_nm` for nanometers, `_mm` for millimeters, `_g` for grams, `_V` for volts, `_mW` for milliwatts and
    `_mW_per_mm2` for milliwatts per square millimeter. Times are in seconds and rates in Hz, as in the rest of
    roiextractors, and their names have no unit.

    A field is here only when a format records it. Values that only the user knows, such as the indicator, the brain
    region or the genotype of the subject, are left to the consumer, and a field for them is added when a format that
    records them is read.
    """

    # Acquisition timing
    acquisition_start_datetime: datetime | None = None  # the wall-clock time when the acquisition started

    # Spatial calibration, in the plane of the image: the height runs across the rows and the width across the columns
    pixel_height_um: float | None = None  # the distance from one row to the next
    pixel_width_um: float | None = None  # the distance from one column to the next
    field_of_view_height_um: float | None = None  # the height of the imaged area, its rows times the pixel height
    field_of_view_width_um: float | None = None  # the width of the imaged area, its columns times the pixel width
    # The position of each plane along z relative to the first plane, in the order of `select_plane()`. The first is 0
    # and the sign follows the z axis of the device that steps between planes. The absolute depth needs an anatomical
    # reference, such as the brain surface, which no microscope file records
    plane_z_offsets_um: list[float] | None = None
    origin_coords_um: list[float] | None = None
    reference_frame: str | None = None

    # Imaging plane (NWB ImagingPlane)
    imaging_plane_name: str | None = None
    imaging_plane_description: str | None = None
    excitation_wavelength_nm: float | None = None

    # Channel, the one the extractor reads
    channel_name: str | None = None
    channel_description: str | None = None
    emission_wavelength_nm: float | None = None

    # Acquisition settings
    excitation_mode: str | None = None  # "one-photon" or "two-photon"
    pmt_high_voltage_V: float | None = None  # the voltage across the PMT of the channel, which sets its gain
    scan_line_rate: float | None = None  # Hz
    exposure_time: float | None = None  # seconds
    excitation_power_mW: float | None = None
    binning: int | None = None
    frames_per_slice: int | None = None

    # Microscope
    microscope_name: str | None = None
    microscope_manufacturer: str | None = None
    microscope_model: str | None = None
    microscope_serial_number: str | None = None
    microscope_description: str | None = None

    # Acquisition software
    acquisition_software_name: str | None = None  # for example "Prairie View", "ScanImage", "ThorImageLS"
    acquisition_software_version: str | None = None

    # One-photon miniscopes
    microscope_focus_um: float | None = None
    camera_gain: float | None = None  # the gain of the image sensor
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
    subject_sex: str | None = None  # "M", "F", "U" or "O", as in NWB
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


def _get_plane_metadata(metadata: Metadata, plane_index: int) -> Metadata:
    """Return the metadata of one plane of a volume.

    The plane keeps its own z offset. Every other field is that of the volume.

    Parameters
    ----------
    metadata : Metadata
        The metadata of the volume.
    plane_index : int
        The index of the plane, as in `select_plane()`.

    Returns
    -------
    plane_metadata : Metadata
        The metadata of the plane.
    """
    plane_z_offsets_um = metadata.plane_z_offsets_um
    if plane_z_offsets_um is not None:
        plane_z_offsets_um = [plane_z_offsets_um[plane_index]]
    return replace(metadata, plane_z_offsets_um=plane_z_offsets_um)


def _get_cropped_metadata(
    metadata: Metadata, image_shape: tuple[int, int], cropped_image_shape: tuple[int, int]
) -> Metadata:
    """Return the metadata of a crop of the field of view.

    The width and the height of the field of view shrink with the columns and the rows of the crop. Every other field
    is that of the full field of view.

    Parameters
    ----------
    metadata : Metadata
        The metadata of the full field of view.
    image_shape : tuple[int, int]
        The (rows, columns) of the full field of view.
    cropped_image_shape : tuple[int, int]
        The (rows, columns) of the crop.

    Returns
    -------
    cropped_metadata : Metadata
        The metadata of the crop.
    """
    num_rows, num_columns = image_shape
    cropped_num_rows, cropped_num_columns = cropped_image_shape
    field_of_view_height_um = metadata.field_of_view_height_um
    if field_of_view_height_um is not None:
        field_of_view_height_um = field_of_view_height_um * cropped_num_rows / num_rows
    field_of_view_width_um = metadata.field_of_view_width_um
    if field_of_view_width_um is not None:
        field_of_view_width_um = field_of_view_width_um * cropped_num_columns / num_columns
    return replace(
        metadata, field_of_view_height_um=field_of_view_height_um, field_of_view_width_um=field_of_view_width_um
    )
