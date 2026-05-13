"""Extractor for Thor TIFF files."""

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from .ometiffimagingextractor import OMETiffImagingExtractor
from ...extraction_tools import PathType


class ThorTiffImagingExtractor(OMETiffImagingExtractor):
    """Read imaging data exported by the ThorImage software.

    A ThorImage acquisition is stored as a folder of TIFF files (one or more) plus a
    sidecar ``Experiment.xml`` in the same folder. Both must be present. Pointing the
    extractor at any of the TIFF files is enough; related files in a multi-file
    dataset are discovered automatically.

    Note that data acquired with a Thor microscope but not exported through ThorImage
    will not have ``Experiment.xml`` and is not supported by this extractor.

    Parameters
    ----------
    file_path : str or Path
        Path to one of the TIFF files in the dataset folder. For multi-file datasets,
        any of the files works.
    channel_name : str or None, optional
        For multi-channel data, the name of the channel to read (e.g. ``"ChanA"``).
        Channel names follow ThorImage's convention as configured in the acquisition.
        Required when more than one channel is present and ignored otherwise.
    """

    extractor_name = "ThorTiffImaging"

    def __init__(self, file_path: PathType, channel_name: str | None = None):
        file_path = Path(file_path)
        folder_path = file_path.parent

        experiment_xml_path = folder_path / "Experiment.xml"
        if not experiment_xml_path.is_file():
            raise FileNotFoundError(f"Experiment.xml file not found in '{folder_path}'.")

        self._experiment_xml_root = ET.parse(experiment_xml_path).getroot()

        # Backwards-compat alias for downstream consumers that traverse the parsed XML
        # as a nested dict (notably neuroconv.ThorImagingInterface). New code should use
        # `_experiment_xml_root` and ElementTree's `.find()/.get()` API. This alias and
        # the helper functions below should be removed after consumers migrate.
        self._experiment_xml_dict = _xml_element_to_dict(self._experiment_xml_root)

        lsm = self._experiment_xml_root.find("LSM")
        if lsm is None or lsm.get("frameRate") is None:
            raise ValueError("Could not find 'LSM' element with frameRate attribute in Experiment.xml.")
        sampling_frequency = float(lsm.get("frameRate"))

        # Build channel names from Wavelengths. ThorImage uses these as the user-facing
        # channel identifiers (e.g. "ChanA", "ChanB"); they're stored before super().__init__()
        # because the base class calls self._get_channel_names() during init for validation.
        self._thor_channel_names = [
            wavelength.get("name")
            for wavelength in self._experiment_xml_root.findall("Wavelengths/Wavelength")
            if wavelength.get("name") is not None
        ]

        super().__init__(
            file_path=file_path,
            sampling_frequency=sampling_frequency,
            channel_name=channel_name,
        )

        self._kwargs = {"file_path": str(file_path), "channel_name": channel_name}

    def _get_channel_names(self) -> list[str]:
        """Return Thor-specific channel names from Experiment.xml when available."""
        if self._thor_channel_names:
            return self._thor_channel_names
        return super()._get_channel_names()

    def _get_session_start_time(self) -> datetime:
        """Return the acquisition start time as a tz-aware UTC ``datetime``.

        Read from the ``Date/@uTime`` Unix timestamp in ``Experiment.xml``.
        """
        date_element = self._experiment_xml_root.find("Date")
        if date_element is None or date_element.get("uTime") is None:
            raise ValueError("Could not find 'Date' element with uTime attribute in Experiment.xml.")
        return datetime.fromtimestamp(int(date_element.get("uTime")), tz=timezone.utc)


def _xml_element_to_dict(elem: ET.Element) -> dict:
    """Convert an ElementTree element into a nested dictionary.

    Attributes are prefixed with ``@`` and repeated child tags become lists. The
    representation is intentionally simple, not a general XML-to-dict converter
    (no namespace handling, no tail text, mixed-content nodes are flattened).

    .. note::
        This helper exists only to populate ``self._experiment_xml_dict``, which is
        kept as a backwards-compatibility for neuroconv.

        We should remove this once we address:
        https://github.com/catalystneuro/roiextractors/issues/578
    """
    dictionary = {elem.tag: {} if elem.attrib else None}
    children = list(elem)
    if children:
        nested_dictionary: dict = {}
        for child in children:
            child_dict = _xml_element_to_dict(child)
            tag = child.tag
            if tag in nested_dictionary:
                if isinstance(nested_dictionary[tag], list):
                    nested_dictionary[tag].append(child_dict[tag])
                else:
                    nested_dictionary[tag] = [nested_dictionary[tag], child_dict[tag]]
            else:
                nested_dictionary.update(child_dict)
        dictionary = {elem.tag: nested_dictionary}
    if elem.attrib:
        dictionary[elem.tag].update({f"@{k}": v for k, v in elem.attrib.items()})
    text = elem.text.strip() if elem.text else ""
    if text:
        if children or elem.attrib:
            dictionary[elem.tag]["#text"] = text
        else:
            dictionary[elem.tag] = text
    return dictionary
