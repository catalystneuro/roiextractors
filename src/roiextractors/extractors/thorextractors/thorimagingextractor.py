import glob
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
from PIL import Image
from parse import parse

from ...imagingextractor import ImagingExtractor
from ...extraction_tools import PathType


def match_paths(base, pattern, sort_by_values=True):
    full_pattern = os.path.join(base, pattern)
    paths = glob.glob(os.path.join(base, "*"))
    out = {}
    for path in paths:
        parsed = parse(full_pattern, path)
        if parsed is not None:
            out[path] = parsed.named

    if sort_by_values:
        out = dict(sorted(out.items(), key=lambda item: tuple(item[1].values())))

    return out


def extract_experiment_details(xml_file_path: str):
    """
    Extract the frameRate from the LSM element and the start time from the Date element.

    Parameters
    ----------
    xml_file_path : str
        Path to the XML file containing the experiment details.

    Returns
    -------
    dict
        A dictionary containing the frameRate and startTime if available.
    """
    # Dictionary to hold the extracted values
    details = {}

    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Extract frameRate from the LSM element
    lsm_element = root.find(".//LSM")
    if lsm_element is not None and "frameRate" in lsm_element.attrib:
        details["frameRate"] = float(lsm_element.attrib["frameRate"])

    # Extract startTime from the Date element
    date_element = root.find(".//Date")
    if date_element is not None and "date" in date_element.attrib:
        date_str = date_element.attrib["date"]
        details["startTime"] = datetime.strptime(date_str, "%m/%d/%Y %H:%M:%S")

    return details


class ThorTiffImagingExtractor(ImagingExtractor):
    """A ImagingExtractor for multiple TIFF files."""

    extractor_name = "ThorTiffImaging"
    is_writable = False

    def __init__(self, folder_path: PathType, pattern="{channel}_001_001_001_{frame:d}.tif"):
        """Create a ThorTiffImagingExtractor instance from a TIFF file.

        Parameters
        ----------
        folder_path : str
            Folder that contains the TIFF files and the Experiment.xml file.
        """

        super().__init__()
        self.folder_path = folder_path

        paths = match_paths(folder_path, pattern)

        channels = list(set(x["channel"] for x in paths.values()))

        self._video = {}
        for channel in channels:
            data = []
            for fpath in paths:
                if paths[fpath]["channel"] != channel:
                    continue
                img = Image.open(fpath)
                data.append(np.array(img))
            self._video[channel] = np.array(data)

        shape = self._video[channels[0]].shape
        self._num_frames, self._num_rows, self._num_columns = shape
        self._num_channels = len(channels)
        self._channel_names = channels

        extracted_metadata = extract_experiment_details(os.path.join(folder_path, "Experiment.xml"))
        self._sampling_frequency = extracted_metadata.get("frameRate", None)
        self.start_time = extracted_metadata.get("startTime", None)

        self._kwargs = {"folder_path": folder_path}

    def get_frames(self, frame_idxs, channel: int = 0):
        return self._video[channel][frame_idxs, ...]

    def get_video(self, start_frame=None, end_frame=None, channel: Optional[int] = 0) -> np.ndarray:
        return self._video[channel][start_frame:end_frame, ...]

    def get_image_size(self) -> Tuple[int, int]:
        return self._num_rows, self._num_columns

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_num_channels(self):
        return self._num_channels

    def get_channel_names(self):
        return self._channel_names
