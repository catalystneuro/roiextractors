"""Python-based module for extracting from, converting between, and handling recorded and optical imaging data from several file formats."""

# Keeping __version__ accessible only to maintain backcompatability.
# Modern appraoch (Python >= 3.8) is to use importlib
try:
    from importlib.metadata import version

    __version__ = version("roiextractors")
except ModuleNotFoundError:  # Remove the except clause when minimal supported version becomes 3.8
    from pkg_resources import get_distribution

    __version__ = get_distribution("roiextractors").version


from .example_datasets import toy_example
from .extraction_tools import show_video
from .extractorlist import *
from .imagingextractor import ImagingExtractor
from .segmentationextractor import SegmentationExtractor
