"""Python-based module for extracting from, converting between, and handling recorded and optical imaging data from several file formats."""

from importlib.metadata import version

__version__ = version("roiextractors")

from .example_datasets import toy_example
from .extraction_tools import show_video
from .extractorlist import *
from .imagingextractor import ImagingExtractor
from .segmentationextractor import SegmentationExtractor
