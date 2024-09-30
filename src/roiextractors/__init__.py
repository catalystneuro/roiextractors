"""Python-based module for extracting from, converting between, and handling recorded and optical imaging data from several file formats."""

from importlib.metadata import version

__version__ = version("roiextractors")

from .extractorlist import *
from .imagingextractor import ImagingExtractor
from .segmentationextractor import SegmentationExtractor
