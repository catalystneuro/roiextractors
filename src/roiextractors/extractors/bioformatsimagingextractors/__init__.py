"""A collection of ImagingExtractors for reading files with Bio-Formats.

Modules
-------
bioformatsimagingextractor
    The base class for Bio-Formats imaging extractors.
cxdimagingextractor
    Specialized extractor for CXD files produced via Hamamatsu Photonics.

Classes
-------
BioFormatsImagingExtractor
    The base ImagingExtractor for Bio-Formats.
CxdImagingExtractor
    Specialized extractor for reading CXD files produced via Hamamatsu Photonics.
"""

from .cxdimagingextractor import CxdImagingExtractor
