"""A collection of ImagingExtractors for TIFF files with various formats.

Modules
-------
tiffimagingextractor
    A ImagingExtractor for TIFF files.
scanimagetiffimagingextractor
    Specialized extractor for reading TIFF files produced via ScanImage.
brukertiffimagingextractor
    Specialized extractor for reading TIFF files produced via Bruker.
micromanagertiffimagingextractor
    Specialized extractor for reading TIFF files produced via Micro-Manager.

Classes
-------
TiffImagingExtractor
    A ImagingExtractor for TIFF files.
ScanImageTiffImagingExtractor
    Legacy extractor for reading TIFF files produced via ScanImage v3.8.
ScanImageTiffSinglePlaneImagingExtractor
    Specialized extractor for reading single-plane TIFF files produced via ScanImage.
ScanImageTiffMultiPlaneImagingExtractor
    Specialized extractor for reading multi-plane TIFF files produced via ScanImage.
BrukerTiffMultiPlaneImagingExtractor
    Specialized extractor for reading TIFF files produced via Bruker.
BrukerTiffSinglePlaneImagingExtractor
    Specialized extractor for reading TIFF files produced via Bruker.
MicroManagerTiffImagingExtractor
    Specialized extractor for reading TIFF files produced via Micro-Manager.
"""

from .tiffimagingextractor import TiffImagingExtractor
from .scanimagetiffimagingextractor import (
    ScanImageTiffImagingExtractor,
    ScanImageTiffMultiPlaneImagingExtractor,
    ScanImageTiffSinglePlaneImagingExtractor,
)
from .brukertiffimagingextractor import BrukerTiffMultiPlaneImagingExtractor, BrukerTiffSinglePlaneImagingExtractor
from .micromanagertiffimagingextractor import MicroManagerTiffImagingExtractor
