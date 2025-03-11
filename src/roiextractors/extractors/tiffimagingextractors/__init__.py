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
thortiffimagingextractor
    Specialized extractor for reading TIFF files produced via Thor.
multitiffmultipageextractor
    Extractor for multiple TIFF files, each with multiple pages.

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
ScanImageTiffSinglePlaneMultiFileImagingExtractor
    Specialized extractor for reading single-plane multi-file TIFF files produced via ScanImage.
ScanImageTiffMultiPlaneMultiFileImagingExtractor
    Specialized extractor for reading multi-plane multi-file TIFF files produced via ScanImage.
BrukerTiffMultiPlaneImagingExtractor
    Specialized extractor for reading TIFF files produced via Bruker.
BrukerTiffSinglePlaneImagingExtractor
    Specialized extractor for reading TIFF files produced via Bruker.
MicroManagerTiffImagingExtractor
    Specialized extractor for reading TIFF files produced via Micro-Manager.
ThorTiffImagingExtractor
    Specialized extractor for reading TIFF files produced via Thor.
MultiTIFFMultiPageExtractor
    An extractor for handling multiple TIFF files, each with multiple pages, organized according to a specified dimension order.
"""

from .tiffimagingextractor import TiffImagingExtractor
from .scanimagetiffimagingextractor import (
    ScanImageTiffImagingExtractor,
    ScanImageTiffMultiPlaneImagingExtractor,
    ScanImageTiffSinglePlaneImagingExtractor,
    ScanImageTiffSinglePlaneMultiFileImagingExtractor,
    ScanImageTiffMultiPlaneMultiFileImagingExtractor,
)
from .brukertiffimagingextractor import BrukerTiffMultiPlaneImagingExtractor, BrukerTiffSinglePlaneImagingExtractor
from .micromanagertiffimagingextractor import MicroManagerTiffImagingExtractor
from .thortiffimagingextractor import ThorTiffImagingExtractor
from .multitiffmultipageextractor import MultiTIFFMultiPageExtractor
