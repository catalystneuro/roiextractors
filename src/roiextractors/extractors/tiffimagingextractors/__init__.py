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
ScanImageLegacyImagingExtractor
    Legacy extractor for reading TIFF files produced via ScanImage v3.8.
ScanImageImagingExtractor
    Specialized extractor for reading TIFF files produced via ScanImage.
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

from .brukertiffimagingextractor import (
    BrukerTiffMultiPlaneImagingExtractor,
    BrukerTiffSinglePlaneImagingExtractor,
)
from .micromanagertiffimagingextractor import MicroManagerTiffImagingExtractor
from .scanimagetiffimagingextractor import (
    ScanImageImagingExtractor,
    ScanImageLegacyImagingExtractor,
)
from .thortiffimagingextractor import ThorTiffImagingExtractor
from .multitiffmultipageextractor import MultiTIFFMultiPageExtractor
from .tiffimagingextractor import TiffImagingExtractor
