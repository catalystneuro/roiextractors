"""Defines memmap-based ImagingExtractors. Currently, only numpy.memmap is supported.

Modules
-------
memmapextractors
    The base class for memmapable imaging extractors.
numpymemampextractor
    The class for reading optical imaging data stored in a binary format with numpy.memmap.

Classes
-------
MemmapImagingExtractor
    The base class for memmapable imaging extractors.
NumpyMemmapImagingExtractor
    The class for reading optical imaging data stored in a binary format with numpy.memmap.
"""

from .memmapextractors import MemmapImagingExtractor
from .numpymemampextractor import NumpyMemmapImagingExtractor
