"""Segmentation extractors for CNMF-E and EXTRACT ROI segmentation method.

Modules
-------
cnmfesegmentationextractor
    A segmentation extractor for CNMF-E ROI segmentation method.
extractsegmentationextractor
    A segmentation extractor for EXTRACT segmentation method.

Classes
-------
CnmfeSegmentationExtractor
    A segmentation extractor for CNMF-E ROI segmentation method.
ExtractSegmentationExtractor
    Abstract class that defines which EXTRACT class to use for a given file (new vs old).
NewExtractSegmentationExtractor
    Extractor for reading the segmentation data that results from calls to newer versions of EXTRACT.
LegacyExtractSegmentationExtractor
    Extractor for reading the segmentation data that results from calls to older versions of EXTRACT.
"""

from .cnmfesegmentationextractor import CnmfeSegmentationExtractor
from .extractsegmentationextractor import (
    LegacyExtractSegmentationExtractor,
    ExtractSegmentationExtractor,
    NewExtractSegmentationExtractor,
)  # TODO: remove legacy imports
