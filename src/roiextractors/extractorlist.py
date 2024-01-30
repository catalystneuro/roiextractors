"""Listing of available formats for extraction."""

from .extractors.caiman import CaimanSegmentationExtractor
from .extractors.hdf5imagingextractor import Hdf5ImagingExtractor
from .extractors.numpyextractors import (
    NumpyImagingExtractor,
    NumpySegmentationExtractor,
)
from .extractors.nwbextractors import NwbImagingExtractor, NwbSegmentationExtractor
from .extractors.schnitzerextractor import (
    CnmfeSegmentationExtractor,
    ExtractSegmentationExtractor,
)
from .extractors.simaextractor import SimaSegmentationExtractor
from .extractors.suite2p import Suite2pSegmentationExtractor
from .extractors.tiffimagingextractors import (
    TiffImagingExtractor,
    ScanImageTiffImagingExtractor,
    ScanImageTiffSinglePlaneImagingExtractor,
    ScanImageTiffMultiPlaneImagingExtractor,
    BrukerTiffMultiPlaneImagingExtractor,
    BrukerTiffSinglePlaneImagingExtractor,
    MicroManagerTiffImagingExtractor,
)
from .extractors.sbximagingextractor import SbxImagingExtractor
from .extractors.memmapextractors import NumpyMemmapImagingExtractor
from .extractors.memmapextractors import MemmapImagingExtractor
from .extractors.miniscopeimagingextractor import MiniscopeImagingExtractor
from .multisegmentationextractor import MultiSegmentationExtractor
from .multiimagingextractor import MultiImagingExtractor
from .volumetricimagingextractor import VolumetricImagingExtractor

imaging_extractor_full_list = [
    NumpyImagingExtractor,
    Hdf5ImagingExtractor,
    TiffImagingExtractor,
    ScanImageTiffImagingExtractor,
    ScanImageTiffSinglePlaneImagingExtractor,
    ScanImageTiffMultiPlaneImagingExtractor,
    BrukerTiffMultiPlaneImagingExtractor,
    BrukerTiffSinglePlaneImagingExtractor,
    MicroManagerTiffImagingExtractor,
    MiniscopeImagingExtractor,
    NwbImagingExtractor,
    SbxImagingExtractor,
    NumpyMemmapImagingExtractor,
    MemmapImagingExtractor,
    VolumetricImagingExtractor,
]

segmentation_extractor_full_list = [
    NumpySegmentationExtractor,
    NwbSegmentationExtractor,
    Suite2pSegmentationExtractor,
    CnmfeSegmentationExtractor,
    ExtractSegmentationExtractor,
    SimaSegmentationExtractor,
    CaimanSegmentationExtractor,
]

imaging_extractor_dict = {imaging_class.extractor_name: imaging_class for imaging_class in imaging_extractor_full_list}
segmentation_extractor_dict = {
    segmentation_class.extractor_name: segmentation_class for segmentation_class in segmentation_extractor_full_list
}
