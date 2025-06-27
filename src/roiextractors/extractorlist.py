"""Listing of available formats for extraction."""

from .extractors.caiman import CaimanSegmentationExtractor
from .extractors.femtonicsimagingextractor import FemtonicsImagingExtractor
from .extractors.hdf5imagingextractor import Hdf5ImagingExtractor
from .extractors.inscopixextractors import (
    InscopixImagingExtractor,
    InscopixSegmentationExtractor,
)
from .extractors.memmapextractors import (
    MemmapImagingExtractor,
    NumpyMemmapImagingExtractor,
)
from .extractors.minian import MinianSegmentationExtractor
from .extractors.miniscopeimagingextractor import MiniscopeImagingExtractor
from .extractors.numpyextractors import (
    NumpyImagingExtractor,
    NumpySegmentationExtractor,
)
from .extractors.nwbextractors import NwbImagingExtractor, NwbSegmentationExtractor
from .extractors.sbximagingextractor import SbxImagingExtractor
from .extractors.schnitzerextractor import (
    CnmfeSegmentationExtractor,
    ExtractSegmentationExtractor,
)
from .extractors.simaextractor import SimaSegmentationExtractor
from .extractors.suite2p import Suite2pSegmentationExtractor
from .extractors.tiffimagingextractors import (
    BrukerTiffMultiPlaneImagingExtractor,
    BrukerTiffSinglePlaneImagingExtractor,
    MicroManagerTiffImagingExtractor,
    ScanImageImagingExtractor,
    ScanImageLegacyImagingExtractor,
    ScanImageTiffMultiPlaneImagingExtractor,
    ScanImageTiffMultiPlaneMultiFileImagingExtractor,
    ScanImageTiffSinglePlaneImagingExtractor,
    ScanImageTiffSinglePlaneMultiFileImagingExtractor,
    ThorTiffImagingExtractor,
    TiffImagingExtractor,
)
from .volumetricimagingextractor import VolumetricImagingExtractor

imaging_extractor_full_list = [
    FemtonicsImagingExtractor,
    NumpyImagingExtractor,
    Hdf5ImagingExtractor,
    TiffImagingExtractor,
    ScanImageLegacyImagingExtractor,
    ScanImageTiffSinglePlaneImagingExtractor,
    ScanImageTiffMultiPlaneImagingExtractor,
    ScanImageTiffSinglePlaneMultiFileImagingExtractor,
    ScanImageTiffMultiPlaneMultiFileImagingExtractor,
    ScanImageImagingExtractor,
    BrukerTiffMultiPlaneImagingExtractor,
    BrukerTiffSinglePlaneImagingExtractor,
    MicroManagerTiffImagingExtractor,
    ThorTiffImagingExtractor,
    MiniscopeImagingExtractor,
    NwbImagingExtractor,
    SbxImagingExtractor,
    NumpyMemmapImagingExtractor,
    MemmapImagingExtractor,
    VolumetricImagingExtractor,
    InscopixImagingExtractor,
]

segmentation_extractor_full_list = [
    NumpySegmentationExtractor,
    NwbSegmentationExtractor,
    Suite2pSegmentationExtractor,
    CnmfeSegmentationExtractor,
    ExtractSegmentationExtractor,
    SimaSegmentationExtractor,
    CaimanSegmentationExtractor,
    InscopixSegmentationExtractor,
    MinianSegmentationExtractor,
]

imaging_extractor_dict = {imaging_class.extractor_name: imaging_class for imaging_class in imaging_extractor_full_list}
segmentation_extractor_dict = {
    segmentation_class.extractor_name: segmentation_class for segmentation_class in segmentation_extractor_full_list
}
