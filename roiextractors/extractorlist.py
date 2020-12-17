from .extractors.caiman import CaimanSegmentationExtractor
from .extractors.hdf5imagingextractor import Hdf5ImagingExtractor
from .extractors.numpyextractors import NumpyImagingExtractor, NumpySegmentationExtractor
from .extractors.nwbextractors import NwbImagingExtractor, NwbSegmentationExtractor
from .extractors.schnitzerextractor import CnmfeSegmentationExtractor, ExtractSegmentationExtractor
from .extractors.simaextractor import SimaSegmentationExtractor
from .extractors.suite2p import Suite2pSegmentationExtractor
from .extractors.tiffimagingextractor import TiffImagingExtractor
from .multisegmentationextractor import MultiSegmentationExtractor

segmentation_extractor_full_list = [
    NumpySegmentationExtractor,
    NwbSegmentationExtractor,
    Suite2pSegmentationExtractor,
    CnmfeSegmentationExtractor,
    ExtractSegmentationExtractor,
    SimaSegmentationExtractor,
    MultiSegmentationExtractor,
    CaimanSegmentationExtractor
]

imaging_extractor_full_list = [
    NumpyImagingExtractor,
    Hdf5ImagingExtractor,
    TiffImagingExtractor,
    NwbImagingExtractor,
]

segmentation_extractor_dict = {segmentation_class.extractor_name: segmentation_class for segmentation_class in
                            segmentation_extractor_full_list}

imaging_extractor_dict = {imaging_class.extractor_name: imaging_class for imaging_class in
                            imaging_extractor_full_list}