from ..mixins.imagingextractor_mixin import ImagingExtractorMixin
from roiextractors import NumpyImagingExtractor
from roiextractors.testing import generate_dummy_video


class TestNumpyImagingExtractor(ImagingExtractorMixin):
    imaging_extractor_cls = NumpyImagingExtractor
    imaging_extractor_kwargs = dict(timeseries=generate_dummy_video(size=(3, 2, 4)), sampling_frequency=20.0)
    expected_video = imaging_extractor_kwargs["timeseries"]
    expected_sampling_frequency = 20.0
