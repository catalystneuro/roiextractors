from ..mixins.imagingextractor_mixin import ImagingExtractorMixin
from roiextractors import NumpyImagingExtractor
from roiextractors.testing import generate_dummy_video
import pytest


class TestNumpyImagingExtractor(ImagingExtractorMixin):
    imaging_extractor_cls = NumpyImagingExtractor

    @pytest.fixture(scope="class")
    def imaging_extractor_kwargs(self):
        return dict(timeseries=generate_dummy_video(size=(3, 2, 4)), sampling_frequency=20.0)

    @pytest.fixture(scope="class")
    def expected_video(self, imaging_extractor_kwargs):
        return imaging_extractor_kwargs["timeseries"]

    @pytest.fixture(scope="class")
    def expected_sampling_frequency(self, imaging_extractor_kwargs):
        return imaging_extractor_kwargs["sampling_frequency"]
