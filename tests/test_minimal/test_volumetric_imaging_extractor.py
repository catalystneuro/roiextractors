from ..mixins.volumetric_imaging_extractor_mixin import VolumetricImagingExtractorMixin
from roiextractors import NumpyImagingExtractor, VolumetricImagingExtractor
from roiextractors.tools.testing import generate_mock_video
import pytest
import numpy as np


class TestVolumetricImagingExtractor(VolumetricImagingExtractorMixin):
    @pytest.fixture(scope="class")
    def expected_video(self):
        return generate_mock_video(size=(5, 2, 4, 3))

    @pytest.fixture(scope="class")
    def expected_sampling_frequency(self):
        return 20.0

    @pytest.fixture(scope="function")
    def imaging_extractor(self, expected_video, expected_sampling_frequency):
        imaging_extractors = []
        for i in range(3):
            timeseries = expected_video[..., i]
            imaging_extractor = NumpyImagingExtractor(
                timeseries=timeseries, sampling_frequency=expected_sampling_frequency
            )
            imaging_extractors.append(imaging_extractor)
        return VolumetricImagingExtractor(imaging_extractors=imaging_extractors)

    @pytest.fixture(scope="function")
    def imaging_extractor2(self, expected_video, expected_sampling_frequency):
        imaging_extractors = []
        for i in range(3):
            timeseries = expected_video[..., i]
            imaging_extractor = NumpyImagingExtractor(
                timeseries=timeseries, sampling_frequency=expected_sampling_frequency
            )
            imaging_extractors.append(imaging_extractor)
        return VolumetricImagingExtractor(imaging_extractors=imaging_extractors)
