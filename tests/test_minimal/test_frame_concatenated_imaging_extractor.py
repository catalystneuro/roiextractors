from ..mixins.frame_concatenated_imaging_extractor_mixin import (
    FrameConcatenatedImagingExtractorMixin,
    FrameConcatenatedFrameSliceImagingExtractorMixin,
)
from roiextractors import NumpyImagingExtractor, FrameConcatenatedImagingExtractor
from roiextractors.tools.testing import generate_mock_video
import pytest


class TestFrameConcatenatedImagingExtractor(
    FrameConcatenatedImagingExtractorMixin,
    FrameConcatenatedFrameSliceImagingExtractorMixin,
):
    @pytest.fixture(scope="class")
    def expected_video(self):
        return generate_mock_video(size=(9, 2, 4))

    @pytest.fixture(scope="class")
    def expected_sampling_frequency(self):
        return 20.0

    @pytest.fixture(scope="function")
    def imaging_extractor(self, expected_video, expected_sampling_frequency):
        imaging_extractors = []
        for i in range(3):
            timeseries = expected_video[i * 3 : (i + 1) * 3]
            imaging_extractor = NumpyImagingExtractor(
                timeseries=timeseries, sampling_frequency=expected_sampling_frequency
            )
            imaging_extractors.append(imaging_extractor)
        return FrameConcatenatedImagingExtractor(imaging_extractors=imaging_extractors)

    @pytest.fixture(scope="function")
    def imaging_extractor2(self, expected_video, expected_sampling_frequency):
        imaging_extractors = []
        for i in range(3):
            timeseries = expected_video[i * 3 : (i + 1) * 3]
            imaging_extractor = NumpyImagingExtractor(
                timeseries=timeseries, sampling_frequency=expected_sampling_frequency
            )
            imaging_extractors.append(imaging_extractor)
        return FrameConcatenatedImagingExtractor(imaging_extractors=imaging_extractors)
