from ..mixins.imagingextractor_mixin import ImagingExtractorMixin, FrameSliceImagingExtractorMixin
from roiextractors import NumpyImagingExtractor
from roiextractors.testing import generate_dummy_video
import pytest
import numpy as np


class TestNumpyImagingExtractor(ImagingExtractorMixin, FrameSliceImagingExtractorMixin):
    @pytest.fixture(scope="class")
    def expected_video(self):
        return generate_dummy_video(size=(3, 2, 4))

    @pytest.fixture(scope="class")
    def expected_sampling_frequency(self):
        return 20.0

    @pytest.fixture(scope="function")
    def imaging_extractor(self, expected_video, expected_sampling_frequency):
        return NumpyImagingExtractor(timeseries=expected_video, sampling_frequency=expected_sampling_frequency)

    @pytest.fixture(scope="function")
    def imaging_extractor2(self, expected_video, expected_sampling_frequency):
        return NumpyImagingExtractor(timeseries=expected_video, sampling_frequency=expected_sampling_frequency)


class TestNumpyImagingExtractorFromFile(ImagingExtractorMixin, FrameSliceImagingExtractorMixin):
    @pytest.fixture(scope="class")
    def expected_video(self):
        return generate_dummy_video(size=(3, 2, 4))

    @pytest.fixture(scope="class")
    def expected_sampling_frequency(self):
        return 20.0

    @pytest.fixture(scope="function")
    def imaging_extractor(self, expected_video, expected_sampling_frequency, tmp_path):
        file_path = tmp_path / "timeseries.npy"
        np.save(file_path, expected_video)
        return NumpyImagingExtractor(timeseries=str(file_path), sampling_frequency=expected_sampling_frequency)

    @pytest.fixture(scope="function")
    def imaging_extractor2(self, expected_video, expected_sampling_frequency, tmp_path):
        file_path = tmp_path / "timeseries2.npy"
        np.save(file_path, expected_video)
        return NumpyImagingExtractor(timeseries=str(file_path), sampling_frequency=expected_sampling_frequency)
