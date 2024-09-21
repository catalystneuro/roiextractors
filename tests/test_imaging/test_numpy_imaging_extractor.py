from ..mixins.imagingextractor_mixin import ImagingExtractorMixin
from roiextractors import NumpyImagingExtractor
from roiextractors.testing import generate_dummy_video
import pytest
import numpy as np


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


class TestNumpyImagingExtractorFromFile(ImagingExtractorMixin):
    imaging_extractor_cls = NumpyImagingExtractor

    @pytest.fixture(scope="class")
    def imaging_extractor_kwargs(self, tmp_path_factory):
        temp_dir = tmp_path_factory.mktemp("data")
        file_path = temp_dir / "timeseries.npy"
        timeseries = generate_dummy_video(size=(3, 2, 4))
        np.save(file_path, timeseries)
        return dict(timeseries=file_path, sampling_frequency=20.0)

    @pytest.fixture(scope="class")
    def expected_video(self, imaging_extractor_kwargs):
        return np.load(imaging_extractor_kwargs["timeseries"])

    @pytest.fixture(scope="class")
    def expected_sampling_frequency(self, imaging_extractor_kwargs):
        return imaging_extractor_kwargs["sampling_frequency"]
