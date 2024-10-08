from ..mixins.imaging_extractor_mixin import ImagingExtractorMixin
import pytest
import numpy as np


class VolumetricImagingExtractorMixin(ImagingExtractorMixin):
    @pytest.mark.parametrize("sampling_frequency", [1, 2, 3])
    def test_frame_to_time_no_times(self, extractor, sampling_frequency):
        extractor._times = None
        extractor._imaging_extractors[0]._sampling_frequency = sampling_frequency
        times = extractor.frame_to_time(frames=[0, 1])
        expected_times = np.array([0, 1]) / sampling_frequency
        assert np.array_equal(times, expected_times)

    @pytest.mark.parametrize("sampling_frequency", [1, 2, 3])
    def test_time_to_frame_no_times(self, extractor, sampling_frequency):
        extractor._times = None
        extractor._imaging_extractors[0]._sampling_frequency = sampling_frequency
        times = np.array([0, 1]) / sampling_frequency
        frames = extractor.time_to_frame(times=times)
        expected_frames = np.array([0, 1])
        assert np.array_equal(frames, expected_frames)
