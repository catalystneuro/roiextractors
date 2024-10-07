from ..mixins.imaging_extractor_mixin import ImagingExtractorMixin, FrameSliceImagingExtractorMixin
from roiextractors import NumpyImagingExtractor, FrameConcatenatedImagingExtractor
from roiextractors.tools.testing import generate_mock_video
import pytest
import numpy as np


class TestFrameConcatenatedImagingExtractor(ImagingExtractorMixin, FrameSliceImagingExtractorMixin):
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

    @pytest.mark.parametrize("sampling_frequency", [1, 2, 3])
    def test_frame_to_time_no_times_frame_slice(self, frame_slice_imaging_extractor, sampling_frequency):
        frame_slice_imaging_extractor._times = None
        frame_slice_imaging_extractor._parent_imaging._imaging_extractors[0]._sampling_frequency = sampling_frequency
        times = frame_slice_imaging_extractor.frame_to_time(frames=[0, 1])
        expected_times = np.array([0, 1]) / sampling_frequency
        assert np.array_equal(times, expected_times)

    @pytest.mark.parametrize("sampling_frequency", [1, 2, 3])
    def test_time_to_frame_no_times_frame_slice(self, frame_slice_imaging_extractor, sampling_frequency):
        frame_slice_imaging_extractor._times = None
        frame_slice_imaging_extractor._parent_imaging._imaging_extractors[0]._sampling_frequency = sampling_frequency
        times = np.array([0, 1]) / sampling_frequency
        frames = frame_slice_imaging_extractor.time_to_frame(times=times)
        expected_frames = np.array([0, 1])
        assert np.array_equal(frames, expected_frames)
