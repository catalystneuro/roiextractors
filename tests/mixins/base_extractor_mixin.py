import pytest
import numpy as np


class BaseExtractorMixin:
    def test_get_image_size(self, extractor, expected_image_size):
        image_size = extractor.get_image_size()
        assert image_size == expected_image_size

    def test_get_num_frames(self, extractor, expected_num_frames):
        num_frames = extractor.get_num_frames()
        assert num_frames == expected_num_frames

    def test_get_sampling_frequency(self, extractor, expected_sampling_frequency):
        sampling_frequency = extractor.get_sampling_frequency()
        assert sampling_frequency == expected_sampling_frequency

    @pytest.mark.parametrize("sampling_frequency", [1, 2, 3])
    def test_frame_to_time_no_times(self, extractor, sampling_frequency):
        extractor._times = None
        extractor._sampling_frequency = sampling_frequency
        times = extractor.frame_to_time(frames=[0, 1])
        expected_times = np.array([0, 1]) / sampling_frequency
        assert np.array_equal(times, expected_times)

    def test_frame_to_time_with_times(self, extractor):
        expected_times = np.array([0, 1])
        extractor._times = expected_times
        times = extractor.frame_to_time(frames=[0, 1])

        assert np.array_equal(times, expected_times)

    @pytest.mark.parametrize("sampling_frequency", [1, 2, 3])
    def test_time_to_frame_no_times(self, extractor, sampling_frequency):
        extractor._times = None
        extractor._sampling_frequency = sampling_frequency
        times = np.array([0, 1]) / sampling_frequency
        frames = extractor.time_to_frame(times=times)
        expected_frames = np.array([0, 1])
        assert np.array_equal(frames, expected_frames)

    def test_time_to_frame_with_times(self, extractor):
        extractor._times = np.array([0, 1])
        times = np.array([0, 1])
        frames = extractor.time_to_frame(times=times)
        expected_frames = np.array([0, 1])
        assert np.array_equal(frames, expected_frames)

    def test_set_times(self, extractor):
        times = np.arange(extractor.get_num_frames())
        extractor.set_times(times)
        assert np.array_equal(extractor._times, times)

    def test_set_times_invalid_length(self, extractor):
        with pytest.raises(AssertionError):
            extractor.set_times(np.arange(extractor.get_num_frames() + 1))

    @pytest.mark.parametrize("times", [None, np.array([0, 1])])
    def test_has_time_vector(self, times, extractor):
        extractor._times = times
        if times is None:
            assert not extractor.has_time_vector()
        else:
            assert extractor.has_time_vector()

    def test_copy_times(self, extractor, extractor2):
        expected_times = np.arange(extractor.get_num_frames())
        extractor._times = expected_times
        extractor2.copy_times(extractor)
        assert np.array_equal(extractor2._times, expected_times)
        assert extractor2._times is not expected_times
