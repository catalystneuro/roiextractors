from typing import Type
from roiextractors.extractors import ImagingExtractor
import pytest
import numpy as np
from roiextractors.extraction_tools import ArrayType


class ImagingExtractorMixin:

    imaging_extractor_cls: Type[ImagingExtractor]
    imaging_extractor_kwargs: dict
    expected_video: ArrayType
    expected_sampling_frequency: float

    @pytest.fixture(scope="class")
    def imaging_extractor(self):
        return self.imaging_extractor_cls(**self.imaging_extractor_kwargs)

    def test_get_image_size(self, imaging_extractor):
        image_size = imaging_extractor.get_image_size()
        assert image_size == (self.expected_video.shape[1], self.expected_video.shape[2])

    def test_get_num_frames(self, imaging_extractor):
        num_frames = imaging_extractor.get_num_frames()
        assert num_frames == self.expected_video.shape[0]

    def test_get_sampling_frequency(self, imaging_extractor):
        sampling_frequency = imaging_extractor.get_sampling_frequency()
        assert sampling_frequency == self.expected_sampling_frequency

    def test_get_dtype(self, imaging_extractor):
        dtype = imaging_extractor.get_dtype()
        assert dtype == self.expected_video.dtype

    def test_get_video(self, imaging_extractor):
        video = imaging_extractor.get_video()
        assert np.array_equal(video, self.expected_video)

    def test_get_video_slice(self, imaging_extractor):
        video = imaging_extractor.get_video(start_frame=0, end_frame=1)
        assert np.array_equal(video, self.expected_video[:1])

    def test_get_video_invalid_start_frame(self, imaging_extractor):
        with pytest.raises(ValueError):
            imaging_extractor.get_video(start_frame=-1)
        with pytest.raises(ValueError):
            imaging_extractor.get_video(start_frame=imaging_extractor.get_num_frames() + 1)
        with pytest.raises(ValueError):
            imaging_extractor.get_video(start_frame=0.5)

    def test_get_video_invalid_end_frame(self, imaging_extractor):
        with pytest.raises(ValueError):
            imaging_extractor.get_video(end_frame=-1)
        with pytest.raises(ValueError):
            imaging_extractor.get_video(end_frame=imaging_extractor.get_num_frames() + 1)
        with pytest.raises(ValueError):
            imaging_extractor.get_video(end_frame=0.5)

    @pytest.mark.parametrize("frame_idxs", [[0], [0, 1], [0, 2], [0, 1, 2], [2, 1, 0]])
    def test_get_frames(self, imaging_extractor, frame_idxs):
        """Test get_frames method.

        This method requires that the imaging_extractor used for testing has at least 3 frames.
        """
        frames = imaging_extractor.get_frames(frame_idxs=frame_idxs)
        assert np.array_equal(frames, self.expected_video[frame_idxs])

    def test_get_frames_invalid_frame_idxs(self, imaging_extractor):
        with pytest.raises(ValueError):
            imaging_extractor.get_frames(frame_idxs=[-1])
        with pytest.raises(ValueError):
            imaging_extractor.get_frames(frame_idxs=[imaging_extractor.get_num_frames()])
        with pytest.raises(ValueError):
            imaging_extractor.get_frames(frame_idxs=[0.5])

    @pytest.mark.parametrize("sampling_frequency", [1, 2, 3])
    def test_frame_to_time_no_times(self, sampling_frequency):
        imaging_extractor = self.imaging_extractor_cls(**self.imaging_extractor_kwargs)
        imaging_extractor._times = None
        times = imaging_extractor.frame_to_time(frames=[0, 1])
        expected_times = np.array([0, 1]) / sampling_frequency
        assert np.array_equal(times, expected_times)

    def test_frame_to_time_with_times(self):
        imaging_extractor = self.imaging_extractor_cls(**self.imaging_extractor_kwargs)
        imaging_extractor._times = np.array([0, 1])
        times = imaging_extractor.frame_to_time(frames=[0, 1])
        assert np.array_equal(times, imaging_extractor._times)

    @pytest.mark.parametrize("sampling_frequency", [1, 2, 3])
    def test_time_to_frame_no_times(self, sampling_frequency):
        imaging_extractor = self.imaging_extractor_cls(**self.imaging_extractor_kwargs)
        imaging_extractor._times = None
        imaging_extractor._sampling_frequency = sampling_frequency
        times = np.array([0, 1]) / sampling_frequency
        frames = imaging_extractor.time_to_frame(times=times)
        expected_frames = np.array([0, 1])
        assert np.array_equal(frames, expected_frames)

    def test_time_to_frame_with_times(self):
        imaging_extractor = self.imaging_extractor_cls(**self.imaging_extractor_kwargs)
        imaging_extractor._times = np.array([0, 1])
        times = np.array([0, 1])
        frames = imaging_extractor.time_to_frame(times=times)
        expected_frames = np.array([0, 1])
        assert np.array_equal(frames, expected_frames)

    def test_set_times(self):
        imaging_extractor = self.imaging_extractor_cls(**self.imaging_extractor_kwargs)
        times = np.arange(imaging_extractor.get_num_frames())
        imaging_extractor.set_times(times)
        assert np.array_equal(imaging_extractor._times, times)

    def test_set_times_invalid_length(self, imaging_extractor):
        with pytest.raises(AssertionError):
            imaging_extractor.set_times(np.arange(imaging_extractor.get_num_frames() + 1))

    @pytest.mark.parametrize("times", [None, np.array([0, 1])])
    def test_has_time_vector(self, times):
        imaging_extractor = self.imaging_extractor_cls(**self.imaging_extractor_kwargs)
        imaging_extractor._times = times
        if times is None:
            assert not imaging_extractor.has_time_vector()
        else:
            assert imaging_extractor.has_time_vector()

    def test_copy_times(self):
        imaging_extractor = self.imaging_extractor_cls(**self.imaging_extractor_kwargs)
        imaging_extractor2 = self.imaging_extractor_cls(**self.imaging_extractor_kwargs)
        imaging_extractor._times = np.array([0, 1])
        imaging_extractor2.copy_times(imaging_extractor)
        assert np.array_equal(imaging_extractor._times, imaging_extractor2._times)
        assert imaging_extractor._times is not imaging_extractor2._times

    def test_eq(self, imaging_extractor):
        imaging_extractor2 = self.imaging_extractor_cls(**self.imaging_extractor_kwargs)
        assert imaging_extractor == imaging_extractor2

    def test_frame_slice(self, imaging_extractor):
        frame_slice = imaging_extractor.frame_slice(start_frame=0, end_frame=1)
        assert np.array_equal(frame_slice.get_video(), imaging_extractor.get_video(start_frame=0, end_frame=1))
