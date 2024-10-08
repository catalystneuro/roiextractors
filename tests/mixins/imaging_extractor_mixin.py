import pytest
import numpy as np
from .base_extractor_mixin import BaseExtractorMixin


class ImagingExtractorMixin(BaseExtractorMixin):
    @pytest.fixture(scope="function")
    def extractor(self, imaging_extractor):
        return imaging_extractor

    @pytest.fixture(scope="function")
    def extractor2(self, imaging_extractor2):
        return imaging_extractor2

    @pytest.fixture(scope="function")
    def expected_image_size(self, expected_video):
        return expected_video.shape[1], expected_video.shape[2]

    @pytest.fixture(scope="function")
    def expected_num_frames(self, expected_video):
        return expected_video.shape[0]

    def test_get_dtype(self, imaging_extractor, expected_video):
        dtype = imaging_extractor.get_dtype()
        assert dtype == expected_video.dtype

    def test_get_video(self, imaging_extractor, expected_video):
        video = imaging_extractor.get_video()
        assert np.array_equal(video, expected_video)

    def test_get_video_slice(self, imaging_extractor, expected_video):
        video = imaging_extractor.get_video(start_frame=0, end_frame=1)
        assert np.array_equal(video, expected_video[:1])

    def test_get_video_invalid_start_frame(self, imaging_extractor):
        with pytest.raises(AssertionError):
            imaging_extractor.get_video(start_frame=-1)
        with pytest.raises(AssertionError):
            imaging_extractor.get_video(start_frame=imaging_extractor.get_num_frames() + 1)
        with pytest.raises(AssertionError):
            imaging_extractor.get_video(start_frame=0.5)

    def test_get_video_invalid_end_frame(self, imaging_extractor):
        with pytest.raises(AssertionError):
            imaging_extractor.get_video(end_frame=-1)
        with pytest.raises(AssertionError):
            imaging_extractor.get_video(end_frame=imaging_extractor.get_num_frames() + 1)
        with pytest.raises(AssertionError):
            imaging_extractor.get_video(end_frame=0.5)

    @pytest.mark.parametrize("frame_idxs", [[0], [0, 1], [0, 2], [0, 1, 2], [2, 1, 0]])
    def test_get_frames(self, imaging_extractor, expected_video, frame_idxs):
        """Test get_frames method.

        This method requires that the imaging_extractor used for testing has at least 3 frames.
        """
        frames = imaging_extractor.get_frames(frame_idxs=frame_idxs)
        assert np.array_equal(frames, expected_video[frame_idxs])

    def test_get_frames_invalid_frame_idxs(self, imaging_extractor):
        with pytest.raises(AssertionError):
            imaging_extractor.get_frames(frame_idxs=[-1])
        with pytest.raises(AssertionError):
            imaging_extractor.get_frames(frame_idxs=[imaging_extractor.get_num_frames()])
        with pytest.raises(AssertionError):
            imaging_extractor.get_frames(frame_idxs=[0.5])

    @pytest.mark.parametrize("start_frame, end_frame", [(None, None), (1, 3), (0, 1)])
    def test_frame_slice(self, imaging_extractor, start_frame, end_frame):
        frame_slice_imaging_extractor = imaging_extractor.frame_slice(start_frame=start_frame, end_frame=end_frame)
        start_frame = 0 if start_frame is None else start_frame
        end_frame = imaging_extractor.get_num_frames() if end_frame is None else end_frame
        assert frame_slice_imaging_extractor._parent_imaging is imaging_extractor
        assert frame_slice_imaging_extractor._start_frame == start_frame
        assert frame_slice_imaging_extractor._end_frame == end_frame
        assert frame_slice_imaging_extractor._num_frames == end_frame - start_frame

    def test_frame_slice_invalid_start_frame(self, imaging_extractor):
        with pytest.raises(AssertionError):
            imaging_extractor.frame_slice(start_frame=-1)
        with pytest.raises(AssertionError):
            imaging_extractor.frame_slice(start_frame=imaging_extractor.get_num_frames() + 1)
        with pytest.raises(AssertionError):
            imaging_extractor.frame_slice(start_frame=0.5)
        with pytest.raises(AssertionError):
            imaging_extractor.frame_slice(start_frame=3, end_frame=1)

    def test_frame_slice_invalid_end_frame(self, imaging_extractor):
        with pytest.raises(AssertionError):
            imaging_extractor.frame_slice(end_frame=-1)
        with pytest.raises(AssertionError):
            imaging_extractor.frame_slice(end_frame=imaging_extractor.get_num_frames() + 1)
        with pytest.raises(AssertionError):
            imaging_extractor.frame_slice(end_frame=0.5)


class FrameSliceImagingExtractorMixin:
    @pytest.fixture(scope="function")
    def frame_slice_imaging_extractor(self, imaging_extractor):
        return imaging_extractor.frame_slice(start_frame=1, end_frame=3)

    @pytest.fixture(scope="function")
    def frame_slice_imaging_extractor2(self, imaging_extractor2):
        return imaging_extractor2.frame_slice(start_frame=1, end_frame=3)

    @pytest.fixture(scope="function")
    def frame_slice_expected_video(self, expected_video):
        return expected_video[1:3]

    def test_get_image_size_frame_slice(self, frame_slice_imaging_extractor, frame_slice_expected_video):
        image_size = frame_slice_imaging_extractor.get_image_size()
        assert image_size == (frame_slice_expected_video.shape[1], frame_slice_expected_video.shape[2])

    def test_get_num_frames_frame_slice(self, frame_slice_imaging_extractor, frame_slice_expected_video):
        num_frames = frame_slice_imaging_extractor.get_num_frames()
        assert num_frames == frame_slice_expected_video.shape[0]

    def test_get_sampling_frequency_frame_slice(self, frame_slice_imaging_extractor, expected_sampling_frequency):
        sampling_frequency = frame_slice_imaging_extractor.get_sampling_frequency()
        assert sampling_frequency == expected_sampling_frequency

    def test_get_dtype_frame_slice(self, frame_slice_imaging_extractor, frame_slice_expected_video):
        dtype = frame_slice_imaging_extractor.get_dtype()
        assert dtype == frame_slice_expected_video.dtype

    def test_get_video_frame_slice(self, frame_slice_imaging_extractor, frame_slice_expected_video):
        video = frame_slice_imaging_extractor.get_video()
        assert np.array_equal(video, frame_slice_expected_video)

    def test_get_video_slice_frame_slice(self, frame_slice_imaging_extractor, frame_slice_expected_video):
        video = frame_slice_imaging_extractor.get_video(start_frame=0, end_frame=1)
        assert np.array_equal(video, frame_slice_expected_video[:1])

    def test_get_video_invalid_start_frame_frame_slice(self, frame_slice_imaging_extractor):
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.get_video(start_frame=-1)
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.get_video(start_frame=frame_slice_imaging_extractor.get_num_frames() + 1)
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.get_video(start_frame=0.5)

    def test_get_video_invalid_end_frame_frame_slice(self, frame_slice_imaging_extractor):
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.get_video(end_frame=-1)
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.get_video(end_frame=frame_slice_imaging_extractor.get_num_frames() + 1)
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.get_video(end_frame=0.5)

    @pytest.mark.parametrize("frame_idxs", [[0], [1], [0, 1], [1, 0]])
    def test_get_frames_frame_slice(self, frame_slice_imaging_extractor, frame_slice_expected_video, frame_idxs):
        frames = frame_slice_imaging_extractor.get_frames(frame_idxs=frame_idxs)
        assert np.array_equal(frames, frame_slice_expected_video[frame_idxs])

    def test_get_frames_invalid_frame_idxs_frame_slice(self, frame_slice_imaging_extractor):
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.get_frames(frame_idxs=[-1])
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.get_frames(frame_idxs=[frame_slice_imaging_extractor.get_num_frames()])
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.get_frames(frame_idxs=[0.5])

    @pytest.mark.parametrize("sampling_frequency", [1, 2, 3])
    def test_frame_to_time_no_times_frame_slice(self, frame_slice_imaging_extractor, sampling_frequency):
        frame_slice_imaging_extractor._times = None
        frame_slice_imaging_extractor._parent_imaging._sampling_frequency = sampling_frequency
        times = frame_slice_imaging_extractor.frame_to_time(frames=[0, 1])
        expected_times = np.array([0, 1]) / sampling_frequency
        assert np.array_equal(times, expected_times)

    def test_frame_to_time_with_times_frame_slice(self, frame_slice_imaging_extractor):
        expected_times = np.array([0, 1])
        frame_slice_imaging_extractor._times = expected_times
        times = frame_slice_imaging_extractor.frame_to_time(frames=[0, 1])

        assert np.array_equal(times, expected_times)

    @pytest.mark.parametrize("sampling_frequency", [1, 2, 3])
    def test_time_to_frame_no_times_frame_slice(self, frame_slice_imaging_extractor, sampling_frequency):
        frame_slice_imaging_extractor._times = None
        frame_slice_imaging_extractor._parent_imaging._sampling_frequency = sampling_frequency
        times = np.array([0, 1]) / sampling_frequency
        frames = frame_slice_imaging_extractor.time_to_frame(times=times)
        expected_frames = np.array([0, 1])
        assert np.array_equal(frames, expected_frames)

    def test_time_to_frame_with_times_frame_slice(self, frame_slice_imaging_extractor):
        frame_slice_imaging_extractor._times = np.array([0, 1])
        times = np.array([0, 1])
        frames = frame_slice_imaging_extractor.time_to_frame(times=times)
        expected_frames = np.array([0, 1])
        assert np.array_equal(frames, expected_frames)

    def test_set_times_frame_slice(self, frame_slice_imaging_extractor):
        times = np.arange(frame_slice_imaging_extractor.get_num_frames())
        frame_slice_imaging_extractor.set_times(times)
        assert np.array_equal(frame_slice_imaging_extractor._times, times)

    def test_set_times_invalid_length_frame_slice(self, frame_slice_imaging_extractor):
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.set_times(np.arange(frame_slice_imaging_extractor.get_num_frames() + 1))

    @pytest.mark.parametrize("times", [None, np.array([0, 1])])
    def test_has_time_vector_frame_slice(self, times, frame_slice_imaging_extractor):
        frame_slice_imaging_extractor._times = times
        if times is None:
            assert not frame_slice_imaging_extractor.has_time_vector()
        else:
            assert frame_slice_imaging_extractor.has_time_vector()

    def test_copy_times_frame_slice(self, frame_slice_imaging_extractor, frame_slice_imaging_extractor2):
        expected_times = np.arange(frame_slice_imaging_extractor.get_num_frames())
        frame_slice_imaging_extractor._times = expected_times
        frame_slice_imaging_extractor2.copy_times(frame_slice_imaging_extractor)
        assert np.array_equal(frame_slice_imaging_extractor2._times, expected_times)
        assert frame_slice_imaging_extractor2._times is not expected_times

    @pytest.mark.parametrize("start_frame, end_frame", [(None, None), (1, 2), (0, 1)])
    def test_frame_slice_on_frame_slice(self, frame_slice_imaging_extractor, start_frame, end_frame):
        twice_sliced_imaging_extractor = frame_slice_imaging_extractor.frame_slice(
            start_frame=start_frame, end_frame=end_frame
        )
        start_frame = 0 if start_frame is None else start_frame
        end_frame = twice_sliced_imaging_extractor.get_num_frames() if end_frame is None else end_frame
        assert twice_sliced_imaging_extractor._parent_imaging is frame_slice_imaging_extractor
        assert twice_sliced_imaging_extractor._start_frame == start_frame
        assert twice_sliced_imaging_extractor._end_frame == end_frame
        assert twice_sliced_imaging_extractor._num_frames == end_frame - start_frame

    def test_frame_slice_invalid_start_frame_on_frame_slice(self, frame_slice_imaging_extractor):
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.frame_slice(start_frame=-1)
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.frame_slice(start_frame=frame_slice_imaging_extractor.get_num_frames() + 1)
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.frame_slice(start_frame=0.5)
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.frame_slice(start_frame=3, end_frame=1)

    def test_frame_slice_invalid_end_frame_on_frame_slice(self, frame_slice_imaging_extractor):
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.frame_slice(end_frame=-1)
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.frame_slice(end_frame=frame_slice_imaging_extractor.get_num_frames() + 1)
        with pytest.raises(AssertionError):
            frame_slice_imaging_extractor.frame_slice(end_frame=0.5)

    def test_depth_slice_not_implemented(self, frame_slice_imaging_extractor):
        with pytest.raises(NotImplementedError):
            frame_slice_imaging_extractor.depth_slice()
