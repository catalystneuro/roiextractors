from ..mixins.imaging_extractor_mixin import ImagingExtractorMixin, FrameSliceImagingExtractorMixin
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

    @pytest.mark.parametrize("start_plane, end_plane", [(None, None), (0, 1), (1, 2), (0, 2)])
    def test_depth_slice(self, imaging_extractor, start_plane, end_plane):
        start_plane = start_plane if start_plane is not None else 0
        end_plane = end_plane if end_plane is not None else imaging_extractor.get_num_planes()
        sliced_extractor = imaging_extractor.depth_slice(start_plane=start_plane, end_plane=end_plane)

        assert sliced_extractor.get_num_planes() == end_plane - start_plane
        assert sliced_extractor.get_image_size() == imaging_extractor.get_image_size()
        video = imaging_extractor.get_video()
        sliced_video = sliced_extractor.get_video()
        np.testing.assert_array_equal(video[..., start_plane:end_plane], sliced_video)
        frames = imaging_extractor.get_frames(frame_idxs=[0, 2])
        sliced_frames = sliced_extractor.get_frames(frame_idxs=[0, 2])
        np.testing.assert_array_equal(frames[..., start_plane:end_plane], sliced_frames)

    @pytest.mark.parametrize("start_plane, end_plane", [(0, -1), (1, 0), (0, 4)])
    def test_depth_slice_invalid(self, imaging_extractor, start_plane, end_plane):
        with pytest.raises(AssertionError):
            imaging_extractor.depth_slice(start_plane=start_plane, end_plane=end_plane)

    def test_depth_slice_twice(self, imaging_extractor):
        sliced_extractor = imaging_extractor.depth_slice(start_plane=0, end_plane=2)
        twice_sliced_extractor = sliced_extractor.depth_slice(start_plane=0, end_plane=1)

        assert twice_sliced_extractor.get_num_planes() == 1
        assert twice_sliced_extractor.get_image_size() == imaging_extractor.get_image_size()
        video = imaging_extractor.get_video()
        sliced_video = twice_sliced_extractor.get_video()
        np.testing.assert_array_equal(video[..., :1], sliced_video)
        frames = imaging_extractor.get_frames(frame_idxs=[0, 2])
        sliced_frames = twice_sliced_extractor.get_frames(frame_idxs=[0, 2])
        np.testing.assert_array_equal(frames[..., :1], sliced_frames)


class VolumetricFrameSliceImagingExtractorMixin(FrameSliceImagingExtractorMixin):
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
