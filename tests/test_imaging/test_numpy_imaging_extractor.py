from ..mixins.imagingextractor_mixin import ImagingExtractorMixin
from roiextractors import NumpyImagingExtractor
from roiextractors.imagingextractor import FrameSliceImagingExtractor
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


class TestNumpyImagingExtractorFrameSlice(ImagingExtractorMixin):
    imaging_extractor_cls = FrameSliceImagingExtractor

    @pytest.fixture(scope="class")
    def imaging_extractor_kwargs(self):
        parent_kwargs = dict(timeseries=generate_dummy_video(size=(10, 2, 4)), sampling_frequency=20.0)
        parent_imaging = NumpyImagingExtractor(**parent_kwargs)
        return dict(parent_imaging=parent_imaging, start_frame=1, end_frame=4)

    @pytest.fixture(scope="class")
    def expected_video(self, imaging_extractor_kwargs):
        return imaging_extractor_kwargs["parent_imaging"].get_video(start_frame=1, end_frame=4)

    @pytest.fixture(scope="class")
    def expected_sampling_frequency(self, imaging_extractor_kwargs):
        return imaging_extractor_kwargs["parent_imaging"].get_sampling_frequency()

    @pytest.mark.parametrize("sampling_frequency", [1, 2, 3])
    def test_frame_to_time_no_times(self, imaging_extractor_kwargs, sampling_frequency):
        imaging_extractor = self.imaging_extractor_cls(**imaging_extractor_kwargs)
        imaging_extractor._parent_imaging._times = None
        imaging_extractor._parent_imaging._sampling_frequency = sampling_frequency
        times = imaging_extractor.frame_to_time(frames=[0, 1])
        expected_times = np.array([0, 1]) / sampling_frequency
        assert np.array_equal(times, expected_times)

    def test_frame_to_time_with_times(self, imaging_extractor_kwargs):
        imaging_extractor = self.imaging_extractor_cls(**imaging_extractor_kwargs)
        imaging_extractor._parent_imaging._times = np.array([0, 1])
        imaging_extractor._times = np.array([0, 1])
        times = imaging_extractor.frame_to_time(frames=[0, 1])
        assert np.array_equal(times, imaging_extractor._parent_imaging._times)

    @pytest.mark.parametrize("sampling_frequency", [1, 2, 3])
    def test_time_to_frame_no_times(self, imaging_extractor_kwargs, sampling_frequency):
        imaging_extractor = self.imaging_extractor_cls(**imaging_extractor_kwargs)
        imaging_extractor._parent_imaging._times = None
        imaging_extractor._parent_imaging._sampling_frequency = sampling_frequency
        imaging_extractor._times = None
        imaging_extractor._sampling_frequency = sampling_frequency
        times = np.array([0, 1]) / sampling_frequency
        frames = imaging_extractor.time_to_frame(times=times)
        expected_frames = np.array([0, 1])
        assert np.array_equal(frames, expected_frames)

    def test_time_to_frame_with_times(self, imaging_extractor_kwargs):
        imaging_extractor = self.imaging_extractor_cls(**imaging_extractor_kwargs)
        imaging_extractor._parent_imaging._times = np.array([0, 1])
        imaging_extractor._times = np.array([0, 1])
        times = np.array([0, 1])
        frames = imaging_extractor.time_to_frame(times=times)
        expected_frames = np.array([0, 1])
        assert np.array_equal(frames, expected_frames)

    def test_set_times(self, imaging_extractor_kwargs):
        imaging_extractor = self.imaging_extractor_cls(**imaging_extractor_kwargs)
        times = np.arange(imaging_extractor.get_num_frames())
        imaging_extractor.set_times(times)
        assert np.array_equal(imaging_extractor._times, times)

    def test_set_times_invalid_length(self, imaging_extractor):
        with pytest.raises(AssertionError):
            imaging_extractor.set_times(np.arange(imaging_extractor.get_num_frames() + 1))

    @pytest.mark.parametrize("times", [None, np.array([0, 1])])
    def test_has_time_vector(self, times, imaging_extractor_kwargs):
        imaging_extractor = self.imaging_extractor_cls(**imaging_extractor_kwargs)
        imaging_extractor._parent_imaging._times = times
        imaging_extractor._times = times
        if times is None:
            assert not imaging_extractor.has_time_vector()
        else:
            assert imaging_extractor.has_time_vector()

    def test_copy_times(self, imaging_extractor_kwargs):
        imaging_extractor = self.imaging_extractor_cls(**imaging_extractor_kwargs)
        imaging_extractor2 = self.imaging_extractor_cls(**imaging_extractor_kwargs)
        imaging_extractor._times = np.arange(imaging_extractor.get_num_frames())
        imaging_extractor2.copy_times(imaging_extractor)
        assert np.array_equal(imaging_extractor._times, imaging_extractor2._times)
        assert imaging_extractor._times is not imaging_extractor2._times
