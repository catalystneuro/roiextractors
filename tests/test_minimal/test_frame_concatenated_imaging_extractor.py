from ..mixins.frame_concatenated_imaging_extractor_mixin import (
    FrameConcatenatedImagingExtractorMixin,
    FrameConcatenatedFrameSliceImagingExtractorMixin,
)
from roiextractors import NumpyImagingExtractor, FrameConcatenatedImagingExtractor
from roiextractors.tools.testing import generate_mock_video, generate_mock_imaging_extractor
import pytest
import numpy as np


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

    @pytest.mark.parametrize("start_frame, end_frame", [(0, 1), (0, 4), (6, 7)])
    def test_get_video_slice(self, imaging_extractor, expected_video, start_frame, end_frame):
        video = imaging_extractor.get_video(start_frame=start_frame, end_frame=end_frame)
        np.testing.assert_array_equal(video, expected_video[start_frame:end_frame])

    @pytest.mark.parametrize("frame_idxs", [[0], [0, 1], [0, 2], [0, 1, 2], [2, 1, 0], [0, 4, 8]])
    def test_get_frames(self, imaging_extractor, expected_video, frame_idxs):
        super().test_get_frames(imaging_extractor, expected_video, frame_idxs)


@pytest.mark.parametrize(
    "parameter_sets",
    [
        [dict(sampling_frequency=1), dict(sampling_frequency=2)],
        [dict(num_rows=1), dict(num_rows=2)],
        [dict(num_columns=1), dict(num_columns=2)],
        [dict(dtype=np.int16), dict(dtype=np.float32)],
    ],
)
def test_check_consistency_between_imaging_extractors(parameter_sets):
    imaging_extractors = [generate_mock_imaging_extractor(**params) for params in parameter_sets]
    with pytest.raises(AssertionError):
        FrameConcatenatedImagingExtractor(imaging_extractors=imaging_extractors)


def test_FrameConcatenatedImagingExtractor_invalid():
    with pytest.raises(AssertionError):
        FrameConcatenatedImagingExtractor(imaging_extractors=[])
    with pytest.raises(AssertionError):
        imaging_extractor = generate_mock_imaging_extractor()
        FrameConcatenatedImagingExtractor(imaging_extractors=imaging_extractor)


def test_FrameConcatenatedImagingExtractor_with_times():
    video = generate_mock_video(size=(9, 2, 4))
    imaging_extractors = []
    times = np.arange(video.shape[0]) * 2.0
    frames = np.arange(video.shape[0])
    for i in range(3):
        timeseries = video[i * 3 : (i + 1) * 3]
        imaging_extractor = NumpyImagingExtractor(timeseries=timeseries, sampling_frequency=20.0)
        imaging_extractor.set_times(times=times[i * 3 : (i + 1) * 3])
        imaging_extractors.append(imaging_extractor)
    imaging_extractor = FrameConcatenatedImagingExtractor(imaging_extractors=imaging_extractors)
    np.testing.assert_array_equal(imaging_extractor.frame_to_time(frames=frames), times)
