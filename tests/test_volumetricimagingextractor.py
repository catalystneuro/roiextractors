import pytest
import numpy as np
from roiextractors.testing import generate_dummy_imaging_extractor
from roiextractors import VolumetricImagingExtractor

num_frames = 10


@pytest.fixture(scope="module", params=[1, 2])
def imaging_extractors(request):
    num_channels = request.param
    return [generate_dummy_imaging_extractor(num_channels=num_channels, num_frames=num_frames) for _ in range(3)]


@pytest.fixture(scope="module")
def volumetric_imaging_extractor(imaging_extractors):
    return VolumetricImagingExtractor(imaging_extractors)


@pytest.mark.parametrize(
    "params",
    [
        [dict(sampling_frequency=1), dict(sampling_frequency=2)],
        [dict(num_rows=1), dict(num_rows=2)],
        [dict(num_channels=1), dict(num_channels=2)],
        [dict(channel_names=["a"], num_channels=1), dict(channel_names=["b"], num_channels=1)],
        [dict(dtype=np.int16), dict(dtype=np.float32)],
        [dict(num_frames=1), dict(num_frames=2)],
    ],
)
def test_check_consistency_between_imaging_extractors(params):
    imaging_extractors = [generate_dummy_imaging_extractor(**param) for param in params]
    with pytest.raises(AssertionError):
        VolumetricImagingExtractor(imaging_extractors=imaging_extractors)


@pytest.mark.parametrize("start_frame, end_frame", [(None, None), (0, num_frames), (3, 7), (-2, -1)])
def test_get_video(volumetric_imaging_extractor, start_frame, end_frame):
    video = volumetric_imaging_extractor.get_video(start_frame=start_frame, end_frame=end_frame)
    expected_video = []
    for extractor in volumetric_imaging_extractor._imaging_extractors:
        expected_video.append(extractor.get_video(start_frame=start_frame, end_frame=end_frame))
    expected_video = np.array(expected_video)
    expected_video = np.moveaxis(expected_video, 0, -1)
    assert np.all(video == expected_video)


@pytest.mark.parametrize("start_frame, end_frame", [(num_frames + 1, None), (None, num_frames + 1), (2, 1)])
def test_get_video_invalid(volumetric_imaging_extractor, start_frame, end_frame):
    with pytest.raises(ValueError):
        volumetric_imaging_extractor.get_video(start_frame=start_frame, end_frame=end_frame)


@pytest.mark.parametrize("frame_idxs", [0, [0, 1, 2], [0, num_frames - 1], [-3, -1]])
def test_get_frames(volumetric_imaging_extractor, frame_idxs):
    frames = volumetric_imaging_extractor.get_frames(frame_idxs=frame_idxs)
    expected_frames = []
    for extractor in volumetric_imaging_extractor._imaging_extractors:
        gotten_frames = extractor.get_frames(frame_idxs=frame_idxs)
        expected_frames.append(extractor.get_frames(frame_idxs=frame_idxs))
    expected_frames = np.array(expected_frames)
    expected_frames = np.moveaxis(expected_frames, 0, -1)
    assert np.all(frames == expected_frames)


@pytest.mark.parametrize("frame_idxs", [num_frames, [0, num_frames], [-num_frames - 1, -1]])
def test_get_frames_invalid(volumetric_imaging_extractor, frame_idxs):
    with pytest.raises(ValueError):
        volumetric_imaging_extractor.get_frames(frame_idxs=frame_idxs)
