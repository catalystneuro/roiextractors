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
        expected_frames.append(extractor.get_frames(frame_idxs=frame_idxs))
    expected_frames = np.array(expected_frames)
    expected_frames = np.moveaxis(expected_frames, 0, -1)
    assert np.all(frames == expected_frames)


@pytest.mark.parametrize("frame_idxs", [num_frames, [0, num_frames], [-num_frames - 1, -1]])
def test_get_frames_invalid(volumetric_imaging_extractor, frame_idxs):
    with pytest.raises(ValueError):
        volumetric_imaging_extractor.get_frames(frame_idxs=frame_idxs)


@pytest.mark.parametrize("num_rows, num_columns, num_planes", [(1, 2, 3), (2, 1, 3), (3, 2, 1)])
def test_get_image_size(num_rows, num_columns, num_planes):
    imaging_extractors = [
        generate_dummy_imaging_extractor(num_rows=num_rows, num_columns=num_columns) for _ in range(num_planes)
    ]
    volumetric_imaging_extractor = VolumetricImagingExtractor(imaging_extractors=imaging_extractors)
    assert volumetric_imaging_extractor.get_image_size() == (num_rows, num_columns, num_planes)


@pytest.mark.parametrize("num_planes", [1, 2, 3])
def test_get_num_planes(num_planes):
    imaging_extractors = [generate_dummy_imaging_extractor() for _ in range(num_planes)]
    volumetric_imaging_extractor = VolumetricImagingExtractor(imaging_extractors=imaging_extractors)
    assert volumetric_imaging_extractor.get_num_planes() == num_planes


@pytest.mark.parametrize("num_frames", [1, 2, 3])
def test_get_num_frames(num_frames):
    imaging_extractors = [generate_dummy_imaging_extractor(num_frames=num_frames)]
    volumetric_imaging_extractor = VolumetricImagingExtractor(imaging_extractors=imaging_extractors)
    assert volumetric_imaging_extractor.get_num_frames() == num_frames


@pytest.mark.parametrize("sampling_frequency", [1, 2, 3])
def test_get_sampling_frequency(sampling_frequency):
    imaging_extractors = [generate_dummy_imaging_extractor(sampling_frequency=sampling_frequency)]
    volumetric_imaging_extractor = VolumetricImagingExtractor(imaging_extractors=imaging_extractors)
    assert volumetric_imaging_extractor.get_sampling_frequency() == sampling_frequency


@pytest.mark.parametrize("channel_names", [["Channel 1"], [" Channel 1 ", "Channel 2"]])
def test_get_channel_names(channel_names):
    imaging_extractors = [
        generate_dummy_imaging_extractor(channel_names=channel_names, num_channels=len(channel_names))
    ]
    volumetric_imaging_extractor = VolumetricImagingExtractor(imaging_extractors=imaging_extractors)
    assert volumetric_imaging_extractor.get_channel_names() == channel_names


@pytest.mark.parametrize("num_channels", [1, 2, 3])
def test_get_num_channels(num_channels):
    imaging_extractors = [generate_dummy_imaging_extractor(num_channels=num_channels)]
    volumetric_imaging_extractor = VolumetricImagingExtractor(imaging_extractors=imaging_extractors)
    assert volumetric_imaging_extractor.get_num_channels() == num_channels


@pytest.mark.parametrize("dtype", [np.float64, np.int16, np.uint8])
def test_get_dtype(dtype):
    imaging_extractors = [generate_dummy_imaging_extractor(dtype=dtype)]
    volumetric_imaging_extractor = VolumetricImagingExtractor(imaging_extractors=imaging_extractors)
    assert volumetric_imaging_extractor.get_dtype() == dtype


def test_depth_slice(volumetric_imaging_extractor):
    start_plane = 1
    end_plane = 2
    new_extractor = volumetric_imaging_extractor.depth_slice(start_plane=start_plane, end_plane=end_plane)
    assert new_extractor.get_num_planes() == end_plane - start_plane
