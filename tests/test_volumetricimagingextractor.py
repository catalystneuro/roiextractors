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


@pytest.mark.parametrize("start_sample, end_sample", [(None, None), (0, num_frames), (3, 7), (-2, -1)])
def test_get_series(volumetric_imaging_extractor, start_sample, end_sample):
    series = volumetric_imaging_extractor.get_series(start_sample=start_sample, end_sample=end_sample)
    expected_series = []
    for extractor in volumetric_imaging_extractor._imaging_extractors:
        expected_series.append(extractor.get_series(start_sample=start_sample, end_sample=end_sample))
    expected_series = np.array(expected_series)
    expected_series = np.moveaxis(expected_series, 0, -1)
    assert np.all(series == expected_series)


@pytest.mark.parametrize("start_sample, end_sample", [(num_frames + 1, None), (None, num_frames + 1), (2, 1)])
def test_get_series_invalid(volumetric_imaging_extractor, start_sample, end_sample):
    with pytest.raises(ValueError):
        volumetric_imaging_extractor.get_series(start_sample=start_sample, end_sample=end_sample)


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


@pytest.mark.parametrize("dtype", [np.float64, np.int16, np.uint8])
def test_get_dtype(dtype):
    imaging_extractors = [generate_dummy_imaging_extractor(dtype=dtype)]
    volumetric_imaging_extractor = VolumetricImagingExtractor(imaging_extractors=imaging_extractors)
    assert volumetric_imaging_extractor.get_dtype() == dtype


@pytest.mark.parametrize("start_plane, end_plane", [(None, None), (0, 1), (1, 2)])
def test_depth_slice(volumetric_imaging_extractor, start_plane, end_plane):
    start_plane = start_plane or 0
    end_plane = end_plane or volumetric_imaging_extractor.get_num_planes()
    sliced_extractor = volumetric_imaging_extractor.depth_slice(start_plane=start_plane, end_plane=end_plane)

    assert sliced_extractor.get_num_planes() == end_plane - start_plane
    assert sliced_extractor.get_image_size() == (
        *volumetric_imaging_extractor.get_image_size()[:2],
        end_plane - start_plane,
    )
    series = volumetric_imaging_extractor.get_series()
    sliced_series = sliced_extractor.get_series()
    assert np.all(series[..., start_plane:end_plane] == sliced_series)
    frames = volumetric_imaging_extractor.get_frames(frame_idxs=[0, 1, 2])
    sliced_frames = sliced_extractor.get_frames(frame_idxs=[0, 1, 2])
    assert np.all(frames[..., start_plane:end_plane] == sliced_frames)


@pytest.mark.parametrize("start_plane, end_plane", [(0, -1), (1, 0), (0, 4)])
def test_depth_slice_invalid(volumetric_imaging_extractor, start_plane, end_plane):
    with pytest.raises(AssertionError):
        volumetric_imaging_extractor.depth_slice(start_plane=start_plane, end_plane=end_plane)


def test_depth_slice_twice(volumetric_imaging_extractor):
    sliced_extractor = volumetric_imaging_extractor.depth_slice(start_plane=0, end_plane=2)
    twice_sliced_extractor = sliced_extractor.depth_slice(start_plane=0, end_plane=1)

    assert twice_sliced_extractor.get_num_planes() == 1
    assert twice_sliced_extractor.get_image_size() == (*volumetric_imaging_extractor.get_image_size()[:2], 1)
    series = volumetric_imaging_extractor.get_series()
    sliced_series = twice_sliced_extractor.get_series()
    assert np.all(series[..., :1] == sliced_series)
    frames = volumetric_imaging_extractor.get_frames(frame_idxs=[0, 1, 2])
    sliced_frames = twice_sliced_extractor.get_frames(frame_idxs=[0, 1, 2])
    assert np.all(frames[..., :1] == sliced_frames)


def test_frame_slice(volumetric_imaging_extractor):
    with pytest.raises(NotImplementedError):
        volumetric_imaging_extractor.frame_slice(start_frame=0, end_frame=1)


def test_is_volumetric_flag(volumetric_imaging_extractor):
    """Test that the is_volumetric flag is True for VolumetricImagingExtractor."""
    assert hasattr(
        volumetric_imaging_extractor, "is_volumetric"
    ), "VolumetricImagingExtractor should have is_volumetric attribute"
    assert (
        volumetric_imaging_extractor.is_volumetric is True
    ), "is_volumetric should be True for VolumetricImagingExtractor"


def test_get_volume_shape(volumetric_imaging_extractor):
    """Test that the get_volume_shape method returns the correct shape."""
    # Check that the method exists
    assert hasattr(
        volumetric_imaging_extractor, "get_volume_shape"
    ), "VolumetricImagingExtractor should have get_volume_shape method"

    # Check that the method returns the correct shape
    image_shape = volumetric_imaging_extractor.get_image_shape()
    num_planes = volumetric_imaging_extractor.get_num_planes()
    volume_shape = volumetric_imaging_extractor.get_volume_shape()

    assert len(volume_shape) == 3, "get_volume_shape should return a 3-tuple"
    assert volume_shape == (
        image_shape[0],
        image_shape[1],
        num_planes,
    ), "get_volume_shape should return (num_rows, num_columns, num_planes)"
