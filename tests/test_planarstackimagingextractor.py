import numpy as np
import pytest

from roiextractors import PlanarStackImagingExtractor
from roiextractors.testing import generate_dummy_imaging_extractor

num_samples = 10


@pytest.fixture(scope="module")
def planar_stack_imaging_extractor():
    imaging_extractors = [generate_dummy_imaging_extractor(num_samples=num_samples) for _ in range(3)]
    return PlanarStackImagingExtractor(imaging_extractors)


@pytest.mark.parametrize(
    "params",
    [
        [dict(sampling_frequency=1), dict(sampling_frequency=2)],
        [dict(num_rows=1), dict(num_rows=2)],
        [dict(dtype=np.int16), dict(dtype=np.float32)],
        [dict(num_samples=1), dict(num_samples=2)],
    ],
)
def test_check_consistency_between_imaging_extractors(params):
    imaging_extractors = [generate_dummy_imaging_extractor(**param) for param in params]
    with pytest.raises(AssertionError):
        PlanarStackImagingExtractor(imaging_extractors=imaging_extractors)


@pytest.mark.parametrize("start_sample, end_sample", [(None, None), (0, num_samples), (3, 7), (-2, -1)])
def test_get_series(planar_stack_imaging_extractor, start_sample, end_sample):
    series = planar_stack_imaging_extractor.get_series(start_sample=start_sample, end_sample=end_sample)
    expected_series = []
    for extractor in planar_stack_imaging_extractor._imaging_extractors:
        expected_series.append(extractor.get_series(start_sample=start_sample, end_sample=end_sample))
    expected_series = np.array(expected_series)
    expected_series = np.moveaxis(expected_series, 0, -1)
    assert np.all(series == expected_series)


@pytest.mark.parametrize("start_sample, end_sample", [(num_samples + 1, None), (None, num_samples + 1), (2, 1)])
def test_get_series_invalid(planar_stack_imaging_extractor, start_sample, end_sample):
    with pytest.raises(ValueError):
        planar_stack_imaging_extractor.get_series(start_sample=start_sample, end_sample=end_sample)


@pytest.mark.parametrize("frame_idxs", [[0], [0, 1, 2], [0, num_samples - 1]])
def test_get_samples(planar_stack_imaging_extractor, frame_idxs):
    frames = planar_stack_imaging_extractor.get_samples(sample_indices=frame_idxs)
    expected_frames = []
    for extractor in planar_stack_imaging_extractor._imaging_extractors:
        expected_frames.append(extractor.get_samples(sample_indices=frame_idxs))
    expected_frames = np.array(expected_frames)
    expected_frames = np.moveaxis(expected_frames, 0, -1)
    assert np.all(frames == expected_frames)


@pytest.mark.parametrize("frame_idxs", [[num_samples], [0, num_samples]])
def test_get_samples_invalid(planar_stack_imaging_extractor, frame_idxs):
    with pytest.raises(AssertionError):
        planar_stack_imaging_extractor.get_samples(sample_indices=frame_idxs)


@pytest.mark.parametrize("num_rows, num_columns, num_planes", [(1, 2, 3), (2, 1, 3), (3, 2, 1)])
def test_get_sample_shape(num_rows, num_columns, num_planes):
    imaging_extractors = [
        generate_dummy_imaging_extractor(num_rows=num_rows, num_columns=num_columns) for _ in range(num_planes)
    ]
    planar_stack_imaging_extractor = PlanarStackImagingExtractor(imaging_extractors=imaging_extractors)
    assert planar_stack_imaging_extractor.get_sample_shape() == (num_rows, num_columns, num_planes)


@pytest.mark.parametrize("num_planes", [1, 2, 3])
def test_get_num_planes(num_planes):
    imaging_extractors = [generate_dummy_imaging_extractor() for _ in range(num_planes)]
    planar_stack_imaging_extractor = PlanarStackImagingExtractor(imaging_extractors=imaging_extractors)
    assert planar_stack_imaging_extractor.get_num_planes() == num_planes


@pytest.mark.parametrize("num_samples", [1, 2, 3])
def test_get_num_samples(num_samples):
    imaging_extractors = [generate_dummy_imaging_extractor(num_samples=num_samples)]
    planar_stack_imaging_extractor = PlanarStackImagingExtractor(imaging_extractors=imaging_extractors)
    assert planar_stack_imaging_extractor.get_num_samples() == num_samples


@pytest.mark.parametrize("sampling_frequency", [1, 2, 3])
def test_get_sampling_frequency(sampling_frequency):
    imaging_extractors = [generate_dummy_imaging_extractor(sampling_frequency=sampling_frequency)]
    planar_stack_imaging_extractor = PlanarStackImagingExtractor(imaging_extractors=imaging_extractors)
    assert planar_stack_imaging_extractor.get_sampling_frequency() == sampling_frequency


@pytest.mark.parametrize("dtype", [np.float64, np.int16, np.uint8])
def test_get_dtype(dtype):
    imaging_extractors = [generate_dummy_imaging_extractor(dtype=dtype)]
    planar_stack_imaging_extractor = PlanarStackImagingExtractor(imaging_extractors=imaging_extractors)
    assert planar_stack_imaging_extractor.get_dtype() == dtype


@pytest.mark.parametrize("start_plane, end_plane", [(None, None), (0, 1), (1, 2)])
def test_depth_slice(planar_stack_imaging_extractor, start_plane, end_plane):
    start_plane = start_plane or 0
    end_plane = end_plane or planar_stack_imaging_extractor.get_num_planes()
    sliced_extractor = planar_stack_imaging_extractor.depth_slice(start_plane=start_plane, end_plane=end_plane)

    assert sliced_extractor.get_num_planes() == end_plane - start_plane
    assert sliced_extractor.get_sample_shape() == (
        *planar_stack_imaging_extractor.get_frame_shape(),
        end_plane - start_plane,
    )
    series = planar_stack_imaging_extractor.get_series()
    sliced_series = sliced_extractor.get_series()
    assert np.all(series[..., start_plane:end_plane] == sliced_series)
    frames = planar_stack_imaging_extractor.get_samples(sample_indices=[0, 1, 2])
    sliced_frames = sliced_extractor.get_samples(sample_indices=[0, 1, 2])
    assert np.all(frames[..., start_plane:end_plane] == sliced_frames)


@pytest.mark.parametrize("start_plane, end_plane", [(0, -1), (1, 0), (0, 4)])
def test_depth_slice_invalid(planar_stack_imaging_extractor, start_plane, end_plane):
    with pytest.raises(AssertionError):
        planar_stack_imaging_extractor.depth_slice(start_plane=start_plane, end_plane=end_plane)


def test_depth_slice_twice(planar_stack_imaging_extractor):
    sliced_extractor = planar_stack_imaging_extractor.depth_slice(start_plane=0, end_plane=2)
    twice_sliced_extractor = sliced_extractor.depth_slice(start_plane=0, end_plane=1)

    assert twice_sliced_extractor.get_num_planes() == 1
    assert twice_sliced_extractor.get_sample_shape() == (*planar_stack_imaging_extractor.get_sample_shape()[:2], 1)
    series = planar_stack_imaging_extractor.get_series()
    sliced_series = twice_sliced_extractor.get_series()
    assert np.all(series[..., :1] == sliced_series)
    samples = planar_stack_imaging_extractor.get_samples(sample_indices=[0, 1, 2])
    sliced_samples = twice_sliced_extractor.get_samples(sample_indices=[0, 1, 2])
    assert np.all(samples[..., :1] == sliced_samples)


def test_slice_samples(planar_stack_imaging_extractor):
    with pytest.raises(NotImplementedError):
        planar_stack_imaging_extractor.slice_samples(start_sample=0, end_sample=1)


def test_is_volumetric_flag(planar_stack_imaging_extractor):
    """Test that the is_volumetric flag is True for PlanarStackImagingExtractor."""
    assert hasattr(
        planar_stack_imaging_extractor, "is_volumetric"
    ), "PlanarStackImagingExtractor should have is_volumetric attribute"
    assert (
        planar_stack_imaging_extractor.is_volumetric is True
    ), "is_volumetric should be True for PlanarStackImagingExtractor"


def test_get_volume_shape(planar_stack_imaging_extractor):
    """Test that the get_volume_shape method returns the correct shape."""
    assert hasattr(
        planar_stack_imaging_extractor, "get_volume_shape"
    ), "PlanarStackImagingExtractor should have get_volume_shape method"

    frame_shape = planar_stack_imaging_extractor.get_frame_shape()
    num_planes = planar_stack_imaging_extractor.get_num_planes()
    volume_shape = planar_stack_imaging_extractor.get_volume_shape()

    assert len(volume_shape) == 3, "get_volume_shape should return a 3-tuple"
    assert volume_shape == (
        frame_shape[0],
        frame_shape[1],
        num_planes,
    ), "get_volume_shape should return (num_rows, num_columns, num_planes)"
