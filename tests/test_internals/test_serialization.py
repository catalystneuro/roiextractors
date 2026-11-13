import copy
import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from roiextractors import (
    MultiImagingExtractor,
    PlanarStackImagingExtractor,
    TiffImagingExtractor,
)
from roiextractors.testing import MockImagingExtractor


def test_a_drawn_seed_is_recorded_so_the_rebuilt_extractor_generates_the_same_values():
    extractor = MockImagingExtractor(num_samples=12, num_rows=6, num_columns=4, seed=None)

    rebuilt = pickle.loads(pickle.dumps(extractor))

    assert_array_equal(rebuilt.get_series(), extractor.get_series())


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda extractor: extractor.slice_samples(start_sample=2, end_sample=9), id="slice_samples"),
        pytest.param(
            lambda extractor: extractor.slice_field_of_view(row_start=1, row_end=5, column_start=0, column_end=3),
            id="slice_field_of_view",
        ),
        pytest.param(
            lambda extractor: extractor.slice_samples(2, 9).slice_field_of_view(1, 5, 0, 3), id="chained_slices"
        ),
        pytest.param(
            lambda extractor: MultiImagingExtractor(
                [extractor, MockImagingExtractor(num_samples=5, num_rows=6, num_columns=4, seed=1)]
            ),
            id="multi",
        ),
        pytest.param(
            lambda extractor: PlanarStackImagingExtractor(
                [extractor, MockImagingExtractor(num_samples=12, num_rows=6, num_columns=4, seed=1)]
            ),
            id="planar_stack",
        ),
        pytest.param(
            lambda extractor: PlanarStackImagingExtractor(
                [extractor, MockImagingExtractor(num_samples=12, num_rows=6, num_columns=4, seed=1)]
            ).select_plane(1),
            id="select_plane",
        ),
    ],
)
def test_wrappers_pickle_through_their_parents(wrap):
    parent = MockImagingExtractor(num_samples=12, num_rows=6, num_columns=4, seed=0)
    extractor = wrap(parent)

    unpickled = pickle.loads(pickle.dumps(extractor))

    assert type(unpickled) is type(extractor)
    assert unpickled.get_sample_shape() == extractor.get_sample_shape()
    assert_array_equal(unpickled.get_series(), extractor.get_series())


def test_pickle_ships_the_construction_rather_than_the_data(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    video = np.random.default_rng(0).integers(0, 2**12, size=(40, 32, 24), dtype="uint16")
    file_path = tmp_path / "movie.tif"
    tifffile.imwrite(file_path, video)
    extractor = TiffImagingExtractor(file_path=file_path, sampling_frequency=20.0)

    payload = pickle.dumps(extractor)
    unpickled = pickle.loads(payload)

    assert len(payload) < video.nbytes / 10
    assert type(unpickled) is TiffImagingExtractor
    assert_array_equal(unpickled.get_series(), video)


def test_pickle_preserves_state():
    # Right now the only state of an imaging extractor is the times from `set_times` and the shift from `shift_times`
    extractor = MockImagingExtractor(num_samples=5, num_rows=2, num_columns=2)
    times = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    extractor.set_times(times)
    extractor.shift_times(2.5)

    unpickled = pickle.loads(pickle.dumps(extractor))

    assert "times" not in unpickled._init_kwargs
    assert_array_equal(unpickled.get_timestamps(), times + 2.5)
    assert unpickled.get_start_time() == 3.5


def test_copy_stays_a_shallow_copy(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    file_path = tmp_path / "movie.tif"
    tifffile.imwrite(file_path, np.zeros((3, 4, 5), dtype="uint16"), photometric="minisblack")
    extractor = TiffImagingExtractor(file_path=file_path, sampling_frequency=20.0)

    copied = copy.copy(extractor)

    assert copied is not extractor
    assert copied._video is extractor._video  # not reopened, as a rebuild from the construction would do
    assert copied._init_kwargs == extractor._init_kwargs
