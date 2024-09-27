from ..mixins.segmentation_extractor_mixin import SegmentationExtractorMixin
from roiextractors import NumpySegmentationExtractor
import pytest
import numpy as np


@pytest.fixture(scope="module")
def rng():
    seed = 1727293748  # int(datetime.now().timestamp()) at the time of writing
    return np.random.default_rng(seed=seed)


@pytest.fixture(scope="module")
def num_rows():
    return 25


@pytest.fixture(scope="module")
def num_columns():
    return 25


@pytest.fixture(scope="module")
def num_rois():
    return 10


@pytest.fixture(scope="module")
def num_frames():
    return 100


@pytest.fixture(scope="module")
def num_background_components():
    return 3


@pytest.fixture(scope="module")
def expected_image_masks(rng, num_rows, num_columns, num_rois):
    return rng.random((num_rows, num_columns, num_rois))


@pytest.fixture(scope="module")
def expected_roi_response_traces(rng, num_frames, num_rois):
    trace_names = ["raw", "dff", "deconvolved", "denoised"]
    traces_dict = {name: rng.random((num_frames, num_rois)) for name in trace_names}
    return traces_dict


@pytest.fixture(scope="module")
def expected_background_response_traces(rng, num_frames, num_background_components):
    trace_names = ["background"]
    traces_dict = {name: rng.random((num_frames, num_background_components)) for name in trace_names}
    return traces_dict


@pytest.fixture(scope="module")
def expected_summary_images(rng, num_rows, num_columns):
    image_names = ["mean", "correlation"]
    summary_images = {name: rng.random((num_rows, num_columns)) for name in image_names}
    return summary_images


@pytest.fixture(scope="module")
def expected_roi_ids(num_rois):
    return list(range(num_rois))


@pytest.fixture(scope="module")
def expected_roi_locations(rng, num_rois, num_rows, num_columns):
    roi_locations_rows = rng.integers(low=0, high=num_rows, size=num_rois)
    roi_locations_columns = rng.integers(low=0, high=num_columns, size=num_rois)
    roi_locations = np.vstack((roi_locations_rows, roi_locations_columns))
    return roi_locations


@pytest.fixture(scope="module")
def expected_accepted_list(rng, expected_roi_ids, num_rois):
    return rng.choice(expected_roi_ids, size=num_rois // 2, replace=False)


@pytest.fixture(scope="module")
def expected_rejected_list(expected_roi_ids, expected_accepted_list):
    return list(set(expected_roi_ids) - set(expected_accepted_list))


@pytest.fixture(scope="module")
def expected_sampling_frequency():
    return 30.0


@pytest.fixture(scope="module")
def expected_background_ids(num_background_components):
    return list(range(num_background_components))


@pytest.fixture(scope="module")
def expected_background_image_masks(rng, num_rows, num_columns, num_background_components):
    return rng.random((num_rows, num_columns, num_background_components))


class TestNumpySegmentationExtractor(SegmentationExtractorMixin):
    @pytest.fixture(scope="function")
    def segmentation_extractor(
        self,
        expected_image_masks,
        expected_roi_response_traces,
        expected_summary_images,
        expected_roi_ids,
        expected_roi_locations,
        expected_accepted_list,
        expected_rejected_list,
        expected_background_response_traces,
        expected_background_ids,
        expected_background_image_masks,
        expected_sampling_frequency,
    ):
        return NumpySegmentationExtractor(
            image_masks=expected_image_masks,
            roi_response_traces=expected_roi_response_traces,
            summary_images=expected_summary_images,
            roi_ids=expected_roi_ids,
            roi_locations=expected_roi_locations,
            accepted_roi_ids=expected_accepted_list,
            rejected_roi_ids=expected_rejected_list,
            sampling_frequency=expected_sampling_frequency,
            background_ids=expected_background_ids,
            background_image_masks=expected_background_image_masks,
            background_response_traces=expected_background_response_traces,
        )


class TestNumpySegmentationExtractorFromFile(SegmentationExtractorMixin):
    @pytest.fixture(scope="function")
    def segmentation_extractor(
        self,
        expected_image_masks,
        expected_roi_response_traces,
        expected_summary_images,
        expected_roi_ids,
        expected_roi_locations,
        expected_accepted_list,
        expected_rejected_list,
        expected_background_response_traces,
        expected_background_ids,
        expected_background_image_masks,
        expected_sampling_frequency,
        tmp_path,
    ):
        name_to_ndarray = dict(
            image_masks=expected_image_masks,
            background_image_masks=expected_background_image_masks,
        )
        name_to_file_path = {}
        for name, ndarray in name_to_ndarray.items():
            file_path = tmp_path / f"{name}.npy"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(file_path, ndarray)
            name_to_file_path[name] = file_path
        name_to_dict_of_ndarrays = dict(
            roi_response_traces=expected_roi_response_traces,
            background_response_traces=expected_background_response_traces,
            summary_images=expected_summary_images,
        )
        name_to_dict_of_file_paths = {}
        for name, dict_of_ndarrays in name_to_dict_of_ndarrays.items():
            name_to_dict_of_file_paths[name] = {}
            for key, ndarray in dict_of_ndarrays.items():
                file_path = tmp_path / f"{name}_{key}.npy"
                np.save(file_path, ndarray)
                name_to_dict_of_file_paths[name][key] = file_path

        return NumpySegmentationExtractor(
            **name_to_file_path,
            **name_to_dict_of_file_paths,
            roi_ids=expected_roi_ids,
            roi_locations=expected_roi_locations,
            accepted_roi_ids=expected_accepted_list,
            rejected_roi_ids=expected_rejected_list,
            sampling_frequency=expected_sampling_frequency,
            background_ids=expected_background_ids,
        )
