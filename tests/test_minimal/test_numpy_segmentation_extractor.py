from ..mixins.segmentation_extractor_mixin import SegmentationExtractorMixin
from roiextractors import NumpySegmentationExtractor
from roiextractors.testing import generate_dummy_video
import pytest
import numpy as np


class TestNumpyImagingExtractor(SegmentationExtractorMixin):
    @pytest.fixture(scope="class")
    def rng(self):
        seed = 1727293748  # int(datetime.now().timestamp()) at the time of writing
        return np.random.default_rng(seed=seed)

    @pytest.fixture(scope="class")
    def num_rows(self):
        return 25

    @pytest.fixture(scope="class")
    def num_columns(self):
        return 25

    @pytest.fixture(scope="class")
    def num_rois(self):
        return 10

    @pytest.fixture(scope="class")
    def num_frames(self):
        return 100

    @pytest.fixture(scope="class")
    def num_background_components(self):
        return 3

    @pytest.fixture(scope="class")
    def expected_image_masks(self, rng, num_rows, num_columns, num_rois):
        return rng.random((num_rows, num_columns, num_rois))

    @pytest.fixture(scope="class")
    def expected_roi_response_traces(self, rng, num_frames, num_rois):
        trace_names = ["raw", "dff", "deconvolved", "denoised"]
        traces_dict = {name: rng.random((num_frames, num_rois)) for name in trace_names}
        return traces_dict

    @pytest.fixture(scope="class")
    def expected_background_response_traces(self, rng, num_frames, num_background_components):
        trace_names = ["background"]
        traces_dict = {name: rng.random((num_frames, num_background_components)) for name in trace_names}
        return traces_dict

    @pytest.fixture(scope="class")
    def expected_summary_images(self, rng, num_rows, num_columns):
        image_names = ["mean", "correlation"]
        summary_images = {name: rng.random((num_rows, num_columns)) for name in image_names}
        return summary_images

    @pytest.fixture(scope="class")
    def expected_roi_ids(self, num_rois):
        return list(range(num_rois))

    @pytest.fixture(scope="class")
    def expected_roi_locations(self, rng, num_rois, num_rows, num_columns):
        roi_locations_rows = rng.integers(low=0, high=num_rows, size=num_rois)
        roi_locations_columns = rng.integers(low=0, high=num_columns, size=num_rois)
        roi_locations = np.vstack((roi_locations_rows, roi_locations_columns))
        return roi_locations

    @pytest.fixture(scope="class")
    def expected_accepted_list(self, rng, expected_roi_ids, num_rois):
        return rng.choice(expected_roi_ids, size=num_rois // 2, replace=False)

    @pytest.fixture(scope="class")
    def expected_rejected_list(self, expected_roi_ids, expected_accepted_list):
        return list(set(expected_roi_ids) - set(expected_accepted_list))

    @pytest.fixture(scope="class")
    def expected_sampling_frequency(self):
        return 30.0

    @pytest.fixture(scope="class")
    def expected_background_ids(self, num_background_components):
        return list(range(num_background_components))

    @pytest.fixture(scope="class")
    def expected_background_image_masks(self, rng, num_rows, num_columns, num_background_components):
        return rng.random((num_rows, num_columns, num_background_components))

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
