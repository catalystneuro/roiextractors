import unittest
from types import MethodType

import numpy as np
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal
from parameterized import param, parameterized

from roiextractors.testing import generate_dummy_segmentation_extractor


def test_sample_slicing_segmentation_times():
    num_samples = 10
    timestamp_shift = 7.1
    times = np.array(range(num_samples)) + timestamp_shift
    start_sample, end_sample = 2, 7

    toy_segmentation_example = generate_dummy_segmentation_extractor(num_frames=num_samples, num_rows=5, num_columns=4)
    toy_segmentation_example.set_times(times=times)

    sample_sliced_segmentation = toy_segmentation_example.slice_samples(
        start_sample=start_sample, end_sample=end_sample
    )
    assert_array_equal(
        sample_sliced_segmentation.frame_to_time(
            frames=np.array([idx for idx in range(sample_sliced_segmentation.get_num_samples())])
        ),
        times[start_sample:end_sample],
    )


def segmentation_name_function(testcase_function, param_number, param):
    return f"{testcase_function.__name__}_{param_number}_{parameterized.to_safe_name(param.kwargs['name'])}"


class BaseTestSampleSliceSegmentation(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.toy_segmentation_example = generate_dummy_segmentation_extractor(num_frames=15, num_rows=5, num_columns=4)
        cls.sample_sliced_segmentation = cls.toy_segmentation_example.slice_samples(start_sample=2, end_sample=7)

    def test_get_frame_shape(self):
        assert self.sample_sliced_segmentation.get_frame_shape() == (5, 4)

    def test_get_num_planes(self):
        return self.sample_sliced_segmentation.get_num_planes() == 1

    def test_get_num_samples(self):
        assert self.sample_sliced_segmentation.get_num_samples() == 5

    def test_get_sampling_frequency(self):
        assert self.sample_sliced_segmentation.get_sampling_frequency() == 30.0

    def test_get_channel_names(self):
        assert self.sample_sliced_segmentation.get_channel_names() == ["channel_num_0"]

    def test_get_num_channels(self):
        assert self.sample_sliced_segmentation.get_num_channels() == 1

    def test_get_num_rois(self):
        assert self.sample_sliced_segmentation.get_num_rois() == 10

    def test_get_accepted_list(self):
        return assert_array_equal(
            self.sample_sliced_segmentation.get_accepted_list(),
            ["roi_0", "roi_1", "roi_2", "roi_3", "roi_4", "roi_5", "roi_6", "roi_7", "roi_8", "roi_9"],
        )

    def test_get_rejected_list(self):
        return assert_array_equal(self.sample_sliced_segmentation.get_rejected_list(), [])

    @parameterized.expand(
        [param(name="raw"), param(name="dff"), param(name="neuropil"), param(name="deconvolved")],
        name_func=segmentation_name_function,
    )
    def test_get_traces(self, name: str):
        assert_array_equal(
            self.sample_sliced_segmentation.get_traces(name=name),
            self.toy_segmentation_example.get_traces(start_frame=2, end_frame=7, name=name),
        )

    def test_get_traces_dict(self):
        true_dict = self.toy_segmentation_example.get_traces_dict()
        for key in true_dict:
            true_dict[key] = true_dict[key][2:7, :] if true_dict[key] is not None else true_dict[key]
        self.assertCountEqual(first=self.sample_sliced_segmentation.get_traces_dict(), second=true_dict)

    def test_get_images_dict(self):
        self.assertCountEqual(
            first=self.sample_sliced_segmentation.get_images_dict(),
            second=self.toy_segmentation_example.get_images_dict(),
        )

    @parameterized.expand([param(name="mean"), param(name="correlation")], name_func=segmentation_name_function)
    def test_get_image(self, name: str):
        assert_array_equal(
            self.sample_sliced_segmentation.get_image(name=name), self.toy_segmentation_example.get_image(name=name)
        )


class TestMissingTraceSampleSliceSegmentation(BaseTestSampleSliceSegmentation):
    @classmethod
    def setUpClass(cls):
        cls.toy_segmentation_example = generate_dummy_segmentation_extractor(
            num_frames=15, num_rows=5, num_columns=4, has_dff_signal=False
        )
        cls.sample_sliced_segmentation = cls.toy_segmentation_example.slice_samples(start_sample=2, end_sample=7)


def test_sample_slicing_segmentation_missing_image_mask_attribute():
    """If the parent object has None for _roi_representations, the sliced extractor should also have None."""
    num_samples = 10
    start_sample, end_sample = 2, 7

    toy_segmentation_example = generate_dummy_segmentation_extractor(num_frames=num_samples)
    # Set to None to simulate an extractor that hasn't populated ROI representations yet
    toy_segmentation_example._roi_representations = None

    sample_sliced_segmentation = toy_segmentation_example.slice_samples(
        start_sample=start_sample, end_sample=end_sample
    )
    # The sliced extractor should have the attribute (initialized in __init__) but it should be None
    assert hasattr(sample_sliced_segmentation, "_roi_representations")
    assert sample_sliced_segmentation._roi_representations is None


def test_sample_slicing_segmentation_get_roi_pixel_masks_override():
    """If the parent overrides the base get_roi_pixel_masks() method, ensure this is used by the sub extractor."""
    num_samples = 10
    start_sample, end_sample = 2, 7

    def get_roi_pixel_masks_override(self, roi_ids=None) -> np.ndarray:
        return np.array([1, 2, 3])

    toy_segmentation_example = generate_dummy_segmentation_extractor(num_frames=num_samples)
    toy_segmentation_example.get_roi_pixel_masks = MethodType(get_roi_pixel_masks_override, toy_segmentation_example)

    sample_sliced_segmentation = toy_segmentation_example.slice_samples(
        start_sample=start_sample, end_sample=end_sample
    )
    np.testing.assert_array_equal(sample_sliced_segmentation.get_roi_pixel_masks(), np.array([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
