import unittest
from types import MethodType

import numpy as np
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal
from parameterized import parameterized, param

from roiextractors.testing import generate_dummy_segmentation_extractor


def test_frame_slicing_segmentation_times():
    num_frames = 10
    timestamp_shift = 7.1
    times = np.array(range(num_frames)) + timestamp_shift
    start_frame, end_frame = 2, 7

    toy_segmentation_example = generate_dummy_segmentation_extractor(num_frames=num_frames, num_rows=5, num_columns=4)
    toy_segmentation_example.set_times(times=times)

    frame_sliced_segmentation = toy_segmentation_example.frame_slice(start_frame=start_frame, end_frame=end_frame)
    assert_array_equal(
        frame_sliced_segmentation.frame_to_time(
            frames=np.array([idx for idx in range(frame_sliced_segmentation.get_num_frames())])
        ),
        times[start_frame:end_frame],
    )


def segmentation_name_function(testcase_function, param_number, param):
    return f"{testcase_function.__name__}_{param_number}_{parameterized.to_safe_name(param.kwargs['name'])}"


class BaseTestFrameSlicesegmentation(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.toy_segmentation_example = generate_dummy_segmentation_extractor(num_frames=15, num_rows=5, num_columns=4)
        cls.frame_sliced_segmentation = cls.toy_segmentation_example.frame_slice(start_frame=2, end_frame=7)

    def test_get_image_size(self):
        assert self.frame_sliced_segmentation.get_image_size() == (5, 4)

    def test_get_num_planes(self):
        return self.frame_sliced_segmentation.get_num_planes() == 1

    def test_get_num_frames(self):
        assert self.frame_sliced_segmentation.get_num_frames() == 5

    def test_get_sampling_frequency(self):
        assert self.frame_sliced_segmentation.get_sampling_frequency() == 30.0

    def test_get_channel_names(self):
        assert self.frame_sliced_segmentation.get_channel_names() == ["channel_num_0"]

    def test_get_num_channels(self):
        assert self.frame_sliced_segmentation.get_num_channels() == 1

    def test_get_num_rois(self):
        assert self.frame_sliced_segmentation.get_num_rois() == 10

    def test_get_accepted_list(self):
        return assert_array_equal(self.frame_sliced_segmentation.get_accepted_list(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_get_rejected_list(self):
        return assert_array_equal(self.frame_sliced_segmentation.get_rejected_list(), [])

    @parameterized.expand(
        [param(name="raw"), param(name="dff"), param(name="neuropil"), param(name="deconvolved")],
        name_func=segmentation_name_function,
    )
    def test_get_traces(self, name: str):
        assert_array_equal(
            self.frame_sliced_segmentation.get_traces(name=name),
            self.toy_segmentation_example.get_traces(start_frame=2, end_frame=7, name=name),
        )

    def test_get_traces_dict(self):
        true_dict = self.toy_segmentation_example.get_traces_dict()
        for key in true_dict:
            true_dict[key] = true_dict[key][2:7, :] if true_dict[key] is not None else true_dict[key]
        self.assertCountEqual(first=self.frame_sliced_segmentation.get_traces_dict(), second=true_dict)

    def test_get_images_dict(self):
        self.assertCountEqual(
            first=self.frame_sliced_segmentation.get_images_dict(),
            second=self.toy_segmentation_example.get_images_dict(),
        )

    @parameterized.expand([param(name="mean"), param(name="correlation")], name_func=segmentation_name_function)
    def test_get_image(self, name: str):
        assert_array_equal(
            self.frame_sliced_segmentation.get_image(name=name), self.toy_segmentation_example.get_image(name=name)
        )


class TestMissingTraceFrameSlicesegmentation(BaseTestFrameSlicesegmentation):
    @classmethod
    def setUpClass(cls):
        cls.toy_segmentation_example = generate_dummy_segmentation_extractor(
            num_frames=15, num_rows=5, num_columns=4, has_dff_signal=False
        )
        cls.frame_sliced_segmentation = cls.toy_segmentation_example.frame_slice(start_frame=2, end_frame=7)


def test_frame_slicing_segmentation_missing_image_mask_attribute():
    """If the parent object does not have a _image_masks attribute, don't try to copy it to the sub extractor."""
    num_frames = 10
    start_frame, end_frame = 2, 7

    toy_segmentation_example = generate_dummy_segmentation_extractor(num_frames=num_frames)
    del toy_segmentation_example._image_masks

    frame_sliced_segmentation = toy_segmentation_example.frame_slice(start_frame=start_frame, end_frame=end_frame)
    assert not hasattr(frame_sliced_segmentation, "_image_masks")


def test_frame_slicing_segmentation_get_roi_pixel_masks_override():
    """If the parent overrides the base get_roi_pixel_masks() method, ensure this is used by the sub extractor."""
    num_frames = 10
    start_frame, end_frame = 2, 7

    def get_roi_pixel_masks_override(self, roi_ids=None) -> np.ndarray:
        return np.array([1, 2, 3])

    toy_segmentation_example = generate_dummy_segmentation_extractor(num_frames=num_frames)
    toy_segmentation_example.get_roi_pixel_masks = MethodType(get_roi_pixel_masks_override, toy_segmentation_example)

    frame_sliced_segmentation = toy_segmentation_example.frame_slice(start_frame=start_frame, end_frame=end_frame)
    np.testing.assert_array_equal(frame_sliced_segmentation.get_roi_pixel_masks(), np.array([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
