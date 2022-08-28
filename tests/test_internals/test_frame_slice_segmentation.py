import unittest

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

    toy_segmentation_example = generate_dummy_segmentation_extractor(
        num_frames=num_frames, num_rows=5, num_columns=4, num_channels=1
    )
    toy_segmentation_example.set_times(times=times)

    frame_sliced_segmentation = toy_segmentation_example.frame_slice(start_frame=start_frame, end_frame=end_frame)
    assert_array_equal(
        frame_sliced_segmentation.frame_to_time(
            frames=np.array([idx for idx in range(frame_sliced_segmentation.get_num_frames())])
        ),
        times[start_frame:end_frame],
    )


def segmentation_name_function(testcase_function, param_number, param):
    return f"{testcase_function.__name__}_{param_number}_{parameterized.to_safe_name(param.kwargs['name'].__name__)}"


class TestFrameSlicesegmentation(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.toy_segmentation_example = generate_dummy_segmentation_extractor(num_frames=10, num_rows=5, num_columns=4)
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
        assert self.frame_sliced_segmentation.get_num_rois() == 30

    def test_get_accepted_list(self):
        return assert_array_equal(
            x=self.frame_sliced_segmentation.get_accepted_list(), y=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        )

    def test_get_rejected_list(self):
        return assert_array_equal(
            x=self.frame_sliced_segmentation.get_rejected_list(),
            y=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        )

    @parameterized.expand(
        [param(name="raw"), param(name="dff"), param(name="neuropil"), param(name="deconvolved")],
        name_func=segmentation_name_function,
    )
    def test_get_traces(self, name: str):
        assert_array_equal(
            x=self.frame_sliced_segmentation.get_traces(name=name),
            y=self.toy_segmentation_example.get_traces(start_frame=2, end_frame=7, name=name),
        )

    def test_get_traces_dict(self):
        true_dict = self.toy_segmentation_example.get_traces_dict()
        for key in true_dict:
            true_dict[key] = true_dict[key][2:7, :]
        self.assertContainerEqual(container1=self.frame_sliced_segmentation.get_traces_dict(), container2=true_dict)

    def test_get_images_dict(self):
        self.assertContainerEqual(
            container1=self.frame_sliced_segmentation.get_images_dict(),
            container2=self.toy_segmentation_example.get_images_dict(),
        )

    @parameterized.expand([param(name="mean"), param(name="correlation")], name_func=segmentation_name_function)
    def test_get_image(self, name: str):
        assert self.frame_sliced_segmentation.get_image(name=name) == self.toy_segmentation_example.get_image(name=name)


if __name__ == "__main__":
    unittest.main()
