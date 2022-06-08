import unittest

from hdmf.testing import TestCase
import numpy as np
from numpy.testing import assert_array_equal

from roiextractors import NumpySegmentationExtractor


class TestFrameSliceSegmentation(TestCase):
    @classmethod
    def setUpClass(cls):
        """Use a toy example of ten frames of a 5 x 4 grayscale image."""
        cls.toy_data = np.random.random(size=(1, 10, 5, 4))
        cls.toy_imaging_example = NumpySegmentationExtractor(timeseries=cls.toy_data, sampling_frequency=1.0)
        cls.frame_sliced_segmentation = cls.toy_imaging_example.frame_slice(start_frame=2, end_frame=7)

    def test_get_image_size(self):
        assert self.frame_sliced_segmentation.get_image_size() == (5, 4)

    def test_get_num_frames(self):
        assert self.frame_sliced_segmentation.get_num_frames() == 5

    def test_get_sampling_frequency(self):
        assert self.frame_sliced_segmentation.get_sampling_frequency() == 1.0

    def test_get_channel_names(self):
        assert self.frame_sliced_segmentation.get_channel_names() == ["channel_0"]

    def test_get_num_channels(self):
        assert self.frame_sliced_segmentation.get_num_channels() == 1

    def test_get_frames_assertion(self):
        with self.assertRaisesWith(
            exc_type=AssertionError, exc_msg="'frame_idxs' range beyond number of available frames!"
        ):
            self.frame_sliced_segmentation.get_frames(frame_idxs=[6])

    def test_get_frames(self):
        assert_array_equal(
            self.frame_sliced_segmentation.get_frames(frame_idxs=[2, 4]), self.toy_data[:, [4, 6], ...].squeeze()
        )


if __name__ == "__main__":
    unittest.main()
