import unittest

from hdmf.testing import TestCase
from numpy.testing import assert_array_equal

from roiextractors.testing import generate_dummy_imaging_extractor


class TestFrameSliceImaging(TestCase):
    @classmethod
    def setUpClass(cls):
        """Use a toy example of ten frames of a 5 x 4 grayscale image."""
        cls.toy_imaging_example = generate_dummy_imaging_extractor(num_frames=10, rows=5, columns=4, num_channels=1)
        cls.frame_sliced_imaging = cls.toy_imaging_example.frame_slice(start_frame=2, end_frame=7)

    def test_get_image_size(self):
        assert self.frame_sliced_imaging.get_image_size() == (5, 4)

    def test_get_num_frames(self):
        assert self.frame_sliced_imaging.get_num_frames() == 5

    def test_get_sampling_frequency(self):
        assert self.frame_sliced_imaging.get_sampling_frequency() == 30.0

    def test_get_channel_names(self):
        assert self.frame_sliced_imaging.get_channel_names() == ["channel_num_0"]

    def test_get_num_channels(self):
        assert self.frame_sliced_imaging.get_num_channels() == 1

    def test_get_frames_assertion(self):
        with self.assertRaisesWith(
            exc_type=AssertionError, exc_msg="'frame_idxs' range beyond number of available frames!"
        ):
            self.frame_sliced_imaging.get_frames(frame_idxs=[6])

    def test_get_frames(self):
        assert_array_equal(
            self.frame_sliced_imaging.get_frames(frame_idxs=[2, 4]),
            self.toy_imaging_example.get_frames(frame_idxs=[4, 6]),
        )


if __name__ == "__main__":
    unittest.main()
