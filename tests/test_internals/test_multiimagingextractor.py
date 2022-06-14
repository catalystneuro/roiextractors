import unittest

import numpy as np
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal
from parameterized import parameterized

from roiextractors.multiimagingextractor import MultiImagingExtractor
from roiextractors.testing import generate_dummy_imaging_extractor


class TestMultiImagingExtractor(TestCase):
    extractors = None

    @classmethod
    def setUpClass(cls):
        dummy_extractor = generate_dummy_imaging_extractor(num_frames=10, rows=3, columns=4, sampling_frequency=20.0)
        cls.extractors = [dummy_extractor] * 3
        cls.multi_imaging_extractor = MultiImagingExtractor(imaging_extractors=cls.extractors)

    def test_get_image_size(self):
        assert self.multi_imaging_extractor.get_image_size() == self.extractors[0].get_image_size()

    def test_get_num_frames(self):
        assert self.multi_imaging_extractor.get_num_frames() == 30

    def test_get_sampling_frequency(self):
        assert self.multi_imaging_extractor.get_sampling_frequency() == 20.0

    def test_get_channel_names(self):
        assert self.multi_imaging_extractor.get_channel_names() == ["channel_num_0"]

    def test_get_num_channels(self):
        assert self.multi_imaging_extractor.get_num_channels() == 1

    def test_get_frames_assertion(self):
        with self.assertRaisesWith(
            exc_type=AssertionError, exc_msg="'frame_idxs' range beyond number of available frames!"
        ):
            self.multi_imaging_extractor.get_frames(frame_idxs=[31])

    def test_get_frames(self):
        expected_frames = np.concatenate(
            (
                self.extractors[0].get_frames(frame_idxs=[8])[np.newaxis, ...],
                self.extractors[1].get_frames(frame_idxs=[0, 2, 5]),
                self.extractors[2].get_frames(frame_idxs=[0, 9]),
            ),
            axis=0,
        )
        assert_array_equal(self.multi_imaging_extractor.get_frames(frame_idxs=[8, 10, 12, 15, 20, 29]), expected_frames)

    @parameterized.expand(
        [
            (3, 4, 15.0, 1, "The sampling frequency is not consistent over the files (found {20.0, 15.0})."),
            (3, 5, 20.0, 1, "The size of a frame is not consistent over the files (found {(3, 4), (3, 5)})."),
            (3, 4, 20.0, 2, "The number of channels is not consistent over the files (found {1, 2})."),
        ]
    )
    def test_inconsistent_sampling_frequency_assertion(self, rows, columns, sampling_frequency, num_channels, msg):
        inconsistent_extractors = [
            self.extractors[0],
            generate_dummy_imaging_extractor(
                num_frames=1,
                rows=rows,
                columns=columns,
                sampling_frequency=sampling_frequency,
                num_channels=num_channels,
            ),
        ]
        with self.assertRaisesWith(
            exc_type=AssertionError,
            exc_msg=msg,
        ):
            MultiImagingExtractor(imaging_extractors=inconsistent_extractors)


if __name__ == "__main__":
    unittest.main()
