import unittest
from pathlib import Path
from tempfile import mkdtemp
from copy import deepcopy
from itertools import product

import numpy as np
from parameterized import parameterized, param

from roiextractors.extraction_tools import VideoStructure, read_numpy_memmap_video


def custom_name_func(testcase_func, param_num, param):
    return f"{testcase_func.__name__}_{param_num}_" f"_{param.kwargs.get('case_name', '')}"


class TestVideoStructureClass(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = 10
        self.columns = 5
        self.num_channels = 3

        self.frame_axis = 0
        self.rows_axis = 1
        self.columns_axis = 2
        self.num_channels_axis = 3

        self.num_frames = 20

    def test_basic_parameters(self):

        expected_frame_shape = (self.rows, self.columns, self.num_channels)
        expected_pixels_per_frame = self.rows * self.columns * self.num_channels
        expected_video_shape = (self.num_frames, self.rows, self.columns, self.num_channels)

        video_structure = VideoStructure(
            rows=self.rows,
            columns=self.columns,
            num_channels=self.num_channels,
            rows_axis=self.rows_axis,
            columns_axis=self.columns_axis,
            num_channels_axis=self.num_channels_axis,
            frame_axis=self.frame_axis,
        )

        assert video_structure.frame_shape == expected_frame_shape
        assert video_structure.number_of_pixels_per_frame == expected_pixels_per_frame
        assert video_structure.build_video_shape(self.num_frames) == expected_video_shape

    def test_axis_permutation_1(self):

        self.frame_axis = 3
        self.rows_axis = 2
        self.columns_axis = 1
        self.num_channels_axis = 0

        expected_frame_shape = (self.num_channels, self.columns, self.rows)
        expected_pixels_per_frame = self.rows * self.columns * self.num_channels
        expected_video_shape = (self.num_channels, self.columns, self.rows, self.num_frames)

        video_structure = VideoStructure(
            rows=self.rows,
            columns=self.columns,
            num_channels=self.num_channels,
            rows_axis=self.rows_axis,
            columns_axis=self.columns_axis,
            num_channels_axis=self.num_channels_axis,
            frame_axis=self.frame_axis,
        )

        assert video_structure.frame_shape == expected_frame_shape
        assert video_structure.number_of_pixels_per_frame == expected_pixels_per_frame
        assert video_structure.build_video_shape(self.num_frames) == expected_video_shape

    def test_axis_permutation_2(self):

        self.frame_axis = 2
        self.rows_axis = 0
        self.columns_axis = 1
        self.num_channels_axis = 3

        expected_frame_shape = (self.rows, self.columns, self.num_channels)
        expected_pixels_per_frame = self.rows * self.columns * self.num_channels
        expected_video_shape = (self.rows, self.columns, self.num_frames, self.num_channels)

        video_structure = VideoStructure(
            rows=self.rows,
            columns=self.columns,
            num_channels=self.num_channels,
            rows_axis=self.rows_axis,
            columns_axis=self.columns_axis,
            num_channels_axis=self.num_channels_axis,
            frame_axis=self.frame_axis,
        )

        assert video_structure.frame_shape == expected_frame_shape
        assert video_structure.number_of_pixels_per_frame == expected_pixels_per_frame
        assert video_structure.build_video_shape(self.num_frames) == expected_video_shape

    def test_invalid_structure_with_repeated_axis(self):

        self.frame_axis = 3
        reg_expression = (
            "^Invalid structure: (.*?) each property axis should be unique value between 0 and 3 (inclusive)?"
        )
        with self.assertRaisesRegex(ValueError, reg_expression):
            video_structure = VideoStructure(
                rows=self.rows,
                columns=self.columns,
                num_channels=self.num_channels,
                rows_axis=self.rows_axis,
                columns_axis=self.columns_axis,
                num_channels_axis=self.num_channels_axis,
                frame_axis=self.frame_axis,
            )

    def test_invalid_structure_with_values_out_of_range(self):
        self.frame_axis = 10
        reg_expression = (
            "^Invalid structure: (.*?) each property axis should be unique value between 0 and 3 (inclusive)?"
        )

        with self.assertRaisesRegex(ValueError, reg_expression):
            video_structure = VideoStructure(
                rows=self.rows,
                columns=self.columns,
                num_channels=self.num_channels,
                rows_axis=self.rows_axis,
                columns_axis=self.columns_axis,
                num_channels_axis=self.num_channels_axis,
                frame_axis=self.frame_axis,
            )


if __name__ == "__main__":
    unittest.main()
