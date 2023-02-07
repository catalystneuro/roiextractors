import unittest
from pathlib import Path
from tempfile import mkdtemp
from itertools import product

import numpy as np
from parameterized import parameterized, param

from roiextractors.extraction_tools import VideoStructure, read_numpy_memmap_video


class TestVideoStructureClass(unittest.TestCase):
    def setUp(self) -> None:
        self.num_rows = 10
        self.num_columns = 5
        self.num_channels = 3

        self.frame_axis = 0
        self.rows_axis = 1
        self.columns_axis = 2
        self.channels_axis = 3

        self.num_frames = 20

    def test_basic_parameters(self):
        expected_frame_shape = (self.num_rows, self.num_columns, self.num_channels)
        expected_pixels_per_frame = self.num_rows * self.num_columns * self.num_channels
        expected_video_shape = (self.num_frames, self.num_rows, self.num_columns, self.num_channels)

        video_structure = VideoStructure(
            num_rows=self.num_rows,
            num_columns=self.num_columns,
            num_channels=self.num_channels,
            rows_axis=self.rows_axis,
            columns_axis=self.columns_axis,
            channels_axis=self.channels_axis,
            frame_axis=self.frame_axis,
        )

        assert video_structure.frame_shape == expected_frame_shape
        assert video_structure.number_of_pixels_per_frame == expected_pixels_per_frame
        assert video_structure.build_video_shape(self.num_frames) == expected_video_shape

    def test_axis_permutation_1(self):
        self.frame_axis = 3
        self.rows_axis = 2
        self.columns_axis = 1
        self.channels_axis = 0

        expected_frame_shape = (self.num_channels, self.num_columns, self.num_rows)
        expected_pixels_per_frame = self.num_rows * self.num_columns * self.num_channels
        expected_video_shape = (self.num_channels, self.num_columns, self.num_rows, self.num_frames)

        video_structure = VideoStructure(
            num_rows=self.num_rows,
            num_columns=self.num_columns,
            num_channels=self.num_channels,
            rows_axis=self.rows_axis,
            columns_axis=self.columns_axis,
            channels_axis=self.channels_axis,
            frame_axis=self.frame_axis,
        )

        assert video_structure.frame_shape == expected_frame_shape
        assert video_structure.number_of_pixels_per_frame == expected_pixels_per_frame
        assert video_structure.build_video_shape(self.num_frames) == expected_video_shape

    def test_axis_permutation_2(self):
        self.frame_axis = 2
        self.rows_axis = 0
        self.columns_axis = 1
        self.channels_axis = 3

        expected_frame_shape = (self.num_rows, self.num_columns, self.num_channels)
        expected_pixels_per_frame = self.num_rows * self.num_columns * self.num_channels
        expected_video_shape = (self.num_rows, self.num_columns, self.num_frames, self.num_channels)

        video_structure = VideoStructure(
            num_rows=self.num_rows,
            num_columns=self.num_columns,
            num_channels=self.num_channels,
            rows_axis=self.rows_axis,
            columns_axis=self.columns_axis,
            channels_axis=self.channels_axis,
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
                num_rows=self.num_rows,
                num_columns=self.num_columns,
                num_channels=self.num_channels,
                rows_axis=self.rows_axis,
                columns_axis=self.columns_axis,
                channels_axis=self.channels_axis,
                frame_axis=self.frame_axis,
            )

    def test_invalid_structure_with_values_out_of_range(self):
        self.frame_axis = 10
        reg_expression = (
            "^Invalid structure: (.*?) each property axis should be unique value between 0 and 3 (inclusive)?"
        )

        with self.assertRaisesRegex(ValueError, reg_expression):
            video_structure = VideoStructure(
                num_rows=self.num_rows,
                num_columns=self.num_columns,
                num_channels=self.num_channels,
                rows_axis=self.rows_axis,
                columns_axis=self.columns_axis,
                channels_axis=self.channels_axis,
                frame_axis=self.frame_axis,
            )


def custom_name_func(testcase_func, param_num, param):
    return f"{testcase_func.__name__}_{param_num}_" f"_{param.kwargs.get('case_name', '')}"


class TestReadNumpyMemmapVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.write_directory = Path(mkdtemp())
        # Reproducible random number generation
        cls.rng = np.random.default_rng(12345)

    def setUp(self):
        self.num_frames = 20
        self.offset = 0

    parameterized_list = list()
    dtype_list = ["uint16", "float", "int"]
    num_channels_list = [1, 3]
    sizes_list = [10, 25]
    for dtype, num_channels, num_rows, num_columns in product(dtype_list, num_channels_list, sizes_list, sizes_list):
        param_case = param(
            dtype=dtype,
            num_channels=num_channels,
            num_rows=num_rows,
            num_columns=num_columns,
            case_name=f"dtype={dtype}, num_channels={num_channels}, num_rows={num_rows}, num_columns={num_columns}",
        )
        parameterized_list.append(param_case)

    @parameterized.expand(input=parameterized_list, name_func=custom_name_func)
    def test_roundtrip(self, dtype, num_channels, num_rows, num_columns, case_name=""):
        permutation = self.rng.choice([0, 1, 2, 3], size=4, replace=False)
        rows_axis, columns_axis, channels_axis, frame_axis = permutation
        # Build a video structure
        video_structure = VideoStructure(
            num_rows=num_rows,
            num_columns=num_columns,
            num_channels=num_channels,
            rows_axis=rows_axis,
            columns_axis=columns_axis,
            channels_axis=channels_axis,
            frame_axis=frame_axis,
        )

        # Build a random video
        memmap_shape = video_structure.build_video_shape(self.num_frames)
        random_video = np.random.randint(low=0, high=256, size=memmap_shape).astype(dtype)

        # Save it to memory
        file_path = self.write_directory / f"video_{case_name}.dat"
        file = np.memmap(file_path, dtype=dtype, mode="w+", shape=memmap_shape)
        file[:] = random_video[:]
        file.flush()
        del file

        # Load extractor and test-it
        memmap_video = read_numpy_memmap_video(
            file_path=file_path, video_structure=video_structure, dtype=dtype, offset=self.offset
        )
        # Compare the extracted video
        np.testing.assert_array_almost_equal(random_video, memmap_video)

        # Test canonical form
        canonical_video = video_structure.transform_video_to_canonical_form(memmap_video)
        assert canonical_video.shape == (self.num_frames, num_rows, num_columns, num_channels)


if __name__ == "__main__":
    unittest.main()
