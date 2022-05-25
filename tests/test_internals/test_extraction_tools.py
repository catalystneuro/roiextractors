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

        structure = VideoStructure(
            rows=self.rows,
            columns=self.columns,
            num_channels=self.num_channels,
            rows_axis=self.rows_axis,
            columns_axis=self.columns_axis,
            num_channels_axis=self.num_channels_axis,
            frame_axis=self.frame_axis,
        )

        assert structure.frame_shape == expected_frame_shape
        assert structure.number_of_pixels_per_frame == expected_pixels_per_frame
        assert structure.build_video_shape(self.num_frames) == expected_video_shape

    def test_axis_permutation_1(self):

        self.frame_axis = 3
        self.rows_axis = 2
        self.columns_axis = 1
        self.num_channels_axis = 0

        expected_frame_shape = (self.num_channels, self.columns, self.rows)
        expected_pixels_per_frame = self.rows * self.columns * self.num_channels
        expected_video_shape = (self.num_channels, self.columns, self.rows, self.num_frames)

        structure = VideoStructure(
            rows=self.rows,
            columns=self.columns,
            num_channels=self.num_channels,
            rows_axis=self.rows_axis,
            columns_axis=self.columns_axis,
            num_channels_axis=self.num_channels_axis,
            frame_axis=self.frame_axis,
        )

        assert structure.frame_shape == expected_frame_shape
        assert structure.number_of_pixels_per_frame == expected_pixels_per_frame
        assert structure.build_video_shape(self.num_frames) == expected_video_shape

    def test_axis_permutation_2(self):

        self.frame_axis = 2
        self.rows_axis = 0
        self.columns_axis = 1
        self.num_channels_axis = 3

        expected_frame_shape = (self.rows, self.columns, self.num_channels)
        expected_pixels_per_frame = self.rows * self.columns * self.num_channels
        expected_video_shape = (self.rows, self.columns, self.num_frames, self.num_channels)

        structure = VideoStructure(
            rows=self.rows,
            columns=self.columns,
            num_channels=self.num_channels,
            rows_axis=self.rows_axis,
            columns_axis=self.columns_axis,
            num_channels_axis=self.num_channels_axis,
            frame_axis=self.frame_axis,
        )

        assert structure.frame_shape == expected_frame_shape
        assert structure.number_of_pixels_per_frame == expected_pixels_per_frame
        assert structure.build_video_shape(self.num_frames) == expected_video_shape

    def test_invalid_structure_with_repeated_axis(self):

        self.frame_axis = 3
        reg_expression = (
            "^Invalid structure: (.*?) each property axis should be unique value between 0 and 3 (inclusive)?"
        )
        with self.assertRaisesRegex(ValueError, reg_expression):
            structure = VideoStructure(
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
            structure = VideoStructure(
                rows=self.rows,
                columns=self.columns,
                num_channels=self.num_channels,
                rows_axis=self.rows_axis,
                columns_axis=self.columns_axis,
                num_channels_axis=self.num_channels_axis,
                frame_axis=self.frame_axis,
            )


# class TestMemmapExtractor(unittest.TestCase):

#     parameterized_list = list()
#     dtype_list = ["uint16", "float", "int"]
#     num_channels_list = [1, 3]
#     sizes_list = [10, 25]
#     for dtype, num_channels, rows, columns in product(dtype_list, num_channels_list, sizes_list, sizes_list):
#         param_case = param(
#             dtype=dtype,
#             num_channels=num_channels,
#             rows=rows,
#             columns=columns,
#             case_name=f"dtype={dtype}, num_channels={num_channels}, rows={rows}, columns={columns}",
#         )
#         parameterized_list.append(param_case)

#     @parameterized.expand(input=parameterized_list, name_func=custom_name_func)
#     def test_extractor_defaults(self, dtype, num_channels, rows, columns, case_name=""):
#         # Build a video
#         num_frames = 25
#         sampling_frequency = 30
#         memmap_shape = (num_frames, num_channels, rows, columns)
#         random_video = np.random.randint(low=1, size=memmap_shape).astype(dtype)

#         # Save it to memory
#         file_path = OUTPUT_PATH / f"video_{case_name}.dat"
#         file = np.memmap(file_path, dtype=dtype, mode="w+", shape=memmap_shape)
#         file[:] = random_video[:]
#         file.flush()
#         del file

#         # Load extractor and test-it
#         frame_shape = (num_channels, rows, columns)
#         extractor = NumpyMemmapImagingExtractor(
#             file_path=file_path,
#             frame_shape=frame_shape,
#             sampling_frequency=sampling_frequency,
#             dtype=dtype,
#         )

#         # Property assertions
#         extractor.get_num_channels(), extractor.get_image_size(), extractor.get_num_frames()
#         self.assertEqual(extractor.get_num_channels(), num_channels)
#         self.assertEqual(extractor.get_image_size(), (rows, columns))
#         self.assertEqual(extractor.get_num_frames(), num_frames)

#         # Compare the extracted video
#         np.testing.assert_array_almost_equal(random_video, extractor.get_frames())

#     def test_frames_on_last_axis(self):

#         # Build a random video
#         num_frames = 50
#         sampling_frequency = 30
#         num_channels = 3
#         rows = 10
#         columns = 10
#         memmap_shape = (num_channels, rows, columns, num_frames)
#         dtype = "uint16"
#         random_video = np.random.randint(low=1, size=memmap_shape).astype(dtype)

#         # Save it to memory
#         file_path = Path(mkdtemp()) / "random_video_last.dat"
#         file = np.memmap(file_path, dtype=dtype, mode="w+", shape=memmap_shape)
#         file[:] = random_video[:]
#         file.flush()
#         del file

#         # Call the extractor and test it
#         frame_axis = 3
#         frame_shape = (num_channels, rows, columns)
#         image_structure_to_axis = dict(frame_axis=frame_axis, num_channels=0, rows=1, columns=2)
#         extractor = NumpyMemmapImagingExtractor(
#             file_path=file_path,
#             frame_shape=frame_shape,
#             sampling_frequency=sampling_frequency,
#             dtype=dtype,
#             image_structure_to_axis=image_structure_to_axis,
#         )

#         # Assertions
#         extractor.get_num_channels(), extractor.get_image_size(), extractor.get_num_frames()
#         self.assertEqual(extractor.get_num_channels(), num_channels)
#         self.assertEqual(extractor.get_image_size(), (rows, columns))
#         self.assertEqual(extractor.get_num_frames(), num_frames)

#         # Compare the extracted video
#         np.testing.assert_array_almost_equal(random_video, extractor.get_frames())

#     def test_channel_and_frames_inversion(self):

#         # Build a random video
#         num_frames = 50
#         sampling_frequency = 30
#         num_channels = 3
#         rows = 10
#         columns = 10
#         memmap_shape = (num_channels, num_frames, rows, columns)
#         dtype = "uint16"
#         random_video = np.random.randint(low=1, size=memmap_shape).astype(dtype)

#         # Save it to memory
#         file_path = Path(mkdtemp()) / "random_video_inversion.dat"
#         file = np.memmap(file_path, dtype=dtype, mode="w+", shape=memmap_shape)
#         file[:] = random_video[:]
#         file.flush()
#         del file

#         # Call the extractor and test it
#         frame_shape = (num_channels, rows, columns)
#         frame_axis = 1
#         image_structure_to_axis = dict(frame_axis=frame_axis, num_channels=0, rows=2, columns=3)
#         extractor = NumpyMemmapImagingExtractor(
#             file_path=file_path,
#             frame_shape=frame_shape,
#             sampling_frequency=sampling_frequency,
#             dtype=dtype,
#             image_structure_to_axis=image_structure_to_axis,
#         )

#         # Assertions
#         extractor.get_num_channels(), extractor.get_image_size(), extractor.get_num_frames()
#         self.assertEqual(extractor.get_num_channels(), num_channels)
#         self.assertEqual(extractor.get_image_size(), (rows, columns))
#         self.assertEqual(extractor.get_num_frames(), num_frames)

#         # Compare the extracted video
#         np.testing.assert_array_almost_equal(random_video, extractor.get_frames())


if __name__ == "__main__":
    unittest.main()
