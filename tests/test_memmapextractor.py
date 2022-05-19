import unittest
from pathlib import Path
from tempfile import mkdtemp
from copy import deepcopy
from itertools import product

import numpy as np
from parameterized import parameterized, param

from roiextractors import MemmapImagingExtractor

from .setup_paths import OUTPUT_PATH


def custom_name_func(testcase_func, param_num, param):
    return f"{testcase_func.__name__}_{param_num}_" f"_{param.kwargs.get('case_name', '')}"


class TestMemmapExtractor(unittest.TestCase):

    parameterized_list = list()
    dtype_list = ["uint16", "float", "int"]
    num_channels_list = [1, 3]
    sizes_list = [10, 25]
    for dtype, num_channels, rows, columns in product(dtype_list, num_channels_list, sizes_list, sizes_list):
        param_case = param(
            dtype=dtype,
            num_channels=num_channels,
            rows=rows,
            columns=columns,
            case_name=f"dtype={dtype}, num_channels={num_channels}, rows={rows}, columns={columns}",
        )
        parameterized_list.append(param_case)

    @parameterized.expand(input=parameterized_list, name_func=custom_name_func)
    def test_extractor_defaults(self, dtype, num_channels, rows, columns, case_name=""):
        # Build a video
        num_frames = 25
        sampling_frequency = 30
        memmap_shape = (num_frames, num_channels, rows, columns)
        random_video = np.random.randint(low=1, size=memmap_shape).astype(dtype)

        # Save it to memory
        file_path = OUTPUT_PATH / f"video_{case_name}.dat"
        file = np.memmap(file_path, dtype=dtype, mode="w+", shape=memmap_shape)
        file[:] = random_video[:]
        file.flush()
        del file

        # Load extractor and test-it
        frame_shape = (num_channels, rows, columns)
        extractor = MemmapImagingExtractor(
            file_path=file_path,
            frame_shape=frame_shape,
            sampling_frequency=sampling_frequency,
            dtype=dtype,
        )

        # Property assertions
        extractor.get_num_channels(), extractor.get_image_size(), extractor.get_num_frames()
        self.assertEqual(extractor.get_num_channels(), num_channels)
        self.assertEqual(extractor.get_image_size(), (rows, columns))
        self.assertEqual(extractor.get_num_frames(), num_frames)

        # Compare the extracted video
        np.testing.assert_array_almost_equal(random_video, extractor.get_frames())

    def test_frames_on_last_axis(self):

        # Build a random video
        num_frames = 50
        sampling_frequency = 30
        num_channels = 3
        rows = 10
        columns = 10
        memmap_shape = (num_channels, rows, columns, num_frames)
        dtype = "uint16"
        random_video = np.random.randint(low=1, size=memmap_shape).astype(dtype)

        # Save it to memory
        file_path = Path(mkdtemp()) / "random_video_last.dat"
        file = np.memmap(file_path, dtype=dtype, mode="w+", shape=memmap_shape)
        file[:] = random_video[:]
        file.flush()
        del file

        # Call the extractor and test it
        frame_axis = 3
        frame_shape = (num_channels, rows, columns)
        image_structure_to_axis = dict(frame_axis=frame_axis, num_channels=0, rows=1, columns=2)
        extractor = MemmapImagingExtractor(
            file_path=file_path,
            frame_shape=frame_shape,
            sampling_frequency=sampling_frequency,
            dtype=dtype,
            image_structure_to_axis=image_structure_to_axis,
        )

        # Assertions
        extractor.get_num_channels(), extractor.get_image_size(), extractor.get_num_frames()
        self.assertEqual(extractor.get_num_channels(), num_channels)
        self.assertEqual(extractor.get_image_size(), (rows, columns))
        self.assertEqual(extractor.get_num_frames(), num_frames)

        # Compare the extracted video
        np.testing.assert_array_almost_equal(random_video, extractor.get_frames())

    def test_channel_and_frames_inversion(self):

        # Build a random video
        num_frames = 50
        sampling_frequency = 30
        num_channels = 3
        rows = 10
        columns = 10
        memmap_shape = (num_channels, num_frames, rows, columns)
        dtype = "uint16"
        random_video = np.random.randint(low=1, size=memmap_shape).astype(dtype)

        # Save it to memory
        file_path = Path(mkdtemp()) / "random_video_inversion.dat"
        file = np.memmap(file_path, dtype=dtype, mode="w+", shape=memmap_shape)
        file[:] = random_video[:]
        file.flush()
        del file

        # Call the extractor and test it
        frame_shape = (num_channels, rows, columns)
        frame_axis = 1
        image_structure_to_axis = dict(frame_axis=frame_axis, num_channels=0, rows=2, columns=3)
        extractor = MemmapImagingExtractor(
            file_path=file_path,
            frame_shape=frame_shape,
            sampling_frequency=sampling_frequency,
            dtype=dtype,
            image_structure_to_axis=image_structure_to_axis,
        )

        # Assertions
        extractor.get_num_channels(), extractor.get_image_size(), extractor.get_num_frames()
        self.assertEqual(extractor.get_num_channels(), num_channels)
        self.assertEqual(extractor.get_image_size(), (rows, columns))
        self.assertEqual(extractor.get_num_frames(), num_frames)

        # Compare the extracted video
        np.testing.assert_array_almost_equal(random_video, extractor.get_frames())


if __name__ == "__main__":
    unittest.main()
