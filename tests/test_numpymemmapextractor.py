import unittest
from pathlib import Path
from tempfile import mkdtemp
from copy import deepcopy
from itertools import product

import numpy as np
from parameterized import parameterized, param

from roiextractors.extraction_tools import VideoStructure
from roiextractors.testing import check_imaging_equal
from roiextractors import NumpyMemmapImagingExtractor


from .setup_paths import OUTPUT_PATH


def custom_name_func(testcase_func, param_num, param):
    return f"{testcase_func.__name__}_{param_num}_" f"_{param.kwargs.get('case_name', '')}"


class TestNumpyMemmapExtractor(unittest.TestCase):
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
    buffer_gb_list = [0.0001, 1]
    for parameters in product(dtype_list, num_channels_list, sizes_list, sizes_list, buffer_gb_list):
        dtype, num_channels, rows, columns, buffer_gb = parameters
        param_case = param(
            dtype=dtype,
            num_channels=num_channels,
            rows=rows,
            columns=columns,
            buffer_gb=buffer_gb,
            case_name=(
                f"dtype={dtype}, num_channels={num_channels}, rows={rows}, columns={columns}, buffer_gb={buffer_gb}"
            ),
        )
        parameterized_list.append(param_case)

    @parameterized.expand(input=parameterized_list, name_func=custom_name_func)
    def test_roundtrip(self, dtype, num_channels, rows, columns, buffer_gb, case_name=""):

        permutation = self.rng.choice([0, 1, 2, 3], size=4, replace=False)
        rows_axis, columns_axis, num_channels_axis, frame_axis = permutation

        # Build a video structure
        video_structure = VideoStructure(
            rows=rows,
            columns=columns,
            num_channels=num_channels,
            rows_axis=rows_axis,
            columns_axis=columns_axis,
            num_channels_axis=num_channels_axis,
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

        # Load the video with the extactor
        extractor = NumpyMemmapImagingExtractor(
            file_path=file_path, video_structure=video_structure, sampling_frequency=1, dtype=dtype, offset=self.offset
        )

        # Use the write method and do a round-trip
        write_path = self.write_directory / f"video_output_{case_name}.dat"
        extractor.write_imaging(extractor, write_path, buffer_gb=buffer_gb)
        roundtrip_extractor = NumpyMemmapImagingExtractor(
            file_path=write_path,
            video_structure=video_structure,
            sampling_frequency=1,
            dtype=dtype,
            offset=self.offset,
        )
        check_imaging_equal(imaging_extractor1=extractor, imaging_extractor2=roundtrip_extractor)


if __name__ == "__main__":
    unittest.main()
