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
    return (
        f"{testcase_func.__name__}_{param_num}_"
        f"_{param.kwargs.get('case_name', '')}"
    )


class TestMemmapExtractor(unittest.TestCase):

    parameterized_list = list()
    dtype_list = ['uint16', 'float', 'int']
    num_channels_list = [1, 3]
    sizes_list = [10, 25]
    for dtype, num_channels, rows, columns in product(dtype_list, num_channels_list, sizes_list, sizes_list):
        param_case = param(
            dtype=dtype,
            num_channels=num_channels,
            rows=rows,
            columns=columns,
            case_name = f"dtype={dtype}, num_channels={num_channels}, rows={rows}, columns={columns}",
        )
        parameterized_list.append(param_case)
    
    @parameterized.expand(input=parameterized_list, name_func=custom_name_func)
    def test_extractor_defaults(self, dtype, num_channels, rows, columns, case_name=""):
        num_frames = 25
        memmap_shape = (num_frames, num_channels, rows, columns)
        frame_shape = (num_channels, rows, columns)
        
        random_video = np.random.randint(low=1, size=memmap_shape).astype(dtype)

        file_path = OUTPUT_PATH / f"video_{case_name}.dat"
        file = np.memmap(file_path, dtype=dtype, mode="w+", shape=memmap_shape)
        file[:] = random_video[:]
        
        # Write the file and delete it
        file.flush()
        del file

        # Load extractor
        extractor = MemmapImagingExtractor(
            file_path=file_path, frame_shape=frame_shape, dtype=dtype
        )
        
        # Assertions
        extractor.get_num_channels(), extractor.get_image_size(), extractor.get_num_frames()
        self.assertEqual(extractor.get_num_channels(), num_channels)
        self.assertEqual(extractor.get_image_size(), (rows, columns))
        self.assertEqual(extractor.get_num_frames(), num_frames)
        
        # Compare the extracted video
        np.testing.assert_array_almost_equal(random_video, extractor.get_frames())
        
    def test_frames_not_on_first_axis(self):
        
        
if __name__ == "__main__":
    unittest.main()
