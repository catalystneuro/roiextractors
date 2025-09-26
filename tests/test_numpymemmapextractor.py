import unittest
from itertools import product
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from parameterized import param


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
    buffer_gb_list = [None, 0.1]
    for parameters in product(dtype_list, num_channels_list, sizes_list, sizes_list, buffer_gb_list):
        dtype, num_channels, num_rows, num_columns, buffer_size_in_gb = parameters
        case_name = (
            f"dtype={dtype}, num_channels={num_channels}, num_rows={num_rows},"
            f"num_columns={num_columns}, buffer_size_in_gb={buffer_size_in_gb}"
        )
        param_case = param(
            dtype=dtype,
            num_channels=num_channels,
            num_rows=num_rows,
            num_columns=num_columns,
            buffer_size_in_gb=buffer_size_in_gb,
            case_name=case_name,
        )
        parameterized_list.append(param_case)


if __name__ == "__main__":
    unittest.main()
