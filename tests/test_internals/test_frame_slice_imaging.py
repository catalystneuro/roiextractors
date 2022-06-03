import unittest

import numnpy as np

from roiextractors import NumpyImagingExtractor


class TestFrameSliceImaging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Use a toy example of ten frames of a 5 x 4 grayscale image."""
        cls.toy_example = NumpyImagingExtractor(timeseries=np.random.random(size=[10, 5, 4, 1]), sampling_frequency=1.0)


if __name__ == "__main__":
    unittest.main()
