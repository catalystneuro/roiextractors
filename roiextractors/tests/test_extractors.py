import h5py
import numpy as np
import os
import sys
import unittest
from ..example_datasets.toy_example import toy_example


class TestSegmentationExtractors(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def _create_example(self, seed):
        size_x = np.random.RandomState(seed=seed).randint(100, 1000)
        size_y = np.random.RandomState(seed=seed).randint(100, 1000)
        roi_size = np.random.RandomState(seed=seed).randint(4, 10)
        num_rois = np.random.RandomState(seed=seed).randint(10, 50)
        duration = np.random.RandomState(seed=seed).randint(100, 5000)
        sampling_frequency = np.float(np.random.RandomState(seed=seed).randint(100, 5000))
        self.numpy_imag, self.numpy_seg = toy_example(num_rois=num_rois, size_x=size_x, size_y=size_y,
                                                      roi_size=roi_size, duration=duration,
                                                      sampling_frequency=sampling_frequency)

    def test_example(self):
        pass

    def test_nwb_segmentationextractor(self):
        pass

    def test_cnmfe_segmentationextractor(self):
        pass

    def test_extract_segmentationextractor(self):
        pass

    def test_suite2p_segmentationextractor(self):
        pass

    def test_sima_segmentationextractor(self):
        pass

    def test_caiman_segmentationextractor(self):
        pass
