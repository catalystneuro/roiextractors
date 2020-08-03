import h5py
from numpy.testing import assert_array_equal
import numpy as np
import os
import sys
import unittest
sys.path.append(os.getcwd())
import segmentationextractors


class TestSchnitzer(unittest.TestCase):

    working_dir = os.getcwd()
    fileloc = working_dir + r'\tests\testdatasets'

    def _setup(self, filelocation):
        try:
            self.f = h5py.File(filelocation, 'r')
            group0_temp = list(self.f.keys())
            self.group0 = [a for a in group0_temp if '#' not in a]
        except OSError:
            raise Exception('could not open .mat file')

    def _teardown(self, obj):
        self.f.close()
        del obj

    def test_extract(self):
        inp_str = self.fileloc + r'\2014_04_01_p203_m19_check01_extractAnalysis.mat'
        try:
            seg_obj = segmentationextractors.ExtractSegmentationExtractor(inp_str)
        except OSError:
            raise Exception('Could not create extract segmentation object')

        self._setup(inp_str)
        raw_images_trans = np.array(self.f[self.group0[0]]['filters']).transpose()
        raw_traces = np.array(self.f[self.group0[0]]['traces']).T
        # equality of ROI data:
        assert_array_equal(raw_traces.shape[1], seg_obj.get_num_frames())
        assert_array_equal(raw_images_trans.shape[0:2], seg_obj.get_movie_framesize())
        assert_array_equal(raw_traces.shape[0], seg_obj.get_num_rois())
        assert_array_equal(raw_traces[1:4, :], seg_obj.get_traces(ROI_ids=[1, 2, 3]))
        assert_array_equal(raw_images_trans[:, :, 1],
                           seg_obj.get_image_masks(ROI_ids=[1, 2, 3])[:, :, 0])
        self._teardown(seg_obj)

    def test_cnmfe(self):
        inp_str = self.fileloc + r'\2014_04_01_p203_m19_check01_cnmfeAnalysis.mat'
        try:
            seg_obj = segmentationextractors.CnmfeSegmentationExtractor(inp_str)
        except OSError:
            raise Exception('Could not create cnmfe segmentation object')

        self._setup(inp_str)
        raw_images_trans = np.array(self.f[self.group0[0]]['extractedImages']).transpose()
        raw_traces = np.array(self.f[self.group0[0]]['extractedSignals']).T
        # equality of ROI data:
        assert_array_equal(raw_traces.shape[1], seg_obj.get_num_frames())
        assert_array_equal(raw_images_trans.shape[0:2], seg_obj.get_movie_framesize())
        assert_array_equal(raw_traces.shape[0], seg_obj.get_num_rois())
        assert_array_equal(raw_traces[1:4, :], seg_obj.get_traces(ROI_ids=[1, 2, 3]))
        assert_array_equal(raw_images_trans[:, :, 1],
                           seg_obj.get_image_masks(ROI_ids=[1, 2, 3])[:, :, 0])
        self._teardown(seg_obj)


if __name__ == '__main__':
    unittest.main()
