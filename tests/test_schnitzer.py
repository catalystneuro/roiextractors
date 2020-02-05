
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

    def _setup(self, filelocation, str1, str2, seg_obj):
        try:
            self.f = h5py.File(filelocation, 'r')
            group0_temp = list(self.f.keys())
            self.group0 = [a for a in group0_temp if '#' not in a]
        except:
            Exception('could not open .mat file')
        raw_images_trans = np.array(self.f[self.group0[0]][str1]).transpose()
        raw_traces = np.array(self.f[self.group0[0]][str2]).T
        # equality of ROI data:
        assert_array_equal(raw_traces.shape[1], seg_obj.get_num_frames())
        assert_array_equal(raw_images_trans.shape[0:2], seg_obj.get_movie_framesize())
        assert_array_equal(raw_traces.shape[0], seg_obj.get_num_rois())
        assert_array_equal(raw_traces, seg_obj.get_traces())
        assert_array_equal(raw_images_trans[:, :, 0],
                           seg_obj.get_image_masks()[:, :, 0])

    def _teardown(self):
        self.f.close()
        del self.segobj

    def test_extract(self):
        inp_str = self.fileloc + r'\2014_04_01_p203_m19_check01_extractAnalysis.mat'
        try:
            self.segobj = segmentationextractors.ExtractSegmentationExtractor(inp_str)
        except:
            Exception('Could not create extract segmentation object')

        self._setup(inp_str, 'filters', 'traces', self.segobj)
        self._teardown()

    def test_cnmfe(self):
        inp_str = self.fileloc + r'\2014_04_01_p203_m19_check01_cnmfeAnalysis.mat'
        try:
            self.segobj = segmentationextractors.CnmfeSegmentationExtractor(inp_str)
        except:
            Exception('Could not create cnmfe segmentation object')

        self._setup(inp_str, 'extractedImages', 'extractedSignals', self.segobj)
        self._teardown()


if __name__ == '__main__':
    unittest.main()
