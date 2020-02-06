
from numpy.testing import assert_array_equal
import numpy as np
import os
import sys
import unittest
import sima
sys.path.append(os.getcwd())
import segmentationextractors


class TestSima(unittest.TestCase):

    working_dir = os.getcwd()
    fileloc = working_dir + r'\tests\testdatasets'

    def _setup(self, filelocation):
        self.sima_dataset = sima.ImagingDataset.load(filelocation)
        try:
            self.simaobj = segmentationextractors.SimaSegmentationExtractor(filelocation)
        except OSError:
            raise Exception('Could not create sima segmentation object')

        # equality of ROI data:
        assert_array_equal(self.sima_dataset.num_frames, self.simaobj.get_num_frames())
        assert_array_equal(self.sima_dataset.frame_shape[1:3], self.simaobj.get_movie_framesize())
        assert_array_equal(len(self.sima_dataset.ROIs['auto_ROIs']), self.simaobj.get_num_rois())
        assert_array_equal(self.sima_dataset.channel_names, self.simaobj.get_channel_names())
        assert_array_equal(self.sima_dataset.signals(channel='Blue')['example_ROI']['raw'][0][1:4, :],
                           self.simaobj.get_traces(ROI_ids=[-1, -2, -3]))
        assert_array_equal(np.moveaxis(np.array([np.squeeze(self.sima_dataset.ROIs['auto_ROIs'][i])
                           for i in range(len(self.sima_dataset.ROIs['auto_ROIs']))])[1:4, :, :], 0, -1),
                           self.simaobj.get_image_masks(ROI_ids=[-1, -2, -3]))

    def _teardown(self):
        del self.simaobj
        del self.sima_dataset

    def test_sima_hdf5(self):
        self._setup(self.fileloc + r'\dataset_hdf5.sima')

    def test_sima_tiff(self):
        self._setup(self.fileloc + r'\dataset_tiff.sima')


if __name__ == '__main__':
    unittest.main()
