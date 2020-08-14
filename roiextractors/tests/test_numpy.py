
from numpy.testing import assert_array_equal
import numpy as np
import os
import sys
import unittest
import sima
sys.path.append(os.getcwd())
import segmentationextractors


class TestNumpy(unittest.TestCase):

    working_dir = os.getcwd()
    fileloc = working_dir + r'\tests\testdatasets'

    def test_numpy(self):
        self.sima_dataset = sima.ImagingDataset.load(self.fileloc + r'\dataset_hdf5.sima')
        # Making variables:
        _masks = np.moveaxis(np.array([np.squeeze(
            self.sima_dataset.ROIs['auto_ROIs'][i])
            for i in range(len(self.sima_dataset.ROIs['auto_ROIs']))]),
            0, -1)
        _signal = self.sima_dataset.signals(channel='Green')['example_ROI']['raw'][0]
        _roi_idx = np.arange(_signal.shape[0])
        _accepted_list = _roi_idx
        _channel_names = self.sima_dataset.channel_names
        try:
            self.numpyobj = segmentationextractors.NumpySegmentationExtractor(
                filepath=self.fileloc + r'\dataset_hdf5.sima', masks=_masks, signal=_signal,
                roi_idx=_roi_idx, accepted_lst=_accepted_list,
                channel_names=_channel_names)
        except OSError:
            raise Exception('Could not create numpy segmentation object')

        # equality of ROI data:
        assert_array_equal(self.sima_dataset.num_frames, self.numpyobj.get_num_frames())
        assert_array_equal(self.sima_dataset.frame_shape[1:3], self.numpyobj.get_movie_framesize())
        assert_array_equal(len(self.sima_dataset.ROIs['auto_ROIs']), self.numpyobj.get_num_rois())
        assert_array_equal(self.sima_dataset.channel_names, self.numpyobj.get_channel_names())
        assert_array_equal(self.sima_dataset.signals(channel='Green')['example_ROI']['raw'][0][1:4, :],
                           self.numpyobj.get_traces(ROI_ids=[1, 2, 3]))
        assert_array_equal(np.moveaxis(np.array([np.squeeze(self.sima_dataset.ROIs['auto_ROIs'][i])
                                                 for i in range(len(self.sima_dataset.ROIs['auto_ROIs']))])[1:4, :, :], 0, -1),
                           self.numpyobj.get_image_masks(ROI_ids=[1, 2, 3]))
        self._teardown()

    def _teardown(self):
        del self.numpyobj
        del self.sima_dataset


if __name__ == '__main__':
    unittest.main()
