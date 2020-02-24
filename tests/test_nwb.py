
from numpy.testing import assert_array_equal
import numpy as np
import os
import sys
import unittest
import sima
sys.path.append(os.getcwd())
import segmentationextractors


class TestNwb(unittest.TestCase):

    def _setup(self):
        self.working_dir = os.getcwd()
        self.fileloc = self.working_dir + r'\tests\testdatasets'
        self.saveloc = self.fileloc + r'\nwbdataset.nwb'
        self.sima_obj_loc = self.fileloc + r'\dataset_tiff.sima'

    def test_nwb_writer(self):
        self._setup()
        if os.path.exists(self.saveloc):
            try:
                self.nwbobj = segmentationextractors.NwbSegmentationExtractor(self.saveloc)
                self.nwbobj.io.close()
                os.remove(self.saveloc)
            except:
                os.remove(self.saveloc)
        try:
            self.simaobj = segmentationextractors.SimaSegmentationExtractor(self.sima_obj_loc)
        except OSError:
            raise Exception('Could not create sima segmentation object')
        segmentationextractors.NwbSegmentationExtractor.write_recording(
            self.simaobj, self.saveloc,
            propertydict=[{'name': 'testvals1',
                           'description': 'testdesc',
                           'data': np.arange(3),
                           'id': self.simaobj.get_roi_ids()[0:3]},
                          {'name': 'testvals2',
                           'description': 'testdesc2',
                           'data': 3 * np.arange(3),
                           'id': self.simaobj.get_roi_ids()[3:6]}],
            nwbfile_kwargs={'session_description': 'nwbfiledesc',
                            'experimenter': 'experimenter name',
                            'lab': 'test lab',
                            'session_id': 'test sess id'},
            emission_lambda=400.0, excitation_lambda=500.0)

    def test_nwb_segmentation(self):
        self._setup()
        self.sima_dataset = sima.ImagingDataset.load(self.sima_obj_loc)
        if os.path.exists(self.saveloc):
            self.nwbobj = segmentationextractors.NwbSegmentationExtractor(self.saveloc)
        else:
            print('file does not exist, creating and running')
            self.test_nwb_writer()
            self.nwbobj = segmentationextractors.NwbSegmentationExtractor(self.saveloc)
        # equality of ROI data:
        assert_array_equal(self.sima_dataset.num_frames, self.nwbobj.get_num_frames())
        assert_array_equal(self.sima_dataset.frame_shape[1:3], self.nwbobj.get_movie_framesize())
        assert_array_equal(len(self.sima_dataset.ROIs['auto_ROIs']), self.nwbobj.get_num_rois())
        assert_array_equal(self.sima_dataset.channel_names, self.nwbobj.get_channel_names())
        assert_array_equal(self.sima_dataset.signals(channel='Green')['example_ROI']['raw'][0][1:4, :],
                           self.nwbobj.get_traces(ROI_ids=[-1, -2, -3]))
        assert_array_equal(np.moveaxis(np.array([np.squeeze(self.sima_dataset.ROIs['auto_ROIs'][i])
                                                 for i in range(len(self.sima_dataset.ROIs['auto_ROIs']))])[1:4, :, :], 0, -1),
                           self.nwbobj.get_image_masks(ROI_ids=[-1, -2, -3]))
        assert_array_equal(np.arange(3), self.nwbobj.get_property_data(
            ['testvals1', 'testvals2'])[0][0:3])
        assert_array_equal(
            3 * np.arange(3), self.nwbobj.get_property_data(['testvals1', 'testvals2'])[1][3:6])
        self.nwbobj.io.close()


if __name__ == '__main__':
    unittest.main()
