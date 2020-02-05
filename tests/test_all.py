
from numpy.testing import assert_array_equal
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
import segmentationextractors


class TestSima():

    working_dir = os.getcwd()
    fileloc = working_dir + r'\tests\testdatasets'

    def setup(self, filelocation):
        import sima
        self.sima_dataset = sima.ImagingDataset.load(filelocation)
        try:
            self.simaobj = segmentationextractors.SimaSegmentationExtractor(filelocation)
        except:
            Exception('Could not create sima segmentation object')

        # equality of ROI data:
        assert_array_equal(self.sima_dataset.num_frames, self.simaobj.get_num_frames())
        assert_array_equal(self.sima_dataset.frame_shape[1:3], self.simaobj.get_movie_framesize())
        assert_array_equal(len(self.sima_dataset.ROIs['auto_ROIs']), self.simaobj.get_num_rois())
        assert_array_equal(self.sima_dataset.channel_names, self.simaobj.get_channel_names())

        assert_array_equal(self.sima_dataset.signals(channel='Blue')['example_ROI']['raw'][0],
                           self.simaobj.get_traces())

        assert_array_equal(np.squeeze(self.sima_dataset.ROIs['auto_ROIs'][0]),
                           self.simaobj.get_image_masks()[:, :, 0])

    def teardown(self):
        del self.simaobj
        del self.sima_dataset

    def test_sima_hdf5(self):
        self.setup(self.fileloc + r'\dataset_hdf5.sima')

    def test_sima_tiff(self):
        self.setup(self.fileloc + r'\dataset_tiff.sima')


class TestSchnitzer():

    working_dir = os.getcwd()
    fileloc = working_dir + r'\tests\testdatasets'

    def setup(self, filelocation, str1, str2, seg_obj):
        import h5py
        try:
            self.f = h5py.File(filelocation, 'r')
            group0_temp = list(self.f.keys())
            self.group0 = [a for a in group0_temp if '#' not in a]
        except:
            Exception('could not open .mat file')
        raw_images_trans = np.array(self.f[self.group0[0]]['str1']).transpose()
        raw_traces = np.array(self.f[self.group0[0]]['str2']).T
        # equality of ROI data:
        assert_array_equal(raw_traces.shape[1], seg_obj.get_num_frames())
        assert_array_equal(raw_images_trans.shape[0:2], seg_obj.get_movie_framesize())
        assert_array_equal(raw_traces.shape[0], seg_obj.get_num_rois())
        assert_array_equal(raw_traces, seg_obj.get_traces())
        assert_array_equal(raw_images_trans[:, :, 0],
                           seg_obj.get_image_masks()[:, :, 0])

    def teardown(self):
        self.f.close()
        del self.segobj

    def test_extract(self):
        inp_str = self.fileloc + r'\2014_04_01_p203_m19_check01_extractAnalysis.mat'
        try:
            self.segobj = segmentationextractors.ExtractSegmentationExtractor(inp_str)
        except:
            Exception('Could not create extract segmentation object')

        self.setup(inp_str, 'filters', 'traces', self.segobj)
        self.teardown()

    def test_cnmfe(self):
        inp_str = self.fileloc + r'\2014_04_01_p203_m19_check01_cnmfeAnalysis.mat'
        try:
            self.segobj = segmentationextractors.CnmfeSegmentationExtractor(inp_str)
        except:
            Exception('Could not create cnmfe segmentation object')

        self.setup(inp_str, 'extractedImages', 'extractedSignals', self.segobj)
        self.teardown()
