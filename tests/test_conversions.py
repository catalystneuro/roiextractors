import tempfile
import unittest
from pathlib import Path

from datalad.api import install
from parameterized import parameterized

from roiextractors import NwbSegmentationExtractor, \
    CaimanSegmentationExtractor, ExtractSegmentationExtractor, \
    Suite2pSegmentationExtractor, CnmfeSegmentationExtractor
from roiextractors.testing import check_segmentations_equal


class TestNwbConversions(unittest.TestCase):

    def setUp(self):
        self.dataset = install('git@gin.g-node.org:/CatalystNeuro/ophys_testing_data.git')
        self.savedir = Path(tempfile.mkdtemp())

    @parameterized.expand([
        (
                CaimanSegmentationExtractor,
                'segmentation_datasets/caiman/caiman_analysis.hdf5',
                'caiman_test.nwb',
                'caiman_test.hdf5'
        ), (
                CnmfeSegmentationExtractor,
                'segmentation_datasets/cnmfe/2014_04_01_p203_m19_check01_cnmfeAnalysis.mat',
                'cnmfe_test.nwb',
                'cnmfe_test.mat'
        ), (
                ExtractSegmentationExtractor,
                'segmentation_datasets/extract/2014_04_01_p203_m19_check01_extractAnalysis.mat',
                'extract_test.nwb',
                'extract_test.mat'
        ), (
                Suite2pSegmentationExtractor,
                'segmentation_datasets/suite2p',
                'suite2p_test.nwb',
                'suite2p_test/plane0',
                'suite2p_test'
        )
    ])
    def test_convert_seg_interface_to_nwb(self, seg_ex_class, dataset_path, save_fname, rt_write_fname,
                                          rt_read_fname=None):
        if rt_read_fname is None:
            rt_read_fname = rt_write_fname
        save_path = self.savedir/save_fname
        rt__write_path = self.savedir/rt_write_fname
        rt_read_path = self.savedir/rt_read_fname
        resp = self.dataset.get(dataset_path)
        path = resp[0]['path']
        seg_ex = seg_ex_class(path)
        NwbSegmentationExtractor.write_segmentation(seg_ex, save_path)
        nwb_seg_ex = NwbSegmentationExtractor(save_path)
        check_segmentations_equal(seg_ex, nwb_seg_ex)
        # round trip:
        seg_ex_class.write_segmentation(nwb_seg_ex, rt__write_path)
        seg_ex_rt = seg_ex_class(rt_read_path)


if __name__ == '__main__':
    unittest.main()
