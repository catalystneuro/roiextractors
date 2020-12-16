import unittest
import tempfile
from datalad.api import install
from pathlib import Path
from roiextractors import NwbSegmentationExtractor, \
    CaimanSegmentationExtractor, ExtractSegmentationExtractor, \
    Suite2pSegmentationExtractor, CnmfeSegmentationExtractor


class TestNwbConversions(unittest.TestCase):

    def setUp(self):
        self.dataset = install('https://gin.g-node.org/CatalystNeuro/ophys_testing_data')
        self.savedir = Path(tempfile.mkdtemp())

    def test_convert_caiman(self):
        resp = self.dataset.get('segmentation_datasets/caiman/caiman_analysis.hdf5')
        path = resp[0]['path']
        seg_ex = CaimanSegmentationExtractor(path)

        NwbSegmentationExtractor.write_segmentation(seg_ex, self.savedir/'caiman_test.nwb')
        #roundtrip:
        nwb_seg_ex = NwbSegmentationExtractor(self.savedir/'caiman_test.nwb')
        CaimanSegmentationExtractor.write_segmentation(nwb_seg_ex, self.savedir/'caiman_test.hdf5')
        seg_ex_rt = CaimanSegmentationExtractor(self.savedir/'caiman_test.hdf5')

    def test_convert_cnmfe(self):
        resp = self.dataset.get('segmentation_datasets/cnmfe/2014_04_01_p203_m19_check01_cnmfeAnalysis.mat')
        path = resp[0]['path']
        seg_ex = CnmfeSegmentationExtractor(path)

        NwbSegmentationExtractor.write_segmentation(seg_ex, self.savedir/'cnmfe_test.nwb')
        # roundtrip:
        nwb_seg_ex = NwbSegmentationExtractor(self.savedir/'cnmfe_test.nwb')
        CnmfeSegmentationExtractor.write_segmentation(nwb_seg_ex, self.savedir/'cnmfe_test.mat')
        seg_ex_rt = CnmfeSegmentationExtractor(self.savedir/'cnmfe_test.mat')

    def test_convert_extract(self):
        resp = self.dataset.get('segmentation_datasets/extract/2014_04_01_p203_m19_check01_extractAnalysis.mat')
        path = resp[0]['path']
        seg_ex = ExtractSegmentationExtractor(path)

        NwbSegmentationExtractor.write_segmentation(seg_ex, self.savedir/'extract_test.nwb')
        # roundtrip:
        nwb_seg_ex = NwbSegmentationExtractor(self.savedir/'extract_test.nwb')
        ExtractSegmentationExtractor.write_segmentation(nwb_seg_ex, self.savedir/'extract_test.mat')
        seg_ex_rt = ExtractSegmentationExtractor(self.savedir/'extract_test.mat')

    def test_convert_suite2p(self):
        resp = self.dataset.get('segmentation_datasets/suite2p')
        path = resp[0]['path']
        seg_ex = Suite2pSegmentationExtractor(path)

        NwbSegmentationExtractor.write_segmentation(seg_ex, self.savedir/'suite2p_test.nwb')
        # roundtrip:
        nwb_seg_ex = NwbSegmentationExtractor(self.savedir/'suite2p_test.nwb')
        Suite2pSegmentationExtractor.write_segmentation(nwb_seg_ex, self.savedir/'suite2p_test/plane0')
        seg_ex_rt = Suite2pSegmentationExtractor(self.savedir/'suite2p_test')

if __name__ == '__main__':
    unittest.main()
