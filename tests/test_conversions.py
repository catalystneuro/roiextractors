import unittest

from datalad.api import install

from roiextractors import NwbSegmentationExtractor, CaimanSegmentationExtractor


class TestNwbConversions(unittest.TestCase):

    def setUp(self):
        self.dataset = install('https://gin.g-node.org/CatalystNeuro/ophys_testing_data')

    def test_convert_caimian(self):
        resp = self.dataset.get('segmentation_datasets/caiman/caiman_analysis.hdf5')
        path = resp[0]['path']
        seg_ex = CaimanSegmentationExtractor(path)

        NwbSegmentationExtractor.write_segmentation(seg_ex, 'caiman_test.nwb')


if __name__ == '__main__':
    unittest.main()
