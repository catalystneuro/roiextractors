import tempfile
import unittest
from pathlib import Path

from datalad.api import install
from parameterized import parameterized

from roiextractors import NwbSegmentationExtractor, \
    CaimanSegmentationExtractor, ExtractSegmentationExtractor, \
    Suite2pSegmentationExtractor, CnmfeSegmentationExtractor, \
    TiffImagingExtractor, Hdf5ImagingExtractor, SbxImagingExtractor, NwbImagingExtractor
from roiextractors.testing import check_segmentations_equal, check_imaging_equal


class TestNwbConversions(unittest.TestCase):

    def setUp(self):
        self.dataset = install('git@gin.g-node.org:/CatalystNeuro/ophys_testing_data.git')
        self.savedir = Path(tempfile.mkdtemp())

    @parameterized.expand([
        (
                TiffImagingExtractor,
                'imaging_datasets/Tif/demoMovie.tif',
                'tiff_imaging_test.nwb',
                'tiff_imaging_test.tif',
                'suite2p_test'
        ),(
                Hdf5ImagingExtractor,
                'imaging_datasets/hdf5/demoMovie.hdf5',
                'hdf5_imaging_test.nwb',
                'hdf5_imaging_test.hdf5',
        ),(
                SbxImagingExtractor,
                'imaging_datasets/Scanbox/TwoTower_foraging_003_006_small.sbx',
                'sbx_imaging_test.nwb',
                'sbx_imaging_test.sbx',
        )
    ])
    def test_convert_seg_interface_to_nwb(self, roi_ex_class, dataset_path, save_fname, rt_write_fname,
                                          rt_read_fname=None):
        if rt_read_fname is None:
            rt_read_fname = rt_write_fname
        save_path = self.savedir/save_fname
        rt__write_path = self.savedir/rt_write_fname
        rt_read_path = self.savedir/rt_read_fname
        resp = self.dataset.get(dataset_path)
        path = resp[0]['path']
        if 'Segmentation' in roi_ex_class.__name__:
            roi_ex = roi_ex_class(path)
            NwbSegmentationExtractor.write_segmentation(roi_ex, save_path)
            nwb_seg_ex = NwbSegmentationExtractor(save_path)
            check_segmentations_equal(roi_ex, nwb_seg_ex)
            try:
                roi_ex_class.write_segmentation(nwb_seg_ex, rt__write_path)
            except NotImplementedError:
                return
            seg_ex_rt = roi_ex_class(rt_read_path)
        else:
            if not 'Sbx' in roi_ex_class.__name__:
                roi_ex = roi_ex_class(path, sampling_frequency=20.0)
            else:
                roi_ex = roi_ex_class(path)
            NwbImagingExtractor.write_imaging(roi_ex, save_path)
            nwb_img_ex = NwbImagingExtractor(save_path)
            check_imaging_equal(roi_ex, nwb_img_ex)
            try:
                roi_ex_class.write_imaging(nwb_img_ex, rt__write_path)
            except NotImplementedError:
                return
            img_ex_rt = roi_ex_class(rt_read_path)



if __name__ == '__main__':
    unittest.main()
