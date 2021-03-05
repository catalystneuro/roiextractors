import tempfile
import unittest
from pathlib import Path

from datalad.api import install, Dataset
from parameterized import parameterized

from roiextractors import (
    NwbSegmentationExtractor,
    CaimanSegmentationExtractor,
    ExtractSegmentationExtractor,
    Suite2pSegmentationExtractor,
    CnmfeSegmentationExtractor,
    TiffImagingExtractor,
    Hdf5ImagingExtractor,
    SbxImagingExtractor,
    NwbImagingExtractor,
)
from roiextractors.testing import check_segmentations_equal, check_imaging_equal


class TestNwbConversions(unittest.TestCase):
    def setUp(self):
        pt = Path.cwd() / "ophys_testing_data"
        if pt.exists():
            self.dataset = Dataset(pt)
        else:
            self.dataset = install(
                "https://gin.g-node.org/CatalystNeuro/ophys_testing_data"
            )
        self.savedir = Path(tempfile.mkdtemp())

    def get_data(self, rt_write_fname, rt_read_fname, save_fname, dataset_path):
        if rt_read_fname is None:
            rt_read_fname = rt_write_fname
        save_path = self.savedir / save_fname
        rt_write_path = self.savedir / rt_write_fname
        rt_read_path = self.savedir / rt_read_fname
        resp = self.dataset.get(dataset_path)

        return rt_write_path, rt_read_path, save_path

    @parameterized.expand(
        [
            (
                CaimanSegmentationExtractor,
                "segmentation_datasets/caiman/caiman_analysis.hdf5",
                "segmentation_datasets/caiman/caiman_analysis.hdf5",
                "caiman_test.nwb",
                "caiman_test.hdf5",
            ),
            (
                CnmfeSegmentationExtractor,
                "segmentation_datasets/cnmfe/2014_04_01_p203_m19_check01_cnmfeAnalysis.mat",
                "segmentation_datasets/cnmfe/2014_04_01_p203_m19_check01_cnmfeAnalysis.mat",
                "cnmfe_test.nwb",
                "cnmfe_test.mat",
            ),
            (
                ExtractSegmentationExtractor,
                "segmentation_datasets/extract/2014_04_01_p203_m19_check01_extractAnalysis.mat",
                "segmentation_datasets/extract/2014_04_01_p203_m19_check01_extractAnalysis.mat",
                "extract_test.nwb",
                "extract_test.mat",
            ),
            (
                Suite2pSegmentationExtractor,
                "segmentation_datasets/suite2p/plane0",
                "segmentation_datasets/suite2p",
                "suite2p_test.nwb",
                "suite2p_test/plane0",
                "suite2p_test",
            ),
        ]
    )
    def test_convert_seg_interface_to_nwb(
        self,
        roi_ex_class,
        dataset_path,
        dataset_path_arg,
        save_fname,
        rt_write_fname,
        rt_read_fname=None,
    ):

        rt_write_path, rt_read_path, save_path = self.get_data(
            rt_write_fname, rt_read_fname, save_fname, dataset_path
        )

        path = Path.cwd() / "ophys_testing_data" / dataset_path_arg
        roi_ex = roi_ex_class(path)
        NwbSegmentationExtractor.write_segmentation(roi_ex, save_path)
        nwb_seg_ex = NwbSegmentationExtractor(save_path)
        check_segmentations_equal(roi_ex, nwb_seg_ex)
        try:
            roi_ex_class.write_segmentation(nwb_seg_ex, rt_write_path)
        except NotImplementedError:
            return
        seg_ex_rt = roi_ex_class(rt_read_path)

    @parameterized.expand(
        [
            (
                TiffImagingExtractor,
                "imaging_datasets/Tif/demoMovie.tif",
                "imaging_datasets/Tif/demoMovie.tif",
                "tiff_imaging_test.nwb",
                "tiff_imaging_test.tif",
            ),
            (
                Hdf5ImagingExtractor,
                "imaging_datasets/hdf5/demoMovie.hdf5",
                "imaging_datasets/hdf5/demoMovie.hdf5",
                "hdf5_imaging_test.nwb",
                "hdf5_imaging_test.hdf5",
            ),
            (
                SbxImagingExtractor,
                "imaging_datasets/Scanbox",
                "imaging_datasets/Scanbox/sample.mat",
                "sbx_imaging_test.nwb",
                "sbx_imaging_test.sbx",
            ),
        ]
    )
    def test_convert_img_interface_to_nwb(
        self,
        roi_ex_class,
        dataset_path,
        dataset_path_arg,
        save_fname,
        rt_write_fname,
        rt_read_fname=None,
    ):

        rt_write_path, rt_read_path, save_path = self.get_data(
            rt_write_fname, rt_read_fname, save_fname, dataset_path
        )

        sampling_freq = 20.0

        path = Path.cwd() / "ophys_testing_data" / dataset_path_arg
        roi_ex = roi_ex_class(file_path=path, sampling_frequency=sampling_freq)
        NwbImagingExtractor.write_imaging(roi_ex, save_path)
        nwb_img_ex = NwbImagingExtractor(save_path)
        check_imaging_equal(roi_ex, nwb_img_ex)
        try:
            roi_ex_class.write_imaging(nwb_img_ex, rt_write_path)
        except NotImplementedError:
            return
        img_ex_rt = roi_ex_class(
            file_path=rt_read_path, sampling_frequency=sampling_freq
        )


if __name__ == "__main__":
    unittest.main()
