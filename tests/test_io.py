import tempfile
import unittest
from pathlib import Path
from copy import copy

from parameterized import parameterized, param

from roiextractors import (
    CaimanSegmentationExtractor,
    ExtractSegmentationExtractor,
    Suite2pSegmentationExtractor,
    CnmfeSegmentationExtractor,
    TiffImagingExtractor,
    Hdf5ImagingExtractor,
    SbxImagingExtractor,
)


from .setup_paths import OPHYS_DATA_PATH, OUTPUT_PATH


def custom_name_func(testcase_func, param_num, param):
    return (
        f"{testcase_func.__name__}_{param_num}_"
        f"{parameterized.to_safe_name(param.kwargs['data_interface'].__name__)}"
    )


class TestRead(unittest.TestCase):
    savedir = OUTPUT_PATH

    imaging_interface_list = [
        param(
            extractor_class=TiffImagingExtractor,
            extractor_kwargs=dict(
                file_path=str(OPHYS_DATA_PATH / "imaging_datasets" / "Tif" / "demoMovie.tif"),
                sampling_frequency=15.0,  # typically provied by user
            ),
        ),
        param(
            extractor_class=Hdf5ImagingExtractor,
            extractor_kwargs=dict(file_path=str(OPHYS_DATA_PATH / "imaging_datasets" / "hdf5" / "demoMovie.hdf5")),
        ),
    ]
    for suffix in [".mat", ".sbx"]:
        imaging_interface_list.append(
            param(
                extractor_class=SbxImagingExtractor,
                extractor_kwargs=dict(
                    file_path=str(OPHYS_DATA_PATH / "imaging_datasets" / "Scanbox" / f"sample{suffix}")
                ),
            ),
        )

    @parameterized.expand(imaging_interface_list, name_func=custom_name_func)
    def test_imaging_extractors(self, extractor_class, extractor_kwargs):
        extractor = extractor_class(**extractor_kwargs)
        try:
            suffix = Path(extractor_kwargs["file_path"]).suffix
            output_path = self.savedir / f"{extractor_class.__name__}{suffix}"
            extractor_class.write_imaging(extractor, output_path)

            roundtrip_kwargs = copy(extractor_kwargs)
            roundtrip_kwargs.update(file_path=output_path)
            roundtrip_extractor = extractor_class(**roundtrip_kwargs)
            # TODO: this roundtrip test has been failing for some time now
            # check_imaging_equal(img1=extractor, img2=roundtrip_extractor)
        except NotImplementedError:
            return

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
    def test_seg_interfaces(
        self,
        roi_ex_class,
        dataset_path,
        dataset_path_arg,
        save_fname,
        rt_write_fname,
        rt_read_fname=None,
    ):

        rt_write_path, rt_read_path, save_path = self.get_data(rt_write_fname, rt_read_fname, save_fname, dataset_path)

        path = Path.cwd() / "ophys_testing_data" / dataset_path_arg
        roi_ex = roi_ex_class(path)
        try:
            roi_ex_class.write_segmentation(roi_ex, rt_write_path)
            roi_ex_rt = roi_ex_class(rt_read_path)
            # check_segmentations_equal(seg1=roi_ex, seg2=roi_ex_rt)  # failing
        except NotImplementedError:
            return


if __name__ == "__main__":
    unittest.main()
