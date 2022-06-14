import unittest
from pathlib import Path
from copy import copy

from parameterized import parameterized, param
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal

from roiextractors import (
    TiffImagingExtractor,
    ScanImageTiffImagingExtractor,
    Hdf5ImagingExtractor,
    SbxImagingExtractor,
    CaimanSegmentationExtractor,
    ExtractSegmentationExtractor,
    Suite2pSegmentationExtractor,
    CnmfeSegmentationExtractor,
)
from roiextractors.testing import check_imaging_equal, check_segmentations_equal

from .setup_paths import OPHYS_DATA_PATH, OUTPUT_PATH


def custom_name_func(testcase_func, param_num, param):
    return (
        f"{testcase_func.__name__}_{param_num}_"
        f"{parameterized.to_safe_name(param.kwargs['extractor_class'].__name__)}"
    )


class TestExtractors(TestCase):
    savedir = OUTPUT_PATH

    imaging_extractor_list = [
        param(
            extractor_class=TiffImagingExtractor,
            extractor_kwargs=dict(
                file_path=str(OPHYS_DATA_PATH / "imaging_datasets" / "Tif" / "demoMovie.tif"),
                sampling_frequency=15.0,  # typically provied by user
            ),
        ),
        param(
            extractor_class=ScanImageTiffImagingExtractor,
            extractor_kwargs=dict(
                file_path=str(OPHYS_DATA_PATH / "imaging_datasets" / "Tif" / "sample_scanimage.tiff")
            ),
        ),
        param(
            extractor_class=Hdf5ImagingExtractor,
            extractor_kwargs=dict(file_path=str(OPHYS_DATA_PATH / "imaging_datasets" / "hdf5" / "demoMovie.hdf5")),
        ),
    ]
    for suffix in [".mat", ".sbx"]:
        imaging_extractor_list.append(
            param(
                extractor_class=SbxImagingExtractor,
                extractor_kwargs=dict(
                    file_path=str(OPHYS_DATA_PATH / "imaging_datasets" / "Scanbox" / f"sample{suffix}")
                ),
            ),
        )

    @parameterized.expand(imaging_extractor_list, name_func=custom_name_func)
    def test_imaging_extractors(self, extractor_class, extractor_kwargs):
        extractor = extractor_class(**extractor_kwargs)
        try:
            suffix = Path(extractor_kwargs["file_path"]).suffix
            output_path = self.savedir / f"{extractor_class.__name__}{suffix}"
            extractor_class.write_imaging(extractor, output_path)

            roundtrip_kwargs = copy(extractor_kwargs)
            roundtrip_kwargs.update(file_path=output_path)
            roundtrip_extractor = extractor_class(**roundtrip_kwargs)
            check_imaging_equal(imaging_extractor1=extractor, imaging_extractor2=roundtrip_extractor)
        except NotImplementedError:
            return

    segmentation_extractor_list = [
        param(
            extractor_class=CaimanSegmentationExtractor,
            extractor_kwargs=dict(
                file_path=str(OPHYS_DATA_PATH / "segmentation_datasets" / "caiman" / "caiman_analysis.hdf5")
            ),
        ),
        param(
            extractor_class=CnmfeSegmentationExtractor,
            extractor_kwargs=dict(
                file_path=str(
                    OPHYS_DATA_PATH
                    / "segmentation_datasets"
                    / "cnmfe"
                    / "2014_04_01_p203_m19_check01_cnmfeAnalysis.mat"
                )
            ),
        ),
        param(
            extractor_class=ExtractSegmentationExtractor,
            extractor_kwargs=dict(
                file_path=str(
                    OPHYS_DATA_PATH
                    / "segmentation_datasets"
                    / "extract"
                    / "2014_04_01_p203_m19_check01_extractAnalysis.mat"
                )
            ),
        ),
        param(
            extractor_class=Suite2pSegmentationExtractor,
            extractor_kwargs=dict(
                # TODO: argument name is 'file_path' on roiextractors, but it clearly refers to a folder_path
                file_path=str(OPHYS_DATA_PATH / "segmentation_datasets" / "suite2p")
            ),
        ),
    ]

    @parameterized.expand(segmentation_extractor_list, name_func=custom_name_func)
    def test_segmentation_extractors(self, extractor_class, extractor_kwargs):
        extractor = extractor_class(**extractor_kwargs)
        try:
            suffix = Path(extractor_kwargs["file_path"]).suffix
            output_path = self.savedir / f"{extractor_class.__name__}{suffix}"

            # TODO:  ExtractSegmentation
            extractors_not_ready = [
                "ExtractSegmentationExtractor",
            ]

            if extractor_class.__name__ not in extractors_not_ready:
                extractor_class.write_segmentation(extractor, output_path)

                roundtrip_kwargs = copy(extractor_kwargs)
                roundtrip_kwargs.update(file_path=output_path)
                roundtrip_extractor = extractor_class(**roundtrip_kwargs)
                check_segmentations_equal(
                    segmentation_extractor1=extractor, segmentation_extractor2=roundtrip_extractor
                )

        except NotImplementedError:
            return


if __name__ == "__main__":
    unittest.main()
