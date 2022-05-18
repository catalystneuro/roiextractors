import unittest
from pathlib import Path
from copy import copy

from parameterized import parameterized, param

from roiextractors import (
    TiffImagingExtractor,
    Hdf5ImagingExtractor,
    SbxImagingExtractor,
    CaimanSegmentationExtractor,
    ExtractSegmentationExtractor,
    Suite2pSegmentationExtractor,
    CnmfeSegmentationExtractor,
)


from .setup_paths import OPHYS_DATA_PATH, OUTPUT_PATH


def custom_name_func(testcase_func, param_num, param):
    return (
        f"{testcase_func.__name__}_{param_num}_"
        f"{parameterized.to_safe_name(param.kwargs['extractor_class'].__name__)}"
    )


class TestExtractors(unittest.TestCase):
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
            # TODO: this roundtrip test has been failing for some time now
            # check_imaging_equal(img1=extractor, img2=roundtrip_extractor)
        except NotImplementedError:
            return

    @parameterized.expand(segmentation_extractor_list, name_func=custom_name_func)
    def test_segmentation_extractors(self, extractor_class, extractor_kwargs):
        extractor = extractor_class(**extractor_kwargs)
        try:
            suffix = Path(extractor_kwargs["file_path"]).suffix
            output_path = self.savedir / f"{extractor_class.__name__}{suffix}"

            # TODO: Suit2P Segmentation fails to make certain files; probably related to how
            # the input argument is a 'file_path' but is actually a folder?
            if extractor_class.__name__ != "Suite2pSegmentationExtractor":
                extractor_class.write_segmentation(extractor, output_path)

                roundtrip_kwargs = copy(extractor_kwargs)
                roundtrip_kwargs.update(file_path=output_path)
                roundtrip_extractor = extractor_class(**roundtrip_kwargs)
                # TODO: this roundtrip test has been failing for some time now
                # check_segmentations_equal(img1=extractor, img2=roundtrip_extractor)
        except NotImplementedError:
            return


if __name__ == "__main__":
    unittest.main()
