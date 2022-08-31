import unittest
from pathlib import Path
from copy import copy

from parameterized import parameterized, param
from hdmf.testing import TestCase

from roiextractors import (
    TiffImagingExtractor,
    ScanImageTiffImagingExtractor,
    Hdf5ImagingExtractor,
    SbxImagingExtractor,
    CaimanSegmentationExtractor,
    Suite2pSegmentationExtractor,
    CnmfeSegmentationExtractor,
)
from roiextractors.extractors.schnitzerextractor import LegacyExtractSegmentationExtractor
from roiextractors.testing import (
    check_imaging_equal,
    check_segmentations_equal,
    assert_get_frames_return_shape,
    check_imaging_return_types,
)


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
                file_path=str(OPHYS_DATA_PATH / "imaging_datasets" / "Tif" / "sample_scanimage.tiff"),
                sampling_frequency=30.0,
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
        check_imaging_return_types(extractor)

        try:
            suffix = Path(extractor_kwargs["file_path"]).suffix
            output_path = self.savedir / f"{extractor_class.__name__}{suffix}"
            extractor_class.write_imaging(extractor, output_path)

            roundtrip_kwargs = copy(extractor_kwargs)
            roundtrip_kwargs.update(file_path=output_path)
            roundtrip_extractor = extractor_class(**roundtrip_kwargs)
            check_imaging_equal(imaging_extractor1=extractor, imaging_extractor2=roundtrip_extractor)
            check_imaging_return_types(roundtrip_extractor)

        except NotImplementedError:
            return

    @parameterized.expand(imaging_extractor_list, name_func=custom_name_func)
    def test_imaging_extractors_canonical_shape(self, extractor_class, extractor_kwargs):
        """Test that get_video and get_frame methods for their shapes and types under different indexing scenarios"""
        extractor = extractor_class(**extractor_kwargs)
        image_size = extractor.get_image_size()
        num_channels = extractor.get_num_channels()

        # Test canonical shape for get video
        video = extractor.get_video()
        canonical_video_shape = [extractor.get_num_frames(), image_size[0], image_size[1]]
        if num_channels > 1:
            canonical_video_shape.append(num_channels)
        assert video.shape == tuple(canonical_video_shape)

        # Test spikeinterface-like behavior
        one_element_video_shape = extractor.get_video(start_frame=0, end_frame=1, channel=0).shape
        expected_shape = (1, image_size[0], image_size[1])
        assert one_element_video_shape == expected_shape

        # Test frames behavior
        assert_get_frames_return_shape(imaging_extractor=extractor)

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
            extractor_class=LegacyExtractSegmentationExtractor,
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
            extractor_kwargs=dict(folder_path=str(OPHYS_DATA_PATH / "segmentation_datasets" / "suite2p")),
        ),
        param(
            extractor_class=Suite2pSegmentationExtractor,
            extractor_kwargs=dict(file_path=str(OPHYS_DATA_PATH / "segmentation_datasets" / "suite2p")),
        ),
    ]

    @parameterized.expand(segmentation_extractor_list, name_func=custom_name_func)
    def test_segmentation_extractors(self, extractor_class, extractor_kwargs):
        extractor = extractor_class(**extractor_kwargs)

        try:
            roundtrip_kwargs = copy(extractor_kwargs)
            if "folder_path" in extractor_kwargs:
                output_path = self.savedir / f"{extractor_class.__name__}"
                roundtrip_kwargs.update(folder_path=output_path)
            elif "file_path" in extractor_kwargs:
                suffix = Path(extractor_kwargs["file_path"]).suffix
                output_path = self.savedir / f"{extractor_class.__name__}{suffix}"
                roundtrip_kwargs.update(file_path=output_path)

            extractor_class.write_segmentation(extractor, output_path)

            roundtrip_kwargs = copy(extractor_kwargs)
            roundtrip_extractor = extractor_class(**roundtrip_kwargs)
            check_segmentations_equal(segmentation_extractor1=extractor, segmentation_extractor2=roundtrip_extractor)

            num_frames = extractor.get_num_frames()
            num_rois = extractor.get_num_rois()
            for trace in extractor.get_traces_dict().values():
                if trace is None:
                    continue
                assert trace.shape[0] == num_frames
                assert trace.shape[1] == num_rois

            rountrip_num_frames = roundtrip_extractor.get_num_frames()
            rountrip_num_rois = roundtrip_extractor.get_num_rois()
            for trace in roundtrip_extractor.get_traces_dict().values():
                if trace is None:
                    continue
                assert trace.shape[0] == rountrip_num_frames
                assert trace.shape[1] == rountrip_num_rois

        except NotImplementedError:
            return


if __name__ == "__main__":
    unittest.main()
