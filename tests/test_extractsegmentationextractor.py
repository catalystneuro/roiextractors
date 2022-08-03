from hdmf.testing import TestCase
from parameterized import parameterized, param

from roiextractors import (
    ExtractSegmentationExtractor,
    NewExtractSegmentationExtractor,
    LegacyExtractSegmentationExtractor,
)

from .setup_paths import OPHYS_DATA_PATH


class TestExtractSegmentationExtractor(TestCase):
    ophys_data_path = OPHYS_DATA_PATH / "segmentation_datasets" / "extract"

    def test_extract_segmentation_extractor_file_path_does_not_exist(self):
        """Test that the extractor raises an error if the file does not exist."""
        not_a_mat_file_path = "not_a_mat_file.txt"
        with self.assertRaisesWith(AssertionError, f"File {not_a_mat_file_path} does not exist."):
            ExtractSegmentationExtractor(file_path=not_a_mat_file_path)

    def test_extract_segmentation_extractor_file_path_is_not_a_mat_file(self):
        """Test that the extractor raises an error if the file is not a .mat file."""
        not_a_mat_file_path = OPHYS_DATA_PATH / "segmentation_datasets" / "nwb" / "nwb_test.nwb"
        with self.assertRaisesWith(AssertionError, f"File {not_a_mat_file_path} must be a .mat file."):
            ExtractSegmentationExtractor(file_path=not_a_mat_file_path)

    param_list = [
        param(
            file_path=ophys_data_path / "2014_04_01_p203_m19_check01_extractAnalysis.mat",
            output_struct_name="extractAnalysisOutput",
            extractor_class=LegacyExtractSegmentationExtractor,
        ),
        param(
            file_path=ophys_data_path / "2014_04_01_p203_m19_check01_extractAnalysis.mat",
            output_struct_name=None,
            extractor_class=LegacyExtractSegmentationExtractor,
        ),
        param(
            file_path=ophys_data_path / "extract_public_output.mat",
            output_struct_name="output",
            extractor_class=NewExtractSegmentationExtractor,
        ),
        param(
            file_path=ophys_data_path / "extract_public_output.mat",
            output_struct_name=None,
            extractor_class=NewExtractSegmentationExtractor,
        ),
    ]

    @parameterized.expand(
        param_list,
    )
    def test_extract_segmentation_extractor_redirects(self, file_path, output_struct_name, extractor_class):
        """
        Test that the extractor class redirects to the correct class
        given the version of the .mat file.
        """
        extractor = ExtractSegmentationExtractor(
            file_path=file_path,
            output_struct_name=output_struct_name,
        )

        self.assertIsInstance(extractor, extractor_class)
