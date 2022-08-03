import h5py
import numpy as np
from hdmf.testing import TestCase
from lazy_ops import DatasetView
from numpy.testing import assert_array_equal
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


class TestNewExtractSegmentationExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_path = OPHYS_DATA_PATH / "segmentation_datasets" / "extract" / "extract_public_output.mat"
        cls.output_struct_name = "output"

    def setUp(self):
        self.extractor = NewExtractSegmentationExtractor(
            file_path=self.file_path,
            output_struct_name=self.output_struct_name,
        )

    def test_extractor_output_struct_assertion(self):
        """Test that the extractor raises an error if the output struct name is not in the file."""
        with self.assertRaisesWith(AssertionError, "Output struct not found in file."):
            NewExtractSegmentationExtractor(
                file_path=self.file_path,
                output_struct_name="not_output",
            )

    def test_extractor_data_validity(self):
        """Test that the extractor class returns the expected data."""

        with h5py.File(self.file_path, "r") as segmentation_file:
            spatial_weights = DatasetView(
                segmentation_file[self.output_struct_name]["spatial_weights"]
            ).lazy_transpose()
            self.assertEqual(self.extractor._image_masks.shape, spatial_weights.shape)

            temporal_weights = DatasetView(segmentation_file[self.output_struct_name]["temporal_weights"])
            self.assertEqual(self.extractor._roi_response_dff.shape, temporal_weights.shape)

            self.assertEqual(self.extractor._roi_response_raw, None)

            summary_image = segmentation_file[self.output_struct_name]["info"]["summary_image"][:]
            self.assertEqual(self.extractor._image_correlation.shape, summary_image.shape)

            runtime = segmentation_file[self.output_struct_name]["info"]["runtime"][:][0, 0]
            self.assertEqual(self.extractor._sampling_frequency, temporal_weights.shape[1] / runtime)
            self.assertIsInstance(self.extractor.get_sampling_frequency(), float)

            assert_array_equal(self.extractor.get_image_size(), summary_image.shape)

            num_rois = temporal_weights.shape[0]
            self.assertEqual(self.extractor.get_num_rois(), num_rois)

            num_frames = temporal_weights.shape[1]
            self.assertEqual(self.extractor.get_num_frames(), num_frames)

            self.assertEqual(self.extractor.get_rejected_list(), [])
            self.assertEqual(self.extractor.get_accepted_list(), list(range(num_rois)))

    param_list = [
        param(accepted_list=[4, 6, 12, 18]),
        param(accepted_list=[]),
        param(accepted_list=list(range(20))),
    ]

    @parameterized.expand(
        param_list,
    )
    def test_extractor_accepted_list(self, accepted_list):
        """Test that the extractor class returns the list of accepted and rejected ROIs
        correctly given the list of non-zero ROIs."""
        dummy_image_mask = np.zeros((20, 50, 50))
        dummy_image_mask[accepted_list, ...] = 1

        self.extractor._image_masks = dummy_image_mask

        assert_array_equal(self.extractor.get_accepted_list(), accepted_list)
        assert_array_equal(
            self.extractor.get_rejected_list(),
            list(set(range(20)) - set(accepted_list)),
        )
