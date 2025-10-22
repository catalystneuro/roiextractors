import h5py
import numpy as np
from hdmf.testing import TestCase
from lazy_ops import DatasetView
from numpy.testing import assert_array_equal
from parameterized import param, parameterized

from roiextractors import ExtractSegmentationExtractor
from roiextractors.extractors.schnitzerextractor import (
    LegacyExtractSegmentationExtractor,
    NewExtractSegmentationExtractor,
)

from .setup_paths import OPHYS_DATA_PATH


class TestExtractSegmentationExtractor(TestCase):
    ophys_data_path = OPHYS_DATA_PATH / "segmentation_datasets" / "extract"

    @classmethod
    def setUpClass(cls):
        cls.sampling_frequency = 30.0
        cls.ophys_data_path = OPHYS_DATA_PATH / "segmentation_datasets" / "extract"

    def test_extract_segmentation_extractor_file_path_does_not_exist(self):
        """Test that the extractor raises an error if the file does not exist."""
        not_a_mat_file_path = "not_a_mat_file.txt"
        with self.assertRaisesWith(AssertionError, f"File {not_a_mat_file_path} does not exist."):
            ExtractSegmentationExtractor(
                file_path=not_a_mat_file_path,
                sampling_frequency=self.sampling_frequency,
            )

    def test_extract_segmentation_extractor_file_path_is_not_a_mat_file(self):
        """Test that the extractor raises an error if the file is not a .mat file."""
        not_a_mat_file_path = OPHYS_DATA_PATH / "segmentation_datasets" / "nwb" / "nwb_test.nwb"
        with self.assertRaisesWith(AssertionError, f"File {not_a_mat_file_path} must be a .mat file."):
            ExtractSegmentationExtractor(
                file_path=not_a_mat_file_path,
                sampling_frequency=self.sampling_frequency,
            )

    def test_extract_segmentation_extractor_user_given_output_struct_name_not_in_file(self):
        """Test that the extractor returns the expected error when a user given output
        struct name is not in the file."""
        file_path = self.ophys_data_path / "2014_04_01_p203_m19_check01_extractAnalysis.mat"
        with self.assertRaisesWith(AssertionError, "Output struct name 'not_output' not found in file."):
            ExtractSegmentationExtractor(
                file_path=file_path,
                sampling_frequency=self.sampling_frequency,
                output_struct_name="not_output",
            )

    param_list = [
        param(
            file_path=ophys_data_path / "2014_04_01_p203_m19_check01_extractAnalysis.mat",
            extractor_class=LegacyExtractSegmentationExtractor,
        ),
        param(
            file_path=ophys_data_path / "extract_public_output.mat",
            extractor_class=NewExtractSegmentationExtractor,
        ),
    ]

    @parameterized.expand(
        param_list,
    )
    def test_extract_segmentation_extractor_redirects(self, file_path, extractor_class):
        """
        Test that the extractor class redirects to the correct class
        given the version of the .mat file.
        """
        extractor = ExtractSegmentationExtractor(
            file_path=file_path,
            sampling_frequency=self.sampling_frequency,
        )

        self.assertIsInstance(extractor, extractor_class)


class TestNewExtractSegmentationExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_path = OPHYS_DATA_PATH / "segmentation_datasets" / "extract" / "extract_public_output.mat"
        cls.output_struct_name = "output"
        cls.sampling_frequency = 30.0

    def setUp(self):
        self.extractor = NewExtractSegmentationExtractor(
            file_path=self.file_path,
            output_struct_name=self.output_struct_name,
            sampling_frequency=self.sampling_frequency,
        )

    def tearDown(self):
        self.extractor.close()

    def test_extractor_output_struct_assertion(self):
        """Test that the extractor raises an error if the output struct name is not in the file."""
        with self.assertRaisesWith(AssertionError, "Output struct not found in file."):
            NewExtractSegmentationExtractor(
                file_path=self.file_path,
                output_struct_name="not_output",
                sampling_frequency=self.sampling_frequency,
            )

    def test_extractor_no_timestamps_or_sampling_frequency(self):
        """Test that the extractor raises an error if neither timestamps
        nor sampling frequency are provided."""
        with self.assertRaisesWith(exc_type=AssertionError, exc_msg=("The sampling_frequency must be provided.")):
            NewExtractSegmentationExtractor(file_path=self.file_path, sampling_frequency=None)

    def test_extractor_data_validity(self):
        """Test that the extractor class returns the expected data."""

        with h5py.File(self.file_path, "r") as segmentation_file:
            spatial_weights = DatasetView(
                segmentation_file[self.output_struct_name]["spatial_weights"]
            ).lazy_transpose()
            # Check via the public API instead of internal attributes
            self.assertEqual(self.extractor._roi_masks.data.shape, spatial_weights.shape)

            dff_traces = self.extractor.get_traces(name="dff")
            self.assertEqual(dff_traces.shape, (2000, 20))

            raw_traces = self.extractor.get_traces(name="raw")
            self.assertIsNone(raw_traces)

            self.assertEqual(self.extractor._sampling_frequency, self.sampling_frequency)
            self.assertIsInstance(self.extractor.get_sampling_frequency(), float)

            assert_array_equal(self.extractor.get_frame_shape(), [50, 50])

            self.assertEqual(self.extractor.get_num_rois(), 20)
            self.assertEqual(self.extractor.get_num_samples(), 2000)

            self.assertEqual(self.extractor.get_rejected_list(), [])
            self.assertEqual(self.extractor.get_accepted_list(), list(range(20)))

    def test_extractor_config(self):
        """Test that the extractor class returns the expected config."""

        # Assert that all keys are extracted without nesting
        self.assertEqual(len(self.extractor.config), 93)
        assert "version" in self.extractor.config
        self.assertEqual(self.extractor.config["version"], "1.1.0")

        assert "preprocess" in self.extractor.config
        self.assertEqual(self.extractor.config["preprocess"], [1])

        assert "S_corr_thresh" in self.extractor.config
        self.assertEqual(self.extractor.config["S_corr_thresh"], [0.1])

        self.assertEqual(self.extractor.config["S_dup_corr_thresh"], [0.95])
        self.assertEqual(self.extractor.config["T_dup_corr_thresh"], [0.95])

        assert "trace_output_option" in self.extractor.config
        self.assertEqual(self.extractor.config["trace_output_option"], "raw")
        assert "cellfind_filter_type" in self.extractor.config
        self.assertEqual(self.extractor.config["cellfind_filter_type"], "none")

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
        dummy_image_mask = np.zeros((50, 50, 20))
        dummy_image_mask[..., accepted_list] = 1

        # Update the internal representation data
        self.extractor._roi_masks.data = dummy_image_mask

        assert_array_equal(self.extractor.get_accepted_list(), accepted_list)
        assert_array_equal(
            self.extractor.get_rejected_list(),
            list(set(range(20)) - set(accepted_list)),
        )

    def test_extractor_get_images_dict(self):
        """Test that the extractor class returns the expected images dict."""
        with h5py.File(self.file_path, "r") as segmentation_file:
            summary_image = DatasetView(
                segmentation_file[self.output_struct_name]["info"]["summary_image"],
            )[:].T
            max_image = DatasetView(
                segmentation_file[self.output_struct_name]["info"]["max_image"],
            )[:].T
            f_per_pixel = DatasetView(
                segmentation_file[self.output_struct_name]["info"]["F_per_pixel"],
            )[:].T

        images_dict = self.extractor.get_images_dict()
        self.assertEqual(len(images_dict), 5)

        self.assertEqual(images_dict["correlation"], None)
        self.assertEqual(images_dict["mean"], None)

        self.assertEqual(images_dict["summary_image"].shape, summary_image.shape)
        self.assertEqual(images_dict["max_image"].shape, max_image.shape)
        self.assertEqual(images_dict["f_per_pixel"].shape, f_per_pixel.shape)

        assert_array_equal(images_dict["summary_image"], summary_image)
        assert_array_equal(images_dict["max_image"], max_image)
        assert_array_equal(images_dict["f_per_pixel"], f_per_pixel)
