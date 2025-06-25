import h5py
import numpy as np
from hdmf.testing import TestCase
from numpy.testing import assert_array_almost_equal

from roiextractors import CaimanSegmentationExtractor
from .setup_paths import OPHYS_DATA_PATH


class TestCaimanSegmentationExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_path = OPHYS_DATA_PATH / "segmentation_datasets" / "caiman" / "caiman_analysis.hdf5"
        cls.extractor = CaimanSegmentationExtractor(file_path=cls.file_path)

    def test_get_num_rois(self):
        expected_num_rois = 72
        self.assertEqual(self.extractor.get_num_rois(), expected_num_rois)

    def test_get_num_samples(self):
        expected_num_samples = 1000
        self.assertEqual(self.extractor.get_num_samples(), expected_num_samples)

    def test_get_frame_shape(self):
        frame_shape = self.extractor.get_frame_shape()
        expected_frame_shape = (128, 128)
        self.assertEqual(tuple(frame_shape), expected_frame_shape)

    def test_get_num_planes(self):
        expected_num_planes = 1
        self.assertEqual(self.extractor.get_num_planes(), expected_num_planes)

    def test_get_sampling_frequency(self):
        expected_sampling_frequency = 30.0
        self.assertEqual(self.extractor.get_sampling_frequency(), expected_sampling_frequency)

    def test_get_roi_ids(self):
        expected_num_rois = 72
        expected_roi_ids = list(range(expected_num_rois))
        self.assertEqual(self.extractor.get_roi_ids(), expected_roi_ids)

    def test_get_accepted_list(self):
        expected_num_rois = 72
        expected_accepted_list = list(range(expected_num_rois))
        self.assertEqual(self.extractor.get_accepted_list(), expected_accepted_list)

    def test_get_rejected_list(self):
        expected_rejected_list = []
        self.assertEqual(self.extractor.get_rejected_list(), expected_rejected_list)

    def test_get_traces_raw(self):
        expected_num_samples = 1000
        expected_num_rois = 72
        raw_traces = self.extractor.get_traces(name="raw")
        self.assertEqual(raw_traces.shape, (expected_num_samples, expected_num_rois))
        with h5py.File(self.file_path, "r") as f:
            C = f["estimates"]["C"][:]
            YrA = f["estimates"]["YrA"][:]
            expected_raw_traces = (C + YrA).T
        assert_array_almost_equal(raw_traces, expected_raw_traces, decimal=6)

    def test_get_traces_with_roi_ids(self):
        expected_num_samples = 1000
        roi_ids = [0, 1, 2]
        traces = self.extractor.get_traces(name="raw", roi_ids=roi_ids)
        expected_shape = (expected_num_samples, len(roi_ids))
        self.assertEqual(traces.shape, expected_shape)
        full_traces = self.extractor.get_traces(name="raw")
        assert_array_almost_equal(traces, full_traces[:, :3], decimal=6)

    def test_get_roi_pixel_masks(self):
        expected_num_rois = 72
        pixel_masks = self.extractor.get_roi_pixel_masks()
        self.assertEqual(len(pixel_masks), expected_num_rois)
        first_mask = pixel_masks[0]
        self.assertEqual(first_mask.shape[1], 3)

    def test_get_image_correlation(self):
        correlation_image = self.extractor.get_image(name="correlation")
        self.assertIsNone(correlation_image)

    def test_get_snr_values(self):
        snr_values = self.extractor._get_snr_values()
        self.assertIsNone(snr_values)

    def test_get_spatial_correlation_values(self):
        r_values = self.extractor._get_spatial_correlation_values()
        self.assertIsNone(r_values)

    def test_get_cnn_predictions(self):
        cnn_preds = self.extractor._get_cnn_predictions()
        self.assertIsNone(cnn_preds)

    def test_get_quality_metrics(self):
        quality_metrics = self.extractor.get_quality_metrics()

        for metric_name in ["snr", "r_values", "cnn_preds"]:
            self.assertNotIn(metric_name, quality_metrics)

    def test_get_traces_denoised(self):
        """Test that get_traces returns the expected denoised traces."""
        expected_num_samples = 1000
        expected_num_rois = 72

        denoised_traces = self.extractor.get_traces(name="denoised")
        self.assertEqual(denoised_traces.shape, (expected_num_samples, expected_num_rois))

        # Compare with expected data from file (C)
        with h5py.File(self.file_path, "r") as f:
            expected_denoised_traces = f["estimates"]["C"][:].T

        assert_array_almost_equal(denoised_traces, expected_denoised_traces, decimal=6)

    def test_get_traces_dff(self):
        """Test that get_traces returns the expected dF/F traces."""
        expected_num_samples = 1000
        expected_num_rois = 72

        dff_traces = self.extractor.get_traces(name="dff")
        self.assertEqual(dff_traces.shape, (expected_num_samples, expected_num_rois))

        # Compare with expected data from file (F_dff)
        with h5py.File(self.file_path, "r") as f:
            expected_dff_traces = f["estimates"]["F_dff"][:].T

        assert_array_almost_equal(dff_traces, expected_dff_traces, decimal=6)

    def test_get_traces_deconvolved(self):
        """Test that get_traces returns the expected deconvolved traces."""
        expected_num_samples = 1000
        expected_num_rois = 72

        deconvolved_traces = self.extractor.get_traces(name="deconvolved")
        self.assertEqual(deconvolved_traces.shape, (expected_num_samples, expected_num_rois))

        # Compare with expected data from file (S)
        with h5py.File(self.file_path, "r") as f:
            expected_deconvolved_traces = f["estimates"]["S"][:].T

        assert_array_almost_equal(deconvolved_traces, expected_deconvolved_traces, decimal=6)

    def test_get_traces_neuropil(self):
        """Test that get_traces returns the expected neuropil traces."""
        expected_num_samples = 1000
        expected_neuropil_components = 2

        neuropil_traces = self.extractor.get_traces(name="neuropil")
        self.assertEqual(neuropil_traces.shape, (expected_num_samples, expected_neuropil_components))

        # Compare with expected data from file (f)
        with h5py.File(self.file_path, "r") as f:
            expected_neuropil_traces = f["estimates"]["f"][:].T

        assert_array_almost_equal(neuropil_traces, expected_neuropil_traces, decimal=6)


class TestCaimanMini100SegmentationExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_path = (
            OPHYS_DATA_PATH
            / "segmentation_datasets"
            / "caiman"
            / "multi_plane_with_imaging_data"
            / "mini_1000_caiman_stubbed_10units_5frames.hdf5"
        )
        cls.extractor = CaimanSegmentationExtractor(file_path=cls.file_path)

    def test_get_num_rois(self):
        expected_num_rois = 10
        self.assertEqual(self.extractor.get_num_rois(), expected_num_rois)

    def test_get_num_samples(self):
        expected_num_samples = 5
        self.assertEqual(self.extractor.get_num_samples(), expected_num_samples)

    def test_get_frame_shape(self):
        frame_shape = self.extractor.get_frame_shape()
        expected_frame_shape = (231, 242)
        self.assertEqual(tuple(frame_shape), expected_frame_shape)

    def test_get_num_planes(self):
        expected_num_planes = 1
        self.assertEqual(self.extractor.get_num_planes(), expected_num_planes)

    def test_get_sampling_frequency(self):
        expected_sampling_frequency = 20.0
        self.assertEqual(self.extractor.get_sampling_frequency(), expected_sampling_frequency)

    def test_get_roi_ids(self):
        expected_num_rois = 10
        expected_roi_ids = list(range(expected_num_rois))
        self.assertEqual(self.extractor.get_roi_ids(), expected_roi_ids)

    def test_get_accepted_list(self):
        expected_accepted_list = [0, 3, 4, 5, 6, 7]
        self.assertEqual(self.extractor.get_accepted_list(), expected_accepted_list)

    def test_get_rejected_list(self):
        expected_rejected_list = [1, 2, 8, 9]
        self.assertEqual(self.extractor.get_rejected_list(), expected_rejected_list)

    def test_get_traces_raw(self):
        """
        Test that the extractor returns the correct raw traces.

        This test checks:
        - The shape of the returned raw traces matches the expected number of samples and ROIs.
        - The values of the raw traces match the sum of the 'C' and 'YrA' datasets from the CaImAn HDF5 file,
        transposed to (num_samples, num_rois) shape.
        """
        expected_num_samples = 5
        expected_num_rois = 10
        raw_traces = self.extractor.get_traces(name="raw")
        self.assertEqual(raw_traces.shape, (expected_num_samples, expected_num_rois))

        # Verify against file data
        with h5py.File(self.file_path, "r") as f:
            C = f["estimates"]["C"][:]
            YrA = f["estimates"]["YrA"][:]
            expected_raw_traces = (C + YrA).T
        assert_array_almost_equal(raw_traces, expected_raw_traces, decimal=6)

    def test_get_traces_with_roi_ids(self):
        """
        Test that the extractor returns the correct traces for a subset of ROI IDs.

        This test checks:
        - The shape of the returned traces matches (num_samples, number of selected ROI IDs).
        - The values of the returned traces for the selected ROI IDs match the corresponding columns
        in the full traces array.
        """
        expected_num_samples = 5
        roi_ids = [0, 1, 2]
        traces = self.extractor.get_traces(name="raw", roi_ids=roi_ids)
        expected_shape = (expected_num_samples, len(roi_ids))
        self.assertEqual(traces.shape, expected_shape)

        full_traces = self.extractor.get_traces(name="raw")
        assert_array_almost_equal(traces, full_traces[:, :3], decimal=6)

    def test_get_traces_denoised(self):
        expected_num_samples = 5
        expected_num_rois = 10

        denoised_traces = self.extractor.get_traces(name="denoised")
        self.assertEqual(denoised_traces.shape, (expected_num_samples, expected_num_rois))

        # Compare with expected data from file (C)
        with h5py.File(self.file_path, "r") as f:
            expected_denoised_traces = f["estimates"]["C"][:].T
        assert_array_almost_equal(denoised_traces, expected_denoised_traces, decimal=6)

    def test_get_traces_dff(self):
        expected_num_samples = 5
        expected_num_rois = 10

        dff_traces = self.extractor.get_traces(name="dff")
        self.assertEqual(dff_traces.shape, (expected_num_samples, expected_num_rois))

        # Compare with expected data from file (F_dff)
        with h5py.File(self.file_path, "r") as f:
            expected_dff_traces = f["estimates"]["F_dff"][:].T
        assert_array_almost_equal(dff_traces, expected_dff_traces, decimal=6)

    def test_get_traces_deconvolved(self):
        expected_num_samples = 5
        expected_num_rois = 10

        deconvolved_traces = self.extractor.get_traces(name="deconvolved")
        self.assertEqual(deconvolved_traces.shape, (expected_num_samples, expected_num_rois))

        # Compare with expected data from file (S)
        with h5py.File(self.file_path, "r") as f:
            expected_deconvolved_traces = f["estimates"]["S"][:].T
        assert_array_almost_equal(deconvolved_traces, expected_deconvolved_traces, decimal=6)

    def test_get_traces_neuropil(self):
        # Based on file analysis, 'f' is stored as empty object array
        neuropil_traces = self.extractor.get_traces(name="neuropil")
        self.assertIsNone(neuropil_traces)

    def test_get_roi_pixel_masks(self):
        expected_num_rois = 10
        pixel_masks = self.extractor.get_roi_pixel_masks()
        self.assertEqual(len(pixel_masks), expected_num_rois)

        first_mask = pixel_masks[0]
        self.assertEqual(first_mask.shape[1], 3)  # [y, x, weight] format

    def test_get_roi_pixel_masks_with_roi_ids(self):
        roi_ids = [0, 2]
        pixel_masks = self.extractor.get_roi_pixel_masks(roi_ids=roi_ids)
        self.assertEqual(len(pixel_masks), len(roi_ids))

    def test_get_image_correlation(self):
        correlation_image = self.extractor.get_image(name="correlation")
        self.assertIsNotNone(correlation_image)
        expected_shape = (231, 242)
        self.assertEqual(correlation_image.shape, expected_shape)

    def test_get_image_mean(self):
        # Based on file analysis, 'b' is stored as empty object array
        mean_image = self.extractor.get_image(name="mean")
        self.assertIsNone(mean_image)

    def test_get_snr_values(self):
        snr_values = self.extractor._get_snr_values()
        self.assertIsNotNone(snr_values)
        expected_snr = np.array(
            [
                2.91673987,
                1.05541609,
                1.59315755,
                4.66348506,
                2.97824475,
                2.98275578,
                2.9776668,
                2.94389109,
                1.19369399,
                1.76922279,
            ]
        )
        assert_array_almost_equal(snr_values, expected_snr, decimal=6)

    def test_get_spatial_correlation_values(self):
        r_values = self.extractor._get_spatial_correlation_values()
        self.assertIsNotNone(r_values)
        expected_r_values = np.array(
            [
                0.35878084,
                0.182326,
                0.34179541,
                0.45524581,
                0.13731668,
                0.58202635,
                0.26673053,
                0.29431159,
                0.3361319,
                0.34527933,
            ]
        )
        assert_array_almost_equal(r_values, expected_r_values, decimal=6)

    def test_get_cnn_predictions(self):
        # cnn_preds has shape=(0,) and size=0
        cnn_preds = self.extractor._get_cnn_predictions()
        self.assertIsNone(cnn_preds)

    def test_get_quality_metrics(self):
        quality_metrics = self.extractor.get_quality_metrics()

        expected_metrics = ["snr", "r_values"]
        for metric_name in expected_metrics:
            self.assertIn(metric_name, quality_metrics)
            self.assertEqual(len(quality_metrics[metric_name]), 10)

        self.assertNotIn("cnn_predictions", quality_metrics)
