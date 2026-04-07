"""Tests for the OMETiffImagingExtractor using real OME-TIFF test data."""

from pathlib import Path

import pytest
from numpy.testing import assert_array_equal

from roiextractors import OMETiffImagingExtractor

OME_TIFF_PATH = Path("/home/heberto/data/ome_tiff")


@pytest.mark.skipif(not OME_TIFF_PATH.exists(), reason=f"Test data not found at {OME_TIFF_PATH}")
class TestOMETiffImagingExtractor:
    """Test the OMETiffImagingExtractor with various OME-TIFF datasets."""

    def test_single_channel_single_plane(self):
        """Test with single-channel, planar time series data in a single file.

        File: MitoCheck/00001_01.ome.tiff
        - Samples (T): 93
        - Channels (C): 1
        - Depth planes (Z): 1
        - Frame shape: 1024 x 1344
        - Volumetric: False
        - Pages/IFDs: 93
        """
        file_path = OME_TIFF_PATH / "MitoCheck" / "00001_01.ome.tiff"

        extractor = OMETiffImagingExtractor(file_path, sampling_frequency=1.0)

        assert extractor.get_num_samples() == 93
        assert extractor.get_image_shape() == (1024, 1344)
        assert extractor.get_num_planes() == 1
        assert extractor.is_volumetric is False

        data = extractor.get_series(0, 2)
        assert data.shape == (2, 1024, 1344)

    def test_multi_channel_single_plane(self):
        """Test with multi-channel, planar time series data across multiple files.

        Files: tubhiswt-3D/tubhiswt_C0.ome.tif, tubhiswt_C1.ome.tif
        - Samples (T): 20
        - Channels (C): 2 (one per file)
        - Depth planes (Z): 1
        - Frame shape: 512 x 512
        - Volumetric: False
        - Pages/IFDs: 20 per file
        """
        file_path = OME_TIFF_PATH / "tubhiswt-3D" / "tubhiswt_C0.ome.tif"

        extractor_ch0 = OMETiffImagingExtractor(file_path, sampling_frequency=1.0, channel_name="0")

        assert extractor_ch0.get_num_samples() == 20
        assert extractor_ch0.get_image_shape() == (512, 512)
        assert extractor_ch0.get_num_planes() == 1
        assert extractor_ch0.is_volumetric is False

        data_ch0 = extractor_ch0.get_series(0, 2)
        assert data_ch0.shape == (2, 512, 512)

        extractor_ch1 = OMETiffImagingExtractor(file_path, sampling_frequency=1.0, channel_name="1")
        data_ch1 = extractor_ch1.get_series(0, 2)
        assert data_ch1.shape == (2, 512, 512)

    def test_multi_channel_multi_plane(self):
        """Test with multi-channel, volumetric time series data across many files.

        Files: tubhiswt-4D/tubhiswt_C{0,1}_TP{0..42}.ome.tif (86 files)
        - Samples (T): 43
        - Channels (C): 2
        - Depth planes (Z): 10
        - Frame shape: 512 x 512
        - Volumetric: True
        - Pages/IFDs: 10 per file
        """
        file_path = OME_TIFF_PATH / "tubhiswt-4D" / "tubhiswt_C0_TP0.ome.tif"

        extractor = OMETiffImagingExtractor(file_path, sampling_frequency=1.0, channel_name="0")

        assert extractor.get_num_samples() == 43
        assert extractor.get_image_shape() == (512, 512)
        assert extractor.get_num_planes() == 10
        assert extractor.is_volumetric is True

        data = extractor.get_series(0, 1)
        assert data.shape == (1, 512, 512, 10)

    def test_single_channel_multi_plane(self):
        """Test with single-channel, volumetric time series data in a single file.

        File: bioformats-artificial/4D-series.ome.tiff
        - Samples (T): 7
        - Channels (C): 1
        - Depth planes (Z): 5
        - Frame shape: 167 x 439
        - Volumetric: True
        - Pages/IFDs: 35
        """
        file_path = OME_TIFF_PATH / "bioformats-artificial" / "4D-series.ome.tiff"

        extractor = OMETiffImagingExtractor(file_path, sampling_frequency=1.0)

        assert extractor.get_num_samples() == 7
        assert extractor.get_image_shape() == (167, 439)
        assert extractor.get_num_planes() == 5
        assert extractor.is_volumetric is True

        data = extractor.get_series(0, 2)
        assert data.shape == (2, 167, 439, 5)

    def test_data_values_and_channel_selection(self):
        """Test that pixel values are correctly mapped and channel selection works.

        File: dimension_order_tests/test_XYCZT.ome.tiff
        - Samples (T): 4
        - Channels (C): 2
        - Depth planes (Z): 3
        - Frame shape: 6 x 8
        - Volumetric: True
        - Pixel values encode: t * 100 + c * 10 + z
        - Pages/IFDs: 24
        """
        file_path = OME_TIFF_PATH / "dimension_order_tests" / "test_XYCZT.ome.tiff"

        # Channel 0
        extractor_ch0 = OMETiffImagingExtractor(file_path, sampling_frequency=1.0, channel_name="0")

        assert extractor_ch0.get_num_samples() == 4
        assert extractor_ch0.get_image_shape() == (6, 8)
        assert extractor_ch0.get_num_planes() == 3
        assert extractor_ch0.is_volumetric is True

        data_ch0 = extractor_ch0.get_series(0, 4)
        assert data_ch0.shape == (4, 6, 8, 3)

        for t in range(4):
            for z in range(3):
                expected_value = t * 100 + 0 * 10 + z
                assert_array_equal(data_ch0[t, :, :, z], expected_value)

        # Channel 1
        extractor_ch1 = OMETiffImagingExtractor(file_path, sampling_frequency=1.0, channel_name="1")
        data_ch1 = extractor_ch1.get_series(0, 4)

        for t in range(4):
            for z in range(3):
                expected_value = t * 100 + 1 * 10 + z
                assert_array_equal(data_ch1[t, :, :, z], expected_value)

    def test_subset_reading(self):
        """Test partial reads with start_sample and end_sample.

        File: dimension_order_tests/test_XYCZT.ome.tiff
        Reads samples 1 and 2 (out of 0-3) and verifies correct values.
        """
        file_path = OME_TIFF_PATH / "dimension_order_tests" / "test_XYCZT.ome.tiff"

        extractor = OMETiffImagingExtractor(file_path, sampling_frequency=1.0, channel_name="0")

        data = extractor.get_series(1, 3)
        assert data.shape == (2, 6, 8, 3)

        for sample_index, t in enumerate([1, 2]):
            for z in range(3):
                expected_value = t * 100 + 0 * 10 + z
                assert_array_equal(data[sample_index, :, :, z], expected_value)

    def test_channel_required_for_multi_channel(self):
        """Test that channel_name is required when the file has multiple channels."""
        file_path = OME_TIFF_PATH / "dimension_order_tests" / "test_XYCZT.ome.tiff"

        with pytest.raises(ValueError, match="channel_name must be specified"):
            OMETiffImagingExtractor(file_path, sampling_frequency=30.0)

    def test_invalid_channel_name(self):
        """Test that an invalid channel index raises an error."""
        file_path = OME_TIFF_PATH / "dimension_order_tests" / "test_XYCZT.ome.tiff"

        with pytest.raises(ValueError, match="channel_index .* is out of range"):
            OMETiffImagingExtractor(file_path, sampling_frequency=1.0, channel_name="5")

    def test_file_not_found(self):
        """Test that a FileNotFoundError is raised for a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            OMETiffImagingExtractor("/nonexistent/path.ome.tif", sampling_frequency=1.0)
