"""Tests for the OMETiffImagingExtractor using real OME-TIFF test data."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tifffile import TiffFile

from roiextractors import OMETiffImagingExtractor
from tests.setup_paths import OPHYS_DATA_PATH

OME_TIFF_PATH = OPHYS_DATA_PATH / "imaging_datasets" / "OME-TIFF"


@pytest.mark.skipif(not OME_TIFF_PATH.exists(), reason=f"Test data not found at {OME_TIFF_PATH}")
class TestOMETiffImagingExtractor:
    """Test the OMETiffImagingExtractor with various OME-TIFF datasets."""

    def test_single_channel_single_plane(self):
        """Test with single-channel, planar time series data in a single file.

        File: planar_single_channel_single_file/00001_01.ome.tiff
        Source: MitoCheck, stubbed from 93 to 5 frames, cropped to 64x64.
        - Samples (T): 5
        - Channels (C): 1
        - Depth planes (Z): 1
        - Frame shape: 64 x 64
        - Volumetric: False
        """
        file_path = OME_TIFF_PATH / "planar_single_channel_single_file" / "00001_01.ome.tiff"

        extractor = OMETiffImagingExtractor(file_path, sampling_frequency=1.0)

        assert extractor.get_num_samples() == 5
        assert extractor.get_image_shape() == (64, 64)
        assert extractor.get_num_planes() == 1
        assert extractor.is_volumetric is False

        with TiffFile(file_path) as tiff:
            tiff_data = np.stack([page.asarray() for page in tiff.pages])

        assert_array_equal(extractor.get_series(), tiff_data)

    def test_multi_channel_single_plane_single_file(self):
        """Test with multi-channel, planar time series data in a single file.

        File: planar_multi_channel_single_file/multi-channel-time-series.ome.tiff
        Source: bioformats-artificial, cropped to 64x64.
        - Samples (T): 7
        - Channels (C): 3
        - Depth planes (Z): 1
        - Frame shape: 64 x 64
        - Volumetric: False
        """
        file_path = OME_TIFF_PATH / "planar_multi_channel_single_file" / "multi-channel-time-series.ome.tiff"

        extractor = OMETiffImagingExtractor(file_path, sampling_frequency=1.0, channel_name="0")

        assert extractor.get_num_samples() == 7
        assert extractor.get_image_shape() == (64, 64)
        assert extractor.get_num_planes() == 1
        assert extractor.is_volumetric is False

        # Dimension order is XYZCT, so pages cycle C first then T.
        # Channel 0 pages are at indices 0, 3, 6, 9, 12, 15, 18
        with TiffFile(file_path) as tiff:
            all_pages = [page.asarray() for page in tiff.pages]
        channel_0_pages = np.stack(all_pages[0::3])

        assert_array_equal(extractor.get_series(), channel_0_pages)

    def test_multi_channel_single_plane_multi_file(self):
        """Test with multi-channel, planar time series data across multiple files.

        Files: planar_two_channels_multi_file/tubhiswt_C{0,1}.ome.tif
        Source: tubhiswt-3D, stubbed from 20 to 5 timepoints, cropped to 64x64.
        - Samples (T): 5
        - Channels (C): 2 (one per file)
        - Depth planes (Z): 1
        - Frame shape: 64 x 64
        - Volumetric: False
        """
        file_path = OME_TIFF_PATH / "planar_two_channels_multi_file" / "tubhiswt_C0.ome.tif"

        extractor_ch0 = OMETiffImagingExtractor(file_path, sampling_frequency=1.0, channel_name="0")

        assert extractor_ch0.get_num_samples() == 5
        assert extractor_ch0.get_image_shape() == (64, 64)
        assert extractor_ch0.get_num_planes() == 1
        assert extractor_ch0.is_volumetric is False

        with TiffFile(file_path) as tiff:
            tiff_data_ch0 = np.stack([page.asarray() for page in tiff.pages])
        assert_array_equal(extractor_ch0.get_series(), tiff_data_ch0)

        ch1_path = OME_TIFF_PATH / "planar_two_channels_multi_file" / "tubhiswt_C1.ome.tif"
        extractor_ch1 = OMETiffImagingExtractor(file_path, sampling_frequency=1.0, channel_name="1")

        with TiffFile(ch1_path) as tiff:
            tiff_data_ch1 = np.stack([page.asarray() for page in tiff.pages])
        assert_array_equal(extractor_ch1.get_series(), tiff_data_ch1)

    def test_single_channel_multi_plane(self):
        """Test with single-channel, volumetric time series data in a single file.

        File: volumetric_single_channel_single_file/4D-series.ome.tiff
        Source: bioformats-artificial, cropped to 64x64.
        - Samples (T): 7
        - Channels (C): 1
        - Depth planes (Z): 5
        - Frame shape: 64 x 64
        - Volumetric: True
        """
        file_path = OME_TIFF_PATH / "volumetric_single_channel_single_file" / "4D-series.ome.tiff"

        extractor = OMETiffImagingExtractor(file_path, sampling_frequency=1.0)

        assert extractor.get_num_samples() == 7
        assert extractor.get_image_shape() == (64, 64)
        assert extractor.get_num_planes() == 5
        assert extractor.is_volumetric is True

        # Dimension order XYZCT, single channel: pages are Z0T0, Z1T0, ..., Z4T0, Z0T1, ...
        # Each volume is 5 consecutive pages
        with TiffFile(file_path) as tiff:
            all_pages = [page.asarray() for page in tiff.pages]

        extractor_data = extractor.get_series()
        for t in range(7):
            for z in range(5):
                page_index = t * 5 + z
                assert_array_equal(extractor_data[t, :, :, z], all_pages[page_index])

    def test_multi_channel_multi_plane_single_file(self):
        """Test with multi-channel, volumetric time series data in a single file.

        File: volumetric_multi_channel_single_file/multi-channel-4D-series.ome.tiff
        Source: bioformats-artificial, cropped to 64x64.
        - Samples (T): 7
        - Channels (C): 3
        - Depth planes (Z): 5
        - Frame shape: 64 x 64
        - Volumetric: True
        """
        file_path = OME_TIFF_PATH / "volumetric_multi_channel_single_file" / "multi-channel-4D-series.ome.tiff"

        extractor = OMETiffImagingExtractor(file_path, sampling_frequency=1.0, channel_name="0")

        assert extractor.get_num_samples() == 7
        assert extractor.get_image_shape() == (64, 64)
        assert extractor.get_num_planes() == 5
        assert extractor.is_volumetric is True

        # Dimension order XYZCT: pages cycle Z first, then C, then T
        # Channel 0 volumes: for each T, pages at Z*3+0 (channel 0 stride)
        with TiffFile(file_path) as tiff:
            all_pages = [page.asarray() for page in tiff.pages]

        extractor_data = extractor.get_series()
        pages_per_volume = 5 * 3  # Z * C
        for t in range(7):
            for z in range(5):
                page_index = t * pages_per_volume + z * 3 + 0  # channel 0
                assert_array_equal(extractor_data[t, :, :, z], all_pages[page_index])

    def test_multi_channel_multi_plane_multi_file(self):
        """Test with multi-channel, volumetric time series data across many files.

        Files: volumetric_two_channels_multi_file/tubhiswt_C{0,1}_TP{0,1,2}.ome.tif (6 files)
        Source: tubhiswt-4D, stubbed from 43 to 3 timepoints, cropped to 64x64.
        - Samples (T): 3
        - Channels (C): 2
        - Depth planes (Z): 10
        - Frame shape: 64 x 64
        - Volumetric: True
        """
        file_path = OME_TIFF_PATH / "volumetric_two_channels_multi_file" / "tubhiswt_C0_TP0.ome.tif"

        extractor = OMETiffImagingExtractor(file_path, sampling_frequency=1.0, channel_name="0")

        assert extractor.get_num_samples() == 3
        assert extractor.get_image_shape() == (64, 64)
        assert extractor.get_num_planes() == 10
        assert extractor.is_volumetric is True

        # Each C0 file has 10 pages (Z-planes for one timepoint)
        for tp in range(3):
            tp_path = OME_TIFF_PATH / "volumetric_two_channels_multi_file" / f"tubhiswt_C0_TP{tp}.ome.tif"
            with TiffFile(tp_path) as tiff:
                tiff_data = np.stack([page.asarray() for page in tiff.pages])

            extractor_volume = extractor.get_series(tp, tp + 1)
            # extractor returns (1, Y, X, Z), tiff_data is (Z, Y, X)
            extractor_planes = np.moveaxis(extractor_volume[0], -1, 0)
            assert_array_equal(extractor_planes, tiff_data)

    def test_sampling_frequency_from_time_increment(self):
        """Test that sampling_frequency is derived from TimeIncrement when not provided.

        File: planar_single_channel_single_file_with_time_increment/00001_01.ome.tiff
        Source: Same as planar_single_channel_single_file but with TimeIncrement="0.1" added
        to the Pixels element (0.1 seconds = 10 Hz).
        - Samples (T): 5
        - Channels (C): 1
        - Depth planes (Z): 1
        - Frame shape: 64 x 64
        - TimeIncrement: 0.1 s
        """
        file_path = OME_TIFF_PATH / "planar_single_channel_single_file_with_time_increment" / "00001_01.ome.tiff"

        extractor = OMETiffImagingExtractor(file_path)

        assert extractor.get_sampling_frequency() == 10.0
        assert extractor.get_num_samples() == 5
        assert extractor.get_image_shape() == (64, 64)

    def test_sampling_frequency_required_when_no_time_increment(self):
        """Test that ValueError is raised when neither sampling_frequency nor TimeIncrement is available."""
        file_path = OME_TIFF_PATH / "planar_single_channel_single_file" / "00001_01.ome.tiff"

        with pytest.raises(ValueError, match="sampling_frequency must be provided"):
            OMETiffImagingExtractor(file_path)

    def test_sampling_frequency_overrides_time_increment(self):
        """Test that an explicit sampling_frequency takes precedence over TimeIncrement."""
        file_path = OME_TIFF_PATH / "planar_single_channel_single_file_with_time_increment" / "00001_01.ome.tiff"

        extractor = OMETiffImagingExtractor(file_path, sampling_frequency=30.0)

        assert extractor.get_sampling_frequency() == 30.0

    def test_file_not_found(self):
        """Test that a FileNotFoundError is raised for a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            OMETiffImagingExtractor("/nonexistent/path.ome.tif", sampling_frequency=1.0)
