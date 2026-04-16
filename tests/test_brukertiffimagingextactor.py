import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hdmf.testing import TestCase
from natsort import natsorted
from numpy.testing import assert_array_equal
from tifffile import tifffile

from roiextractors import (
    BrukerTiffImagingExtractor,
    BrukerTiffMultiPlaneImagingExtractor,
    BrukerTiffSinglePlaneImagingExtractor,
)

from .setup_paths import OPHYS_DATA_PATH


def _get_test_video(file_paths):
    frames = [
        tifffile.memmap(
            file,
            mode="r",
            _multifile=False,
        )
        for file in file_paths
    ]
    return np.stack(frames, axis=0)


class TestBrukerTiffExtractorSinglePlaneCase(TestCase):
    @classmethod
    def setUpClass(cls):
        folder_path = str(
            OPHYS_DATA_PATH / "imaging_datasets" / "BrukerTif" / "NCCR32_2023_02_20_Into_the_void_t_series_baseline-000"
        )
        cls.available_streams = dict(channel_streams=["Ch2"])

        cls.folder_path = folder_path
        extractor = BrukerTiffSinglePlaneImagingExtractor(folder_path=folder_path)
        cls.extractor = extractor

        file_paths = natsorted(Path(folder_path).glob("*.ome.tif"))
        cls.video = _get_test_video(file_paths=file_paths)

        # temporary directory for testing assertion when xml file is missing
        test_dir = tempfile.mkdtemp()
        cls.test_dir = test_dir
        shutil.copy(file_paths[0], Path(test_dir) / file_paths[0].name)

    @classmethod
    def tearDownClass(cls):
        # remove the temporary directory and its contents
        shutil.rmtree(cls.test_dir)

    def test_stream_names(self):
        self.assertEqual(
            BrukerTiffSinglePlaneImagingExtractor.get_streams(folder_path=self.folder_path), self.available_streams
        )

    def test_incorrect_stream_name_raises(self):
        exc_msg = f"The selected stream 'Ch1' is not in the available channel_streams '['Ch2']'!"
        with self.assertRaisesWith(ValueError, exc_msg=exc_msg):
            BrukerTiffSinglePlaneImagingExtractor(folder_path=self.folder_path, stream_name="Ch1")

    def test_tif_files_are_missing_assertion(self):
        folder_path = "not a tiff path"
        exc_msg = f"The TIF image files are missing from '{folder_path}'."
        with self.assertRaisesWith(AssertionError, exc_msg=exc_msg):
            BrukerTiffSinglePlaneImagingExtractor(folder_path=folder_path)

    def test_volumetric_extractor_cannot_be_used_for_non_volumetric_data(self):
        exc_msg = "BrukerTiffMultiPlaneImagingExtractor is for volumetric imaging. For single imaging plane data use BrukerTiffSinglePlaneImagingExtractor."
        with self.assertRaisesWith(AssertionError, exc_msg=exc_msg):
            BrukerTiffMultiPlaneImagingExtractor(folder_path=self.folder_path)

    def test_xml_configuration_file_is_missing_assertion(self):
        folder_path = self.test_dir
        exc_msg = f"The XML configuration file is not found at '{folder_path}'."
        with self.assertRaisesWith(AssertionError, exc_msg=exc_msg):
            BrukerTiffSinglePlaneImagingExtractor(folder_path=folder_path)

    def test_brukertiffextractor_image_size(self):
        self.assertEqual(self.extractor.get_image_shape(), (64, 64))

    def test_brukertiffextractor_num_frames(self):
        self.assertEqual(self.extractor.get_num_samples(), 10)

    def test_brukertiffextractor_sampling_frequency(self):
        self.assertEqual(self.extractor.get_sampling_frequency(), 29.873732099062256)

    def test_brukertiffextractor_channel_names(self):
        self.assertEqual(self.extractor.get_channel_names(), ["Ch2"])

    def test_brukertiffextractor_dtype(self):
        self.assertEqual(self.extractor.get_dtype(), np.uint16)

    def test_brukertiffextractor_get_series(self):
        series = self.extractor.get_series()
        assert_array_equal(series, self.video)
        self.assertEqual(series.dtype, np.uint16)
        assert_array_equal(self.extractor.get_series(start_sample=0, end_sample=1), self.video[:1])


class TestBrukerTiffExtractorDualPlaneCase(TestCase):
    @classmethod
    def setUpClass(cls):
        folder_path = str(
            OPHYS_DATA_PATH / "imaging_datasets" / "BrukerTif" / "NCCR32_2022_11_03_IntoTheVoid_t_series-005"
        )
        cls.folder_path = folder_path
        cls.extractor = BrukerTiffMultiPlaneImagingExtractor(folder_path=folder_path)

        first_plane_file_paths = [
            f"{cls.folder_path}/NCCR32_2022_11_03_IntoTheVoid_t_series-005_Cycle0000{num + 1}_Ch2_000001.ome.tif"
            for num in range(5)
        ]
        second_plane_file_paths = [
            f"{cls.folder_path}/NCCR32_2022_11_03_IntoTheVoid_t_series-005_Cycle0000{num + 1}_Ch2_000002.ome.tif"
            for num in range(5)
        ]

        cls.available_streams = dict(
            channel_streams=["Ch2"],
            plane_streams=dict(Ch2=["Ch2_000001", "Ch2_000002"]),
        )
        cls.test_video = np.zeros((5, 64, 64, 2), dtype=np.uint16)
        first_plane_video = _get_test_video(file_paths=first_plane_file_paths)
        cls.test_video[..., 0] = first_plane_video
        second_plane_video = _get_test_video(file_paths=second_plane_file_paths)
        cls.test_video[..., 1] = second_plane_video

    def test_stream_names(self):
        found_streams = BrukerTiffMultiPlaneImagingExtractor.get_streams(folder_path=self.folder_path)
        expected_streams = self.available_streams
        self.assertEqual(found_streams, expected_streams)

    def test_brukertiffextractor_image_size(self):
        self.assertEqual(self.extractor.get_image_shape(), (64, 64))

    def test_brukertiffextractor_num_frames(self):
        self.assertEqual(self.extractor.get_num_samples(), 5)

    def test_brukertiffextractor_sampling_frequency(self):
        self.assertEqual(self.extractor.get_sampling_frequency(), 20.629515014336377)

    def test_brukertiffextractor_channel_names(self):
        self.assertEqual(self.extractor.get_channel_names(), ["Ch2"])

    def test_brukertiffextractor_dtype(self):
        self.assertEqual(self.extractor.get_dtype(), np.uint16)

    def test_incorrect_stream_with_disjoint_plane_raises(self):
        exc_msg = (
            "The selected stream 'Ch2_000003' is not in the available plane_streams '['Ch2_000001', 'Ch2_000002']'!"
        )
        with self.assertRaisesWith(ValueError, exc_msg=exc_msg):
            BrukerTiffMultiPlaneImagingExtractor(
                folder_path=self.folder_path,
                stream_name="Ch2_000003",
            )

    def test_brukertiffextractor_get_series(self):
        series = self.extractor.get_series()
        assert_array_equal(series, self.test_video)
        self.assertEqual(series.dtype, np.uint16)
        assert_array_equal(self.extractor.get_series(start_sample=2, end_sample=4), self.test_video[2:4])

    def test_is_volumetric_flag(self):
        """Test that the is_volumetric flag is True for BrukerTiffMultiPlaneImagingExtractor."""
        assert hasattr(
            self.extractor, "is_volumetric"
        ), "BrukerTiffMultiPlaneImagingExtractor should have is_volumetric attribute"
        assert (
            self.extractor.is_volumetric is True
        ), "is_volumetric should be True for BrukerTiffMultiPlaneImagingExtractor"

    def test_get_volume_shape(self):
        """Test that the get_volume_shape method returns the correct shape."""
        # Check that the method exists
        assert hasattr(
            self.extractor, "get_volume_shape"
        ), "BrukerTiffMultiPlaneImagingExtractor should have get_volume_shape method"

        # Check that the method returns the correct shape
        image_shape = self.extractor.get_image_shape()
        num_planes = self.extractor.get_num_planes()
        volume_shape = self.extractor.get_volume_shape()

        assert len(volume_shape) == 3, "get_volume_shape should return a 3-tuple"
        assert volume_shape == (
            image_shape[0],
            image_shape[1],
            num_planes,
        ), "get_volume_shape should return (num_rows, num_columns, num_planes)"


class TestBrukerTiffExtractorDualColorCase(TestCase):
    @classmethod
    def setUpClass(cls):
        folder_path = str(
            OPHYS_DATA_PATH / "imaging_datasets" / "BrukerTif" / "NCCR62_2023_07_06_IntoTheVoid_t_series_Dual_color-000"
        )
        cls.folder_path = folder_path
        cls.available_streams = dict(channel_streams=["Ch1", "Ch2"])
        cls.extractor = BrukerTiffSinglePlaneImagingExtractor(folder_path=cls.folder_path, stream_name="Ch1")

        file_paths = natsorted(Path(folder_path).glob("*.ome.tif"))
        cls.test_video_ch1 = tifffile.TiffFile(file_paths[0]).asarray()
        cls.test_video_ch2 = tifffile.TiffFile(file_paths[1]).asarray()

    def test_not_selecting_stream_raises(self):
        exc_msg = "More than one recording stream is detected! Please specify which stream you wish to load with the `stream_name` argument. To see what streams are available, call `BrukerTiffSinglePlaneImagingExtractor.get_stream_names(folder_path=...)`."
        with self.assertRaisesWith(ValueError, exc_msg=exc_msg):
            BrukerTiffSinglePlaneImagingExtractor(folder_path=self.folder_path)

    def test_stream_names(self):
        assert_array_equal(
            BrukerTiffSinglePlaneImagingExtractor.get_streams(folder_path=self.folder_path), self.available_streams
        )

    def test_brukertiffextractor_image_size(self):
        self.assertEqual(self.extractor.get_image_shape(), (512, 512))

    def test_brukertiffextractor_channel_names(self):
        self.assertEqual(self.extractor.get_channel_names(), ["Ch1"])

    def test_brukertiffextractor_dtype(self):
        self.assertEqual(self.extractor.get_dtype(), np.uint16)

    def test_brukertiffextractor_sampling_frequency(self):
        self.assertEqual(self.extractor.get_sampling_frequency(), 29.873615189896864)

    def test_brukertiffextractor_get_series(self):
        assert_array_equal(self.extractor.get_series(start_sample=0, end_sample=1), self.test_video_ch1[:1])
        series = self.extractor.get_series()
        assert_array_equal(series, self.test_video_ch1)
        self.assertEqual(series.dtype, np.uint16)

    def test_brukertiffextractor_second_stream_get_series(self):
        extractor = BrukerTiffSinglePlaneImagingExtractor(folder_path=self.folder_path, stream_name="Ch2")
        series = extractor.get_series()
        assert_array_equal(extractor.get_series(), self.test_video_ch2)
        self.assertEqual(series.dtype, np.uint16)

    def test_brukertiffextractor_second_stream_sampling_frequency(self):
        extractor = BrukerTiffSinglePlaneImagingExtractor(folder_path=self.folder_path, stream_name="Ch2")
        self.assertEqual(
            self.extractor.get_sampling_frequency(),
            extractor.get_sampling_frequency(),
        )


BRUKER_STUB_PATH = OPHYS_DATA_PATH / "imaging_datasets" / "BrukerTif"


class TestBrukerTiffImagingExtractorSinglePlane:
    """Test BrukerTiffImagingExtractor with single-plane, single-channel data.

    Uses the NCCR32_2023 stub: 10 timepoints, 1 channel (Ch2), 1 plane, 64x64 frames.
    """

    folder_path = BRUKER_STUB_PATH / "NCCR32_2023_02_20_Into_the_void_t_series_baseline-000"

    def test_single_plane_single_channel(self):
        """Test basic metadata and data access for single-plane, single-channel Bruker data.

        Stub: NCCR32_2023_02_20_Into_the_void_t_series_baseline-000
        - Samples (T): 10
        - Channels (C): 1 (Ch2)
        - Depth planes (Z): 1
        - Frame shape: 64 x 64
        - Files: 10 .ome.tif
        """
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)

        assert extractor.get_image_shape() == (64, 64)
        assert extractor.get_sample_shape() == (64, 64)
        assert extractor.get_num_samples() == 10
        assert extractor.get_dtype() == np.uint16
        assert extractor.is_volumetric is False

        # Verify data values match the raw TIFF files
        file_paths = sorted(self.folder_path.glob("*.ome.tif"))
        expected_data = np.stack([tifffile.imread(f) for f in file_paths], axis=0)
        assert_array_equal(extractor.get_series(), expected_data)

    def test_timestamps_and_sampling_frequency(self):
        """Timestamps are the relativeTime values from the Bruker configuration XML.

        After init, get_timestamps() should return the native timestamps (not values
        recomputed from sampling_frequency). The sampling frequency is derived from
        these timestamps via calculate_regular_series_rate.
        """
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)

        # Hardcoded relativeTime values extracted from the stub's Bruker XML
        expected_timestamps = np.array(
            [
                0.0,
                0.03347422,
                0.06694845,
                0.10042267,
                0.1338969,
                0.16737112,
                0.20084534,
                0.23431957,
                0.26779379,
                0.30126802,
            ]
        )
        assert_array_equal(extractor.get_timestamps(), extractor.get_native_timestamps())
        np.testing.assert_allclose(extractor.get_timestamps(), expected_timestamps, atol=1e-6)
        expected_frequency = 1.0 / np.mean(np.diff(expected_timestamps))
        assert extractor.get_sampling_frequency() == pytest.approx(expected_frequency, rel=1e-4)


class TestBrukerTiffImagingExtractorVolumetric:
    """Test BrukerTiffImagingExtractor with volumetric, single-channel data.

    Uses the NCCR32_2022 stub: 5 timepoints, 1 channel (Ch2), 2 planes, 64x64 frames.
    """

    folder_path = BRUKER_STUB_PATH / "NCCR32_2022_11_03_IntoTheVoid_t_series-005"

    def test_volumetric_single_channel(self):
        """Test metadata and data access for volumetric, single-channel Bruker data.

        Stub: NCCR32_2022_11_03_IntoTheVoid_t_series-005
        - Samples (T): 5
        - Channels (C): 1 (Ch2)
        - Depth planes (Z): 2
        - Frame shape: 64 x 64
        - Files: 10 .ome.tif (5 cycles x 2 planes)
        """
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)

        assert extractor.is_volumetric is True
        assert extractor.get_num_samples() == 5
        assert extractor.get_num_planes() == 2
        assert extractor.get_volume_shape() == (64, 64, 2)
        assert extractor.get_sample_shape() == (64, 64, 2)

        # Verify data values: files are ordered (t0_z0, t0_z1, t1_z0, ...).
        # Reshape to (T, Z, H, W) then transpose to (T, H, W, Z) to match get_series().
        file_paths = sorted(self.folder_path.glob("*.ome.tif"))
        all_frames = np.stack([tifffile.imread(f) for f in file_paths])
        expected_data = all_frames.reshape(5, 2, 64, 64).transpose(0, 2, 3, 1)
        assert_array_equal(extractor.get_series(), expected_data)

    def test_timestamps_and_sampling_frequency(self):
        """Timestamps are subsampled by num_planes to give one per volume.

        The Bruker XML has one relativeTime per frame (plane). For volumetric data
        with 2 planes, every other timestamp is taken so get_timestamps() has one
        entry per volume. The sampling frequency is the volume rate, roughly half
        the per-frame rate.
        """
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)

        # Hardcoded relativeTime values extracted from the stub's Bruker XML (every 2nd, one per volume)
        expected_timestamps = np.array([0.0, 0.09694847, 0.19389695, 0.29084542, 0.3877939])
        assert_array_equal(extractor.get_timestamps(), extractor.get_native_timestamps())
        np.testing.assert_allclose(extractor.get_timestamps(), expected_timestamps, atol=1e-6)
        expected_frequency = 1.0 / np.mean(np.diff(expected_timestamps))
        assert extractor.get_sampling_frequency() == pytest.approx(expected_frequency, rel=1e-4)


class TestBrukerTiffImagingExtractorDualChannel:
    """Test BrukerTiffImagingExtractor with single-plane, dual-channel data.

    Uses the TSeries-20240527-001 stub: 5 timepoints, 2 channels, 1 plane, 64x64 frames.
    """

    folder_path = BRUKER_STUB_PATH / "TSeries-20240527-001"

    def test_dual_channel(self):
        """Test channel selection and data access for dual-channel Bruker data.

        Stub: TSeries-20240527-001
        - Samples (T): 5
        - Channels (C): 2 (Ch1, Ch2)
        - Depth planes (Z): 1
        - Frame shape: 64 x 64
        - Files: 10 .ome.tif (5 timepoints x 2 channels)
        """
        with pytest.raises(ValueError, match="channel_name must be specified"):
            BrukerTiffImagingExtractor(folder_path=self.folder_path)

        ext_ch0 = BrukerTiffImagingExtractor(folder_path=self.folder_path, channel_name="0")
        assert ext_ch0.get_image_shape() == (64, 64)
        assert ext_ch0.get_num_samples() == 5
        assert ext_ch0.get_dtype() == np.uint16
        assert ext_ch0.is_volumetric is False

        ext_ch1 = BrukerTiffImagingExtractor(folder_path=self.folder_path, channel_name="1")
        assert ext_ch1.get_num_samples() == 5

        # Verify data values match the raw TIFF files per channel
        ch1_files = sorted(self.folder_path.glob("*_Ch1_*.ome.tif"))
        ch2_files = sorted(self.folder_path.glob("*_Ch2_*.ome.tif"))
        expected_ch0 = np.stack([tifffile.imread(f) for f in ch1_files])
        expected_ch1 = np.stack([tifffile.imread(f) for f in ch2_files])
        assert_array_equal(ext_ch0.get_series(), expected_ch0)
        assert_array_equal(ext_ch1.get_series(), expected_ch1)
        assert not np.array_equal(expected_ch0, expected_ch1)

    def test_invalid_channel_raises(self):
        with pytest.raises(ValueError, match="channel_index .* is out of range"):
            BrukerTiffImagingExtractor(folder_path=self.folder_path, channel_name="5")

    def test_timestamps_and_sampling_frequency(self):
        """Timestamps are the relativeTime values from the Bruker configuration XML.

        After init, get_timestamps() should return the native timestamps (not values
        recomputed from sampling_frequency). The sampling frequency is derived from
        these timestamps.
        """
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path, channel_name="0")

        # Hardcoded relativeTime values extracted from the stub's Bruker XML
        expected_timestamps = np.array([0.0, 0.00474538, 0.00949076, 0.01423614, 0.01898152])
        assert_array_equal(extractor.get_timestamps(), extractor.get_native_timestamps())
        np.testing.assert_allclose(extractor.get_timestamps(), expected_timestamps, atol=1e-6)
        expected_frequency = 1.0 / np.mean(np.diff(expected_timestamps))
        assert extractor.get_sampling_frequency() == pytest.approx(expected_frequency, rel=1e-4)


@pytest.mark.skip(reason="No dual-channel volumetric Bruker test data available")
class TestBrukerTiffImagingExtractorDualChannelVolumetric:
    """TODO: Test BrukerTiffImagingExtractor with dual-channel volumetric data.

    This combination (num_channels > 1 and num_planes > 1) is untested because
    no sample data exists. The extractor emits a UserWarning for this case.
    If you have this type of data, please open an issue at
    https://github.com/catalystneuro/roiextractors/issues
    """

    pass


class TestBrukerTiffImagingExtractorErrors:
    """Test error handling for BrukerTiffImagingExtractor."""

    def test_no_ome_tif_files(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .ome.tif files found"):
            BrukerTiffImagingExtractor(folder_path=tmp_path)

    def test_pre_51_format_raises(self, tmp_path):
        """Plain .tif files (no .ome.tif) indicate pre-5.1 Prairie View data."""
        (tmp_path / "Image_001.tif").touch()
        with pytest.raises(ValueError, match="No Bruker extractor in roiextractors supports pre-5.1 data"):
            BrukerTiffImagingExtractor(folder_path=tmp_path)

    def test_missing_bruker_xml(self, tmp_path):
        dummy = tmp_path / "dummy.ome.tif"
        dummy.touch()
        with pytest.raises(FileNotFoundError, match="Bruker XML configuration file not found"):
            BrukerTiffImagingExtractor(folder_path=tmp_path)
