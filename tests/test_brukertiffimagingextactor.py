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

    def test_init_opens_only_the_shape_probe_tiff(self):
        """Construction must not open every TIFF just to count pages.

        The point of the Bruker ``_get_ifds_per_file()`` override is to read per-file page
        counts from the configuration XML instead of opening each file. The base class still
        opens exactly one file to probe frame shape and dtype, so a full construction must open
        ``TiffFile`` exactly once regardless of file count.

        ``num_samples`` and pixel data are identical whether the counts come from the XML or
        from opening every file, so the test measures the file opens directly. ``tifffile.TiffFile``
        is the call that opens a TIFF on disk. ``patch("tifffile.TiffFile", ...)`` swaps it for a
        spy for the duration of the ``with`` block and restores the original on exit; the spy
        intercepts every call the extractor makes to it. ``wraps=tifffile.TiffFile`` makes the spy
        forward each call to the real opener and return its real result, so the shape/dtype probe
        still gets a genuine file handle while the spy records the call. ``call_count`` is then the
        number of files opened during construction, which must be exactly one.

        This stub has 10 files; without the override, init would open 11 (one probe plus one per
        file to count pages). Pinning the count to one catches a regression back to per-file opens.
        """
        from unittest.mock import patch

        with patch("tifffile.TiffFile", wraps=tifffile.TiffFile) as mock_tiff_file:
            BrukerTiffImagingExtractor(folder_path=self.folder_path)

        assert mock_tiff_file.call_count == 1


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
    Channel names ("Green", "Red") are user-set fluorophore labels from the Bruker
    configuration XML, distinct from the generic Ch1/Ch2 in OME-XML and in the file
    naming convention.
    """

    folder_path = BRUKER_STUB_PATH / "TSeries-20240527-001"

    def test_dual_channel(self):
        """Test channel selection and data access for dual-channel Bruker data.

        Stub: TSeries-20240527-001
        - Samples (T): 5
        - Channels (C): 2 (Green, Red)
        - Depth planes (Z): 1
        - Frame shape: 64 x 64
        - Files: 10 .ome.tif (5 timepoints x 2 channels)
        """
        with pytest.raises(ValueError, match="channel_name must be specified"):
            BrukerTiffImagingExtractor(folder_path=self.folder_path)

        ext_green = BrukerTiffImagingExtractor(folder_path=self.folder_path, channel_name="Green")
        assert ext_green.get_image_shape() == (64, 64)
        assert ext_green.get_num_samples() == 5
        assert ext_green.get_dtype() == np.uint16
        assert ext_green.is_volumetric is False

        ext_red = BrukerTiffImagingExtractor(folder_path=self.folder_path, channel_name="Red")
        assert ext_red.get_num_samples() == 5

        # Verify data values match the raw TIFF files per channel. The user-set labels do
        # NOT match the generic Ch1/Ch2 file naming: in this recording channel 1 (the
        # *_Ch1_* files) is "Red" and channel 2 (the *_Ch2_* files) is "Green". The
        # extractor must map each label to its real files (alphabetical name order would
        # swap them).
        ch1_files = sorted(self.folder_path.glob("*_Ch1_*.ome.tif"))  # channel 1 = Red
        ch2_files = sorted(self.folder_path.glob("*_Ch2_*.ome.tif"))  # channel 2 = Green
        expected_red = np.stack([tifffile.imread(f) for f in ch1_files])
        expected_green = np.stack([tifffile.imread(f) for f in ch2_files])
        assert_array_equal(ext_green.get_series(), expected_green)
        assert_array_equal(ext_red.get_series(), expected_red)
        assert not np.array_equal(expected_green, expected_red)

    def test_invalid_channel_raises(self):
        with pytest.raises(ValueError, match="Channel '5' not found.*Available channels"):
            BrukerTiffImagingExtractor(folder_path=self.folder_path, channel_name="5")

    def test_timestamps_and_sampling_frequency(self):
        """Timestamps are the relativeTime values from the Bruker configuration XML.

        After init, get_timestamps() should return the native timestamps (not values
        recomputed from sampling_frequency). The sampling frequency is derived from
        these timestamps.
        """
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path, channel_name="Green")

        # Hardcoded relativeTime values extracted from the stub's Bruker XML
        expected_timestamps = np.array([0.0, 0.00474538, 0.00949076, 0.01423614, 0.01898152])
        assert_array_equal(extractor.get_timestamps(), extractor.get_native_timestamps())
        np.testing.assert_allclose(extractor.get_timestamps(), expected_timestamps, atol=1e-6)
        expected_frequency = 1.0 / np.mean(np.diff(expected_timestamps))
        assert extractor.get_sampling_frequency() == pytest.approx(expected_frequency, rel=1e-4)


class TestBrukerTiffImagingExtractorBinaryOnlyOMEXMLNoCompanion:
    """Test BrukerTiffImagingExtractor on PV 5.7+ BinaryOnly OME-XML without the companion sidecar.

    A dual-channel recording where each ``.ome.tif``'s embedded OME-XML is only a
    ``<BinaryOnly MetadataFile="...companion.ome" .../>`` stub rather than a full ``<Pixels>``
    block, and the companion sidecar is absent. With no ``<Pixels>`` available anywhere, the
    extractor must read all structural metadata from the Bruker configuration XML; this is
    the case that fails if anything still depends on OME-XML.
    """

    folder_path = BRUKER_STUB_PATH / "NCCR62_2023_07_06_IntoTheVoid_t_series_Dual_color-000"

    def test_construct_and_select_channels(self):
        """Both channels must be selectable and return distinct data."""
        with pytest.raises(ValueError, match="channel_name must be specified"):
            BrukerTiffImagingExtractor(folder_path=self.folder_path)

        ext_ch1 = BrukerTiffImagingExtractor(folder_path=self.folder_path, channel_name="Ch1")
        assert ext_ch1.get_num_samples() == 10
        assert ext_ch1.get_image_shape() == (512, 512)
        assert ext_ch1.get_dtype() == np.uint16
        assert ext_ch1.is_volumetric is False

        ext_ch2 = BrukerTiffImagingExtractor(folder_path=self.folder_path, channel_name="Ch2")
        assert ext_ch2.get_num_samples() == 10

        # Each channel must return exactly the frames from its own file, in order. Here each
        # channel is stored as a single multi-page file (page = timepoint), so the extractor
        # must map Ch1/Ch2 to the correct *_Ch1_*/*_Ch2_* files rather than mixing pages
        # across them. A single imread returns the full (num_samples, height, width) stack.
        expected_ch1 = tifffile.imread(next(self.folder_path.glob("*_Ch1_*.ome.tif")))
        expected_ch2 = tifffile.imread(next(self.folder_path.glob("*_Ch2_*.ome.tif")))
        assert_array_equal(ext_ch1.get_series(), expected_ch1)
        assert_array_equal(ext_ch2.get_series(), expected_ch2)
        assert not np.array_equal(expected_ch1, expected_ch2)


class TestBrukerTiffImagingExtractorMultiSequenceBOT:
    """Multi-burst Brightness Over Time recording: relativeTime resets per <Sequence>.

    The recording is not uniformly sampled (real gaps between bursts), so the extractor must
    construct (not raise "Could not determine sampling frequency"), report the within-burst
    frame rate as sampling_frequency, and expose the true gapped timeline via get_timestamps().
    """

    folder_path = BRUKER_STUB_PATH / "TSeries-02022026-001_multisequence"

    def test_multisequence_within_burst_rate_and_gapped_timestamps(self):
        with pytest.warns(UserWarning, match="multiple <Sequence>"):
            extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path, channel_name="Ch2")

        assert extractor.get_num_samples() == 15  # 3 bursts x 5 frames, single channel
        assert extractor.get_image_shape() == (64, 64)
        assert extractor.is_volumetric is False
        # within-burst frame rate (no single global rate exists across the gaps)
        assert extractor.get_sampling_frequency() == pytest.approx(15.02, rel=1e-3)

        # true timeline: regular within each burst, real inter-burst gaps preserved
        assert extractor.has_time_vector()
        timestamps = extractor.get_timestamps()
        assert len(timestamps) == 15
        assert timestamps[0] == pytest.approx(0.0, abs=1e-6)
        assert timestamps[1] - timestamps[0] == pytest.approx(0.0666, rel=1e-2)  # within burst
        assert timestamps[5] - timestamps[4] == pytest.approx(10.34, rel=1e-2)  # burst 1 -> 2
        assert timestamps[10] - timestamps[9] == pytest.approx(10.13, rel=1e-2)  # burst 2 -> 3


class TestBrukerTiffImagingExtractorMultiSequenceTimedElement:
    """Multi-cycle Timed Element recording (non-BOT) that also resets relativeTime per <Sequence>.

    Confirms the multi-sequence sampling-frequency bug is not specific to Brightness Over Time.
    "Timed Element" means the frames within a cycle are regularly timed, not the cycles, which are
    irregularly spaced in this recording, so the inter-cycle gaps differ.
    """

    folder_path = BRUKER_STUB_PATH / "TSeries-08162024-1918-002_multisequence"

    def test_multisequence_timed_element_within_cycle_rate_and_gapped_timestamps(self):
        with pytest.warns(UserWarning, match="multiple <Sequence>"):
            extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)

        assert extractor.get_num_samples() == 9  # 3 cycles x 3 frames, single channel
        assert extractor.get_image_shape() == (64, 64)
        assert extractor.is_volumetric is False
        assert extractor.get_sampling_frequency() == pytest.approx(15.11, rel=1e-3)

        assert extractor.has_time_vector()
        timestamps = extractor.get_timestamps()
        assert len(timestamps) == 9
        assert timestamps[0] == pytest.approx(0.0, abs=1e-6)
        assert timestamps[1] - timestamps[0] == pytest.approx(0.0662, rel=1e-2)  # within cycle
        # the cycles are irregularly spaced in this recording (real, not uniform)
        assert timestamps[3] - timestamps[2] == pytest.approx(25.72, rel=1e-2)  # cycle 0 -> 1
        assert timestamps[6] - timestamps[5] == pytest.approx(7.70, rel=1e-2)  # cycle 1 -> 2


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
