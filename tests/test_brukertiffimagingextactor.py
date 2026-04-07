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
        self.assertEqual(self.extractor.get_image_shape(), (512, 512))

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
        cls.test_video = np.zeros((5, 512, 512, 2), dtype=np.uint16)
        first_plane_video = _get_test_video(file_paths=first_plane_file_paths)
        cls.test_video[..., 0] = first_plane_video
        second_plane_video = _get_test_video(file_paths=second_plane_file_paths)
        cls.test_video[..., 1] = second_plane_video

    def test_stream_names(self):
        found_streams = BrukerTiffMultiPlaneImagingExtractor.get_streams(folder_path=self.folder_path)
        expected_streams = self.available_streams
        self.assertEqual(found_streams, expected_streams)

    def test_brukertiffextractor_image_size(self):
        self.assertEqual(self.extractor.get_image_shape(), (512, 512))

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
    """Test BrukerTiffImagingExtractor with single-plane, single-channel stub data."""

    folder_path = BRUKER_STUB_PATH / "single_plane_single_channel"

    def test_image_shape(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        assert extractor.get_image_shape() == (6, 8)

    def test_num_samples(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        assert extractor.get_num_samples() == 4

    def test_sampling_frequency(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        assert abs(extractor.get_sampling_frequency() - 1.0 / 0.033) < 0.1

    def test_dtype(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        assert extractor.get_dtype() == np.uint16

    def test_not_volumetric(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        assert extractor.is_volumetric is False

    def test_get_series(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        data = extractor.get_series(0, 4)
        assert data.shape == (4, 6, 8)
        for t in range(4):
            assert_array_equal(data[t, :, :], t * 100)

    def test_get_series_subset(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        data = extractor.get_series(1, 3)
        assert data.shape == (2, 6, 8)
        assert_array_equal(data[0, :, :], 100)
        assert_array_equal(data[1, :, :], 200)

    def test_xml_metadata(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        assert isinstance(extractor.xml_metadata, dict)
        assert "linesPerFrame" in extractor.xml_metadata
        assert "pixelsPerLine" in extractor.xml_metadata

    def test_isinstance_ome(self):
        from roiextractors import OMETiffImagingExtractor

        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        assert isinstance(extractor, OMETiffImagingExtractor)


class TestBrukerTiffImagingExtractorVolumetric:
    """Test BrukerTiffImagingExtractor with volumetric, single-channel stub data."""

    folder_path = BRUKER_STUB_PATH / "volumetric_single_channel"

    def test_is_volumetric(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        assert extractor.is_volumetric is True

    def test_num_samples(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        assert extractor.get_num_samples() == 3

    def test_num_planes(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        assert extractor.get_num_planes() == 2

    def test_volume_shape(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        assert extractor.get_volume_shape() == (6, 8, 2)

    def test_sampling_frequency_is_volume_rate(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        # Frame period is 0.033s, 2 planes per volume, so volume rate = 1/(0.033*2)
        expected_volume_rate = 1.0 / (0.033 * 2)
        assert abs(extractor.get_sampling_frequency() - expected_volume_rate) < 0.1

    def test_get_series(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        data = extractor.get_series(0, 3)
        assert data.shape == (3, 6, 8, 2)
        for t in range(3):
            for z in range(2):
                assert_array_equal(data[t, :, :, z], t * 100 + z)

    def test_get_series_subset(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path)
        data = extractor.get_series(1, 2)
        assert data.shape == (1, 6, 8, 2)
        assert_array_equal(data[0, :, :, 0], 100)
        assert_array_equal(data[0, :, :, 1], 101)


class TestBrukerTiffImagingExtractorDualChannel:
    """Test BrukerTiffImagingExtractor with single-plane, dual-channel stub data."""

    folder_path = BRUKER_STUB_PATH / "single_plane_dual_channel"

    def test_channel_name_required(self):
        with pytest.raises(ValueError, match="channel_name must be specified"):
            BrukerTiffImagingExtractor(folder_path=self.folder_path)

    def test_channel_0(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path, channel_name="0")
        assert extractor.get_num_samples() == 3
        assert extractor.is_volumetric is False
        data = extractor.get_series(0, 3)
        assert data.shape == (3, 6, 8)
        for t in range(3):
            assert_array_equal(data[t, :, :], t * 100)

    def test_channel_1(self):
        extractor = BrukerTiffImagingExtractor(folder_path=self.folder_path, channel_name="1")
        data = extractor.get_series(0, 3)
        assert data.shape == (3, 6, 8)
        for t in range(3):
            assert_array_equal(data[t, :, :], t * 100 + 10)

    def test_invalid_channel_raises(self):
        with pytest.raises(ValueError, match="channel_index .* is out of range"):
            BrukerTiffImagingExtractor(folder_path=self.folder_path, channel_name="5")
