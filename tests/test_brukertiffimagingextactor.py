import shutil
import tempfile
from pathlib import Path

import numpy as np
from hdmf.testing import TestCase
from natsort import natsorted
from numpy.testing import assert_array_equal
from tifffile import tifffile

from roiextractors import BrukerTiffMultiPlaneImagingExtractor, BrukerTiffSinglePlaneImagingExtractor
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
        self.assertEqual(self.extractor.get_image_size(), (512, 512))

    def test_brukertiffextractor_num_frames(self):
        self.assertEqual(self.extractor.get_num_frames(), 10)

    def test_brukertiffextractor_sampling_frequency(self):
        self.assertEqual(self.extractor.get_sampling_frequency(), 29.873732099062256)

    def test_brukertiffextractor_channel_names(self):
        self.assertEqual(self.extractor.get_channel_names(), ["Ch2"])

    def test_brukertiffextractor_num_channels(self):
        self.assertEqual(self.extractor.get_num_channels(), 1)

    def test_brukertiffextractor_dtype(self):
        self.assertEqual(self.extractor.get_dtype(), np.uint16)

    def test_brukertiffextractor_get_video(self):
        video = self.extractor.get_video()
        assert_array_equal(video, self.video)
        self.assertEqual(video.dtype, np.uint16)
        assert_array_equal(self.extractor.get_video(start_frame=0, end_frame=1), self.video[:1])

    def test_brukertiffextractor_get_single_frame(self):
        assert_array_equal(self.extractor.get_frames(frames=[0]), self.video[0][np.newaxis, ...])


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
        self.assertEqual(self.extractor.get_image_size(), (512, 512, 2))

    def test_brukertiffextractor_num_frames(self):
        self.assertEqual(self.extractor.get_num_frames(), 5)

    def test_brukertiffextractor_sampling_frequency(self):
        self.assertEqual(self.extractor.get_sampling_frequency(), 20.629515014336377)

    def test_brukertiffextractor_channel_names(self):
        self.assertEqual(self.extractor.get_channel_names(), ["Ch2"])

    def test_brukertiffextractor_num_channels(self):
        self.assertEqual(self.extractor.get_num_channels(), 1)

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

    def test_brukertiffextractor_get_video(self):
        video = self.extractor.get_video()
        assert_array_equal(video, self.test_video)
        self.assertEqual(video.dtype, np.uint16)
        assert_array_equal(self.extractor.get_video(start_frame=2, end_frame=4), self.test_video[2:4])

    def test_brukertiffextractor_get_single_frame(self):
        assert_array_equal(self.extractor.get_frames(frames=[0]), self.test_video[0][np.newaxis, ...])


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
        self.assertEqual(self.extractor.get_image_size(), (512, 512))

    def test_brukertiffextractor_channel_names(self):
        self.assertEqual(self.extractor.get_channel_names(), ["Ch1"])

    def test_brukertiffextractor_dtype(self):
        self.assertEqual(self.extractor.get_dtype(), np.uint16)

    def test_brukertiffextractor_sampling_frequency(self):
        self.assertEqual(self.extractor.get_sampling_frequency(), 29.873615189896864)

    def test_brukertiffextractor_get_video(self):
        assert_array_equal(self.extractor.get_video(start_frame=0, end_frame=1), self.test_video_ch1[:1])
        video = self.extractor.get_video()
        assert_array_equal(video, self.test_video_ch1)
        self.assertEqual(video.dtype, np.uint16)

    def test_brukertiffextractor_second_stream_get_video(self):
        extractor = BrukerTiffSinglePlaneImagingExtractor(folder_path=self.folder_path, stream_name="Ch2")
        video = extractor.get_video()
        assert_array_equal(extractor.get_video(), self.test_video_ch2)
        self.assertEqual(video.dtype, np.uint16)

    def test_brukertiffextractor_second_stream_sampling_frequency(self):
        extractor = BrukerTiffSinglePlaneImagingExtractor(folder_path=self.folder_path, stream_name="Ch2")
        self.assertEqual(
            self.extractor.get_sampling_frequency(),
            extractor.get_sampling_frequency(),
        )
