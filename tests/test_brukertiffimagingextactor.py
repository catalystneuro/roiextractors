import shutil
import tempfile
from pathlib import Path

import numpy as np
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal
from tifffile import tifffile

from roiextractors import BrukerTiffImagingExtractor
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

        cls.folder_path = folder_path
        extractor = BrukerTiffImagingExtractor(folder_path=folder_path)
        cls.extractor = extractor

        file_paths = [Path(folder_path) / file for file in extractor._file_paths]
        cls.video = _get_test_video(file_paths=file_paths)

        # temporary directory for testing assertion when xml file is missing
        test_dir = tempfile.mkdtemp()
        cls.test_dir = test_dir
        shutil.copy(file_paths[0], Path(test_dir) / file_paths[0].name)

    @classmethod
    def tearDownClass(cls):
        # remove the temporary directory and its contents
        shutil.rmtree(cls.test_dir)

    def test_tif_files_are_missing_assertion(self):
        folder_path = "not a tiff path"
        exc_msg = f"The TIF image files are missing from '{folder_path}'."
        with self.assertRaisesWith(AssertionError, exc_msg=exc_msg):
            BrukerTiffImagingExtractor(folder_path=folder_path)

    def test_xml_configuration_file_is_missing_assertion(self):
        folder_path = self.test_dir
        exc_msg = f"The XML configuration file is not found at '{folder_path}'."
        with self.assertRaisesWith(AssertionError, exc_msg=exc_msg):
            BrukerTiffImagingExtractor(folder_path=folder_path)

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
        assert_array_equal(self.extractor.get_video(), self.video)

    def test_brukertiffextractor_get_single_frame(self):
        assert_array_equal(self.extractor.get_frames(frame_idxs=[0]), self.video[0])

    def test_brukertiffextractor_get_video_multi_channel_assertion(self):
        exc_msg = "The BrukerTiffImagingExtractor does not currently support multiple color channels."
        with self.assertRaisesWith(NotImplementedError, exc_msg=exc_msg):
            self.extractor.get_video(channel=1)


class TestBrukerTiffExtractorDualPlaneCase(TestCase):
    @classmethod
    def setUpClass(cls):
        folder_path = str(
            OPHYS_DATA_PATH / "imaging_datasets" / "BrukerTif" / "NCCR32_2022_11_03_IntoTheVoid_t_series-005"
        )
        cls.folder_path = folder_path
        cls.extractor = BrukerTiffImagingExtractor(folder_path=folder_path)

        first_plane_file_paths = [
            f"{cls.folder_path}/NCCR32_2022_11_03_IntoTheVoid_t_series-005_Cycle0000{num + 1}_Ch2_000001.ome.tif"
            for num in range(5)
        ]
        second_plane_file_paths = [
            f"{cls.folder_path}/NCCR32_2022_11_03_IntoTheVoid_t_series-005_Cycle0000{num + 1}_Ch2_000002.ome.tif"
            for num in range(5)
        ]

        cls.test_video = np.zeros((5, 512, 512, 2), dtype=np.uint16)
        first_plane_video = _get_test_video(file_paths=first_plane_file_paths)
        cls.test_video[..., 0] = first_plane_video
        second_plane_video = _get_test_video(file_paths=second_plane_file_paths)
        cls.test_video[..., 1] = second_plane_video

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

    def test_brukertiffextractor_get_video(self):
        assert_array_equal(self.extractor.get_video(), self.test_video)
        assert_array_equal(self.extractor.get_video(start_frame=2, end_frame=4), self.test_video[2:4])
