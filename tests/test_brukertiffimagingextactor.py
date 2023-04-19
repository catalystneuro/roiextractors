from pathlib import Path

import numpy as np
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal
from tifffile import tifffile

from roiextractors import BrukerTiffImagingExtractor
from tests.setup_paths import OPHYS_DATA_PATH


class TestBrukerTiffExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        folder_path = str(
            OPHYS_DATA_PATH / "imaging_datasets" / "Brukertif" / "NCCR32_2023_02_20_Into_the_void_t_series_baseline-000"
        )

        cls.folder_path = folder_path
        extractor = BrukerTiffImagingExtractor(folder_path=folder_path)
        cls.extractor = extractor

        frames = []
        for file in extractor._file_paths:
            with tifffile.TiffFile(Path(folder_path) / file, _multifile=False) as tif:
                frames.append(tif.asarray())
        cls.video = np.stack(frames, axis=0)

    def test_tif_files_are_missing_assertion(self):
        folder_path = "not a tiff path"
        exc_msg = f"The TIF image files are missing from '{folder_path}'."
        with self.assertRaisesWith(AssertionError, exc_msg=exc_msg):
            BrukerTiffImagingExtractor(folder_path=folder_path)

    # TODO: create temporary folder, without xml and test on that folder
    # def test_xml_configuration_file_is_missing_assertion(self):
    #     folder_path = "/Volumes/t7-ssd/Pinto/brukertiff/test_no_xml"
    #     exc_msg = f"The XML configuration file is not found at '{folder_path}'."
    #     with self.assertRaisesWith(AssertionError, exc_msg=exc_msg):
    #         BrukerTiffImagingExtractor(folder_path=folder_path)

    def test_brukertiffextractor_image_size(self):
        self.assertEqual(self.extractor.get_image_size(), (512, 512))

    def test_brukertiffextractor_num_frames(self):
        self.assertEqual(self.extractor.get_num_frames(), 10)

    def test_brukertiffextractor_sampling_frequency(self):
        self.assertEqual(self.extractor.get_sampling_frequency(), 30.345939461428763)

    def test_brukertiffextractor_channel_names(self):
        self.assertEqual(self.extractor.get_channel_names(), ["Ch2"])

    def test_brukertiffextractor_num_channels(self):
        self.assertEqual(self.extractor.get_num_channels(), 1)

    def test_brukertiffextractor_dtype(self):
        self.assertEqual(self.extractor.get_dtype(), np.uint16)

    def test_brukertiffextractor_get_video(self):
        assert_array_equal(self.extractor.get_video(), self.video)

    def test_brukertiffextractor_get_video_multi_channel_assertion(self):
        exc_msg = "The BrukerTiffImagingExtractor does not currently support multiple color channels."
        with self.assertRaisesWith(NotImplementedError, exc_msg=exc_msg):
            self.extractor.get_video(channel=1)

    def test_brukertiffextractor_xml_metadata(self):
        xml_metadata = self.extractor.xml_metadata

        self.assertEqual(xml_metadata["version"], "5.6.64.400")
        self.assertEqual(xml_metadata["date"], "2/20/2023 3:58:25 PM")
        self.assertEqual(xml_metadata["framePeriod"], "0.032953338")
        self.assertEqual(
            xml_metadata["micronsPerPixel"], [{"XAxis": "1.1078125"}, {"YAxis": "1.1078125"}, {"ZAxis": "5"}]
        )
