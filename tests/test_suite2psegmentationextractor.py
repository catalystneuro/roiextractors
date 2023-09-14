from hdmf.testing import TestCase

from roiextractors import Suite2pSegmentationExtractor
from tests.setup_paths import OPHYS_DATA_PATH


class TestSuite2pSegmentationExtractor(TestCase):
    @classmethod
    def setUpClass(cls):
        folder_path = str(OPHYS_DATA_PATH / "segmentation_datasets" / "suite2p")
        cls.available_streams = dict(
            channel_streams=["chan1", "chan2"],
            plane_streams=dict(chan1=["chan1_plane0", "chan1_plane1", "chan1_combined"], chan2=["chan2_plane0", "chan2_plane1"])
        )

        cls.folder_path = folder_path

        extractor = Suite2pSegmentationExtractor(folder_path=folder_path, stream_name="chan1_plane2")
        cls.extractor = extractor

    def test_stream_names(self):
        self.assertEqual(Suite2pSegmentationExtractor.get_streams(folder_path=self.folder_path), self.available_streams)

    def test_multi_stream_warns(self):
        exc_msg = (
            "More than one channel is detected! Please specify which stream you wish to load with the `stream_name` argument. "
            "To see what streams are available, call `Suite2pSegmentationExtractor.get_streams(folder_path=...)`."
            "This is going to raise ValueError in the future."
        )
        with self.assertRaisesWith(exc_type=ValueError, exc_msg=exc_msg):
            Suite2pSegmentationExtractor(folder_path=self.folder_path)

    def test_invalid_stream_raises(self):
        exc_msg = (
            "The selected stream 'plane0' is not a valid stream name. To see what streams are available, call `Suite2pSegmentationExtractor.get_streams(folder_path=...)`."
        )
        with self.assertRaisesWith(exc_type=ValueError, exc_msg=exc_msg):
            Suite2pSegmentationExtractor(folder_path=self.folder_path, stream_name="plane0")

    def test_incorrect_stream_raises(self):
        exc_msg = (
            "The selected stream 'chan1_plane2' is not in the available plane_streams '['chan1_plane0', 'chan1_plane1', 'chan1_combined']'!"
        )
        with self.assertRaisesWith(exc_type=ValueError, exc_msg=exc_msg):
            Suite2pSegmentationExtractor(folder_path=self.folder_path, stream_name="chan1_plane2")
