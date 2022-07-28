import unittest
from tempfile import mkdtemp
from datetime import datetime
from pathlib import Path

import numpy as np

from pynwb import NWBFile, NWBHDF5IO
from pynwb.ophys import TwoPhotonSeries, OpticalChannel

from roiextractors import NwbImagingExtractor
from roiextractors.testing import generate_dummy_video, assert_get_frames_return_shape


class TestNwbImagingExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.session_start_time = datetime.now().astimezone()
        self.file_path = Path(mkdtemp()) / "test_nwb_imaging_extractor.nwb"

        self.sampling_frequency = 30.0
        self.num_frames = 30
        self.rows = 50
        self.columns = 25
        self.num_channels = 1

        self.nwbfile = NWBFile(
            session_description="session_description",
            identifier="file_id",
            session_start_time=self.session_start_time,
        )
        self.device = self.nwbfile.create_device(name="Microscope")

        channel_names = [f"channel_num_{num}" for num in range(self.num_channels)]
        self.optical_channel_list = [
            OpticalChannel(name=channel_name, description="description", emission_lambda=500.0)
            for channel_name in channel_names
        ]
        self.video_shape = (self.num_frames, self.rows, self.columns)
        self.image_size = (self.rows, self.columns)

        self.dtype = "uint"
        self.video = generate_dummy_video(size=self.video_shape, dtype=self.dtype)

        self.imaging_plane = self.nwbfile.create_imaging_plane(
            name="ImagingPlane",
            optical_channel=self.optical_channel_list,
            imaging_rate=self.sampling_frequency,
            description="a very interesting part of the brain",
            device=self.device,
            excitation_lambda=600.0,
            indicator="GFP",
            location="the location in the brain",
        )

        # using internal data. this data will be stored inside the NWB file
        self.image_series = TwoPhotonSeries(
            name="TwoPhotonSeries",
            data=self.video.transpose([0, 2, 1]),
            imaging_plane=self.imaging_plane,
            rate=self.sampling_frequency,
            unit="normalized amplitude",
        )

        self.nwbfile.add_acquisition(self.image_series)

        with NWBHDF5IO(self.file_path, "w") as io:
            io.write(self.nwbfile)

    def test_basic_setup(self):

        nwb_imaging_extractor = NwbImagingExtractor(file_path=self.file_path)

        image_size = nwb_imaging_extractor.get_image_size()
        num_frames = nwb_imaging_extractor.get_num_frames()
        num_channels = nwb_imaging_extractor.get_num_channels()

        expected_image_size = self.image_size
        expected_num_frames = self.num_frames
        expected_num_channels = self.num_channels

        assert image_size == expected_image_size
        assert num_frames == expected_num_frames
        assert num_channels == expected_num_channels

        # Test numpy like behavior for frame_idxs
        frame_idxs = 0
        frames_with_scalar = nwb_imaging_extractor.get_frames(frame_idxs)
        expected_frames = self.video[frame_idxs, ...]
        np.testing.assert_array_almost_equal(frames_with_scalar, expected_frames)

        frame_idxs = [0]
        frames_with_singleton = nwb_imaging_extractor.get_frames(frame_idxs)
        expected_frames = self.video[frame_idxs, ...]
        np.testing.assert_array_almost_equal(frames_with_singleton, expected_frames)

        frame_idxs = [0, 1]
        frames_with_list = nwb_imaging_extractor.get_frames(frame_idxs)
        expected_frames = self.video[frame_idxs, ...]
        np.testing.assert_array_almost_equal(frames_with_list, expected_frames)

        frame_idxs = np.array([0, 1])
        frames_with_array = nwb_imaging_extractor.get_frames(frame_idxs)
        expected_frames = self.video[frame_idxs, ...]
        np.testing.assert_array_almost_equal(frames_with_array, expected_frames)

        # Test spikeinterface-like behavior for get_video
        one_element_video_shape = nwb_imaging_extractor.get_video(start_frame=0, end_frame=1, channel=0).shape
        expected_shape = (1, image_size[0], image_size[1])
        assert one_element_video_shape == expected_shape

        video = nwb_imaging_extractor.get_video()
        expected_video = self.video

        np.testing.assert_array_almost_equal(video, expected_video)

    def test_get_frames_indexing_with_single_channel(self):

        nwb_imaging_extractor = NwbImagingExtractor(file_path=self.file_path)
        assert_get_frames_return_shape(imaging_extractor=nwb_imaging_extractor)


if __name__ == "__main__":
    unittest.main()
