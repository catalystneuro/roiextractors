import unittest
from tempfile import mkdtemp
from datetime import datetime
from pathlib import Path

import numpy as np

from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from pynwb.image import ImageSeries
from pynwb.ophys import TwoPhotonSeries, OpticalChannel, ImageSegmentation

from roiextractors import NwbImagingExtractor


class TestNwbImagingExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.session_start_time = datetime.now()
        self.file_path = Path(mkdtemp()) / "test_nwb_imaging_extractor.nwb"

        self.nwbfile = NWBFile(
            session_description="session_description",
            identifier="file_id",
            session_start_time=self.session_start_time,
        )
        self.device = self.nwbfile.create_device(
            name="Microscope", description="My two-photon microscope", manufacturer="The best microscope manufacturer"
        )
        self.optical_channel = OpticalChannel(
            name="OpticalChannel", description="an optical channel", emission_lambda=500.0
        )

        self.sampling_frequency = 30.0
        self.num_frames = 30
        self.rows = 50
        self.columns = 25
        self.num_channels = 1

        self.video_shape = (self.num_frames, self.rows, self.columns, self.num_channels)
        self.image_size = (self.rows, self.columns)

        self.dtype = "uint"
        self.video = np.random.randint(low=0, high=256, size=self.video_shape).astype(self.dtype)

        self.imaging_plane = self.nwbfile.create_imaging_plane(
            name="ImagingPlane",
            optical_channel=self.optical_channel,
            imaging_rate=self.sampling_frequency,
            description="a very interesting part of the brain",
            device=self.device,
            excitation_lambda=600.0,
            indicator="GFP",
            location="V1",
            grid_spacing=[0.01, 0.01],
            grid_spacing_unit="meters",
            origin_coords=[1.0, 2.0, 3.0],
            origin_coords_unit="meters",
        )

        # using internal data. this data will be stored inside the NWB file
        self.image_series1 = TwoPhotonSeries(
            name="TwoPhotonSeries1",
            data=self.video,
            imaging_plane=self.imaging_plane,
            rate=self.sampling_frequency,
            unit="normalized amplitude",
        )

        self.nwbfile.add_acquisition(self.image_series1)

        with NWBHDF5IO(self.file_path, "w") as io:
            io.write(self.nwbfile)

    def test_basic_setup(self):

        nwb_imaging_extractor = NwbImagingExtractor(file_path=self.file_path, optical_series_name=None)

        image_size = nwb_imaging_extractor.get_image_size()
        num_frames = nwb_imaging_extractor.get_num_frames()
        num_channels = nwb_imaging_extractor.get_num_channels()

        expected_image_size = self.image_size
        expected_num_frames = self.num_frames
        expected_num_channels = self.num_channels

        assert image_size == expected_image_size
        assert num_frames == expected_num_frames
        assert num_channels == expected_num_channels

        video = nwb_imaging_extractor.get_video()
        expected_video = self.video

        np.testing.assert_array_almost_equal(video, expected_video)


if __name__ == "__main__":
    unittest.main()
