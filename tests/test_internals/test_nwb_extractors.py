import unittest
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ophys import OpticalChannel, TwoPhotonSeries
from pynwb.testing.mock.file import mock_NWBFile
from pynwb.testing.mock.ophys import mock_ImagingPlane

from roiextractors import NwbImagingExtractor
from roiextractors.testing import generate_dummy_video


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

        image_size = nwb_imaging_extractor.get_image_shape()
        num_frames = nwb_imaging_extractor.get_num_samples()

        expected_image_size = self.image_size
        expected_num_frames = self.num_frames
        expected_num_channels = self.num_channels

        assert image_size == expected_image_size
        assert num_frames == expected_num_frames

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

        video = nwb_imaging_extractor.get_series()
        expected_video = self.video

        np.testing.assert_array_almost_equal(video, expected_video)


@pytest.fixture(scope="module")
def nwb_volumetric_file(tmp_path_factory):
    """Create a volumetric (3D) NWB file for testing."""
    tmp_path = tmp_path_factory.mktemp("nwb_volumetric")
    file_path = tmp_path / "test_nwb_volumetric_imaging_extractor.nwb"

    sampling_frequency = 30.0
    num_samples = 30
    rows = 50
    columns = 25
    num_planes = 10

    nwbfile = mock_NWBFile()

    # Generate volumetric data: (time, rows, cols, planes)
    video_shape = (num_samples, rows, columns, num_planes)

    dtype = "uint16"
    video = generate_dummy_video(size=video_shape, dtype=dtype)

    imaging_plane = mock_ImagingPlane(nwbfile=nwbfile)

    # NWB format: (time, width, height, depth)
    # So transpose from (time, rows, cols, planes) to (time, cols, rows, planes)
    image_series = TwoPhotonSeries(
        name="TwoPhotonSeries",
        data=video.transpose([0, 2, 1, 3]),  # roiextractors -> NWB transpose
        imaging_plane=imaging_plane,
        rate=sampling_frequency,
        unit="normalized amplitude",
    )

    nwbfile.add_acquisition(image_series)

    with NWBHDF5IO(file_path, "w") as io:
        io.write(nwbfile)

    return {
        "file_path": file_path,
        "video": video,
        "frame_shape": (rows, columns),
        "num_samples": num_samples,
        "num_planes": num_planes,
    }


class TestNwbVolumetricImagingExtractor:
    """Tests for volumetric (3D) NWB imaging data."""

    def test_get_series_volumetric(self, nwb_volumetric_file):
        nwb_imaging_extractor = NwbImagingExtractor(file_path=nwb_volumetric_file["file_path"])
        video = nwb_volumetric_file["video"]

        # Test full series
        series = nwb_imaging_extractor.get_series()
        expected_series = video
        assert series.shape == expected_series.shape
        np.testing.assert_array_almost_equal(series, expected_series)


if __name__ == "__main__":
    unittest.main()
