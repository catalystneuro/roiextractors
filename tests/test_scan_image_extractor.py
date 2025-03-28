import pytest
import numpy as np
from pathlib import Path
from numpy.testing import assert_array_equal

from roiextractors import (
    ScanImageTiffSinglePlaneImagingExtractor,
    ScanImageTiffMultiPlaneImagingExtractor,
    ScanImageTiffSinglePlaneMultiFileImagingExtractor,
    ScanImageImagingExtractor,
)
from .setup_paths import OPHYS_DATA_PATH

# Define the path to the ScanImage test files
SCANIMAGE_PATH = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage"


class TestScanImageExtractors:
    """Test the ScanImage extractor classes with various ScanImage files."""

    def test_scanimage_noroi(self):
        """Test with a ScanImage file without ROIs."""
        file_path = SCANIMAGE_PATH / "scanimage_20220923_noroi.tif"

        # Get available channels and planes
        channel_names = ScanImageTiffSinglePlaneImagingExtractor.get_available_channels(file_path)
        plane_names = ScanImageTiffSinglePlaneImagingExtractor.get_available_planes(file_path)

        # Create extractor for the first channel and plane
        extractor = ScanImageTiffSinglePlaneImagingExtractor(
            file_path=file_path, channel_name=channel_names[0], plane_name=plane_names[0]
        )

        # Basic properties
        assert extractor.get_num_frames() > 0
        assert len(extractor.get_image_shape()) == 2
        assert extractor.get_sampling_frequency() > 0

        # Test frame retrieval
        frames = extractor.get_frames([0, 1])
        assert frames.shape[0] == 2
        assert frames.shape[1:] == extractor.get_image_shape()

        # Test video retrieval
        video = extractor.get_video(start_frame=0, end_frame=2)
        assert video.shape[0] == 2
        assert video.shape[1:] == extractor.get_image_shape()

        # Compare frames and video
        assert_array_equal(frames, video)

    def test_scanimage_roi(self):
        """Test with a ScanImage file with ROIs."""
        # This is frames per slice 2, it should fail.

        file_path = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"

        # Get available channels and planes
        channel_names = ScanImageTiffSinglePlaneImagingExtractor.get_available_channels(file_path)
        plane_names = ScanImageTiffSinglePlaneImagingExtractor.get_available_planes(file_path)

        # Create extractor for the first channel and plane
        extractor = ScanImageTiffSinglePlaneImagingExtractor(
            file_path=file_path, channel_name=channel_names[0], plane_name=plane_names[0]
        )

        # Basic properties
        assert extractor.get_num_frames() > 0
        assert len(extractor.get_image_shape()) == 2
        assert extractor.get_sampling_frequency() > 0

        # Test frame retrieval
        frames = extractor.get_frames([0, 1])
        assert frames.shape[0] == 2
        assert frames.shape[1:] == extractor.get_image_shape()

        # Test video retrieval
        video = extractor.get_video(start_frame=0, end_frame=2)
        assert video.shape[0] == 2
        assert video.shape[1:] == extractor.get_image_shape()

        # Compare frames and video
        assert_array_equal(frames, video)

    def test_scanimage_version_3_8(self):
        """Test with a ScanImage version 3.8 file."""
        file_path = SCANIMAGE_PATH / "sample_scanimage_version_3_8.tiff"

        # Get available channels and planes
        channel_names = ScanImageTiffSinglePlaneImagingExtractor.get_available_channels(file_path)
        plane_names = ScanImageTiffSinglePlaneImagingExtractor.get_available_planes(file_path)

        # Create extractor for the first channel and plane
        extractor = ScanImageTiffSinglePlaneImagingExtractor(
            file_path=file_path, channel_name=channel_names[0], plane_name=plane_names[0]
        )

        # Basic properties
        assert extractor.get_num_frames() > 0
        assert len(extractor.get_image_shape()) == 2
        assert extractor.get_sampling_frequency() > 0

        # Test frame retrieval
        frames = extractor.get_frames([0, 1])
        assert frames.shape[0] == 2
        assert frames.shape[1:] == extractor.get_image_shape()

        # Test video retrieval
        video = extractor.get_video(start_frame=0, end_frame=2)
        assert video.shape[0] == 2
        assert video.shape[1:] == extractor.get_image_shape()

        # Compare frames and video
        assert_array_equal(frames, video)

    def test_scanimage_multivolume(self):
        """Test with a ScanImage multivolume file."""
        file_path = SCANIMAGE_PATH / "scanimage_20220801_multivolume.tif"

        # Create multi-plane extractor
        extractor = ScanImageTiffMultiPlaneImagingExtractor(file_path=file_path)

        # Basic properties
        assert extractor.get_num_frames() > 0
        assert extractor.get_sampling_frequency() > 0
        assert extractor.get_num_planes() > 1  # Should be multiplane

        # Test accessing individual planes
        for plane_idx in range(extractor.get_num_planes()):
            plane_extractor = extractor.imaging_extractors[plane_idx]
            assert plane_extractor.get_num_frames() > 0
            assert len(plane_extractor.get_image_shape()) == 2

            # Test frame retrieval for this plane
            frames = plane_extractor.get_frames([0, 1])
            assert frames.shape[0] == 2
            assert frames.shape[1:] == plane_extractor.get_image_shape()

    def test_scanimage_single(self):
        """Test with a ScanImage single plane file."""
        # This is frame per slice 24 and should fail
        file_path = SCANIMAGE_PATH / "scanimage_20220801_single.tif"

        # Get available channels and planes
        channel_names = ScanImageTiffSinglePlaneImagingExtractor.get_available_channels(file_path)
        plane_names = ScanImageTiffSinglePlaneImagingExtractor.get_available_planes(file_path)

        # Create extractor for the first channel and plane
        extractor = ScanImageTiffSinglePlaneImagingExtractor(
            file_path=file_path, channel_name=channel_names[0], plane_name=plane_names[0]
        )

        # Basic properties
        assert extractor.get_num_frames() > 0
        assert len(extractor.get_image_shape()) == 2
        assert extractor.get_sampling_frequency() > 0

        # Test frame retrieval
        frames = extractor.get_frames([0, 1])
        assert frames.shape[0] == 2
        assert frames.shape[1:] == extractor.get_image_shape()

        # Test video retrieval
        video = extractor.get_video(start_frame=0, end_frame=2)
        assert video.shape[0] == 2
        assert video.shape[1:] == extractor.get_image_shape()

        # Compare frames and video
        assert_array_equal(frames, video)

    def test_scanimage_volume(self):
        """Test with a ScanImage volume file."""
        # This is frames per slice 8 and should fail
        file_path = SCANIMAGE_PATH / "scanimage_20220801_volume.tif"

        # Create multi-plane extractor
        extractor = ScanImageTiffMultiPlaneImagingExtractor(file_path=file_path)

        # Basic properties
        assert extractor.get_num_frames() > 0
        assert extractor.get_sampling_frequency() > 0
        assert extractor.get_num_planes() > 1  # Should be multiplane

        # Test accessing individual planes
        for plane_idx in range(extractor.get_num_planes()):
            plane_extractor = extractor.imaging_extractors[plane_idx]
            assert plane_extractor.get_num_frames() > 0
            assert len(plane_extractor.get_image_shape()) == 2

            # Test frame retrieval for this plane
            frames = plane_extractor.get_frames([0, 1])
            assert frames.shape[0] == 2
            assert frames.shape[1:] == plane_extractor.get_image_shape()

    def test_scanimage_multifile(self):
        """Test with a ScanImage multifile series."""
        # This is non-volumetric data
        folder_path = SCANIMAGE_PATH
        file_pattern = "scanimage_20240320_multifile_*.tif"

        # Get available channels and planes from the first file
        first_file = SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif"
        channel_names = ScanImageTiffSinglePlaneImagingExtractor.get_available_channels(first_file)
        plane_names = ScanImageTiffSinglePlaneImagingExtractor.get_available_planes(first_file)

        # Create multi-file extractor
        extractor = ScanImageTiffSinglePlaneMultiFileImagingExtractor(
            folder_path=folder_path, file_pattern=file_pattern, channel_name=channel_names[0], plane_name=plane_names[0]
        )

        # Basic properties
        assert extractor.get_num_frames() > 0
        assert len(extractor.get_image_shape()) == 2
        assert extractor.get_sampling_frequency() > 0

        # Test frame retrieval
        frames = extractor.get_frames([0, 1])
        assert frames.shape[0] == 2
        assert frames.shape[1:] == extractor.get_image_shape()

        # Test video retrieval
        video = extractor.get_video(start_frame=0, end_frame=2)
        assert video.shape[0] == 2
        assert video.shape[1:] == extractor.get_image_shape()

        # Compare frames and video
        assert_array_equal(frames, video)

        # Test retrieving frames across file boundaries
        # Assuming each file has at least 5 frames
        if extractor.get_num_frames() > 10:
            # Get frames from different files
            cross_file_frames = extractor.get_frames([4, 8])
            assert cross_file_frames.shape[0] == 2
            assert cross_file_frames.shape[1:] == extractor.get_image_shape()

    def test_channel_selection(self):
        """Test channel selection in ScanImageTiffSinglePlaneImagingExtractor."""
        # Use a file that has multiple channels
        file_path = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"

        # Get available channels and planes
        channel_names = ScanImageTiffSinglePlaneImagingExtractor.get_available_channels(file_path)
        plane_names = ScanImageTiffSinglePlaneImagingExtractor.get_available_planes(file_path)

        # If there are multiple channels, test selecting different ones
        if len(channel_names) > 1:
            # Create extractors for different channels
            extractor1 = ScanImageTiffSinglePlaneImagingExtractor(
                file_path=file_path, channel_name=channel_names[0], plane_name=plane_names[0]
            )

            extractor2 = ScanImageTiffSinglePlaneImagingExtractor(
                file_path=file_path, channel_name=channel_names[1], plane_name=plane_names[0]
            )

            # Get frames from both extractors and verify they're different
            frames1 = extractor1.get_frames([0])
            frames2 = extractor2.get_frames([0])

            # The frames should be different if they're from different channels
            assert not np.array_equal(frames1, frames2)

    def test_invalid_channel(self):
        """Test that an invalid channel name raises a ValueError."""
        file_path = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"

        with pytest.raises(ValueError, match="Channel name .* not found"):
            ScanImageTiffSinglePlaneImagingExtractor(file_path=file_path, channel_name="InvalidChannel", plane_name="0")

    def test_scanimageimagingextractor_noroi(self):
        """Test the ScanImageImagingExtractor with a ScanImage file without ROIs."""
        file_path = SCANIMAGE_PATH / "scanimage_20220923_noroi.tif"
        extractor = ScanImageImagingExtractor(file_path=file_path)

        # Basic properties
        assert extractor.get_num_frames() > 0
        assert len(extractor.get_image_shape()) == 2
        assert extractor.get_sampling_frequency() > 0
        assert len(extractor.get_channel_names()) > 0

        # Test frame retrieval
        frames = extractor.get_frames([0, 1])
        assert frames.shape[0] == 2
        assert frames.shape[1:] == extractor.get_image_shape()

        # Test video retrieval
        video = extractor.get_video(start_frame=0, end_frame=2)
        assert video.shape[0] == 2
        assert video.shape[1:] == extractor.get_image_shape()

        # Compare frames and video
        assert_array_equal(frames, video)

    def test_scanimageimagingextractor_roi(self):
        """Test the ScanImageImagingExtractor with a ScanImage file with ROIs."""
        file_path = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"
        extractor = ScanImageImagingExtractor(file_path=file_path)

        # Basic properties
        assert extractor.get_num_frames() > 0
        assert len(extractor.get_image_shape()) == 2
        assert extractor.get_sampling_frequency() > 0
        assert len(extractor.get_channel_names()) > 0

        # Test frame retrieval
        frames = extractor.get_frames([0, 1])
        assert frames.shape[0] == 2
        assert frames.shape[1:] == extractor.get_image_shape()

        # Test video retrieval
        video = extractor.get_video(start_frame=0, end_frame=2)
        assert video.shape[0] == 2
        assert video.shape[1:] == extractor.get_image_shape()

        # Compare frames and video
        assert_array_equal(frames, video)

    def test_scanimageimagingextractor_multivolume(self):
        """Test the ScanImageImagingExtractor with a ScanImage multivolume file."""
        file_path = SCANIMAGE_PATH / "scanimage_20220801_multivolume.tif"
        extractor = ScanImageImagingExtractor(file_path=file_path)

        # Basic properties
        assert extractor.get_num_frames() > 0
        assert len(extractor.get_image_shape()) == 2
        assert extractor.get_sampling_frequency() > 0
        assert len(extractor.get_channel_names()) > 0
        assert extractor.get_num_planes() > 1  # Should be multiplane
        assert extractor.is_volumetric  # Should be volumetric

        # Test frame retrieval for volumetric data
        frames = extractor.get_frames([0, 1])
        assert frames.shape[0] == 2
        assert frames.shape[1:3] == extractor.get_image_shape()
        assert frames.shape[3] == extractor.get_num_planes()

        # Test video retrieval
        video = extractor.get_video(start_frame=0, end_frame=2)
        assert video.shape[0] == 2

        # Compare frames and video
        assert_array_equal(frames, video)

    def test_scanimageimagingextractor_single(self):
        """Test the ScanImageImagingExtractor with a ScanImage single plane file."""
        file_path = SCANIMAGE_PATH / "scanimage_20220801_single.tif"
        extractor = ScanImageImagingExtractor(file_path=file_path)

        # Basic properties
        assert extractor.get_num_frames() > 0
        assert len(extractor.get_image_shape()) == 2
        assert extractor.get_sampling_frequency() > 0
        assert len(extractor.get_channel_names()) > 0
        assert extractor.get_num_planes() == 1  # Should be single plane
        assert not extractor.is_volumetric  # Should not be volumetric

        # Test frame retrieval
        frames = extractor.get_frames([0, 1])
        assert frames.shape[0] == 2
        assert frames.shape[1:] == extractor.get_image_shape()

        # Test video retrieval
        video = extractor.get_video(start_frame=0, end_frame=2)
        assert video.shape[0] == 2
        assert video.shape[1:] == extractor.get_image_shape()

        # Compare frames and video
        assert_array_equal(frames, video)

    def test_scanimageimagingextractor_volume(self):
        """Test the ScanImageImagingExtractor with a ScanImage volume file."""
        file_path = SCANIMAGE_PATH / "scanimage_20220801_volume.tif"
        extractor = ScanImageImagingExtractor(file_path=file_path)

        # Basic properties
        assert extractor.get_num_frames() > 0
        assert len(extractor.get_image_shape()) == 2
        assert extractor.get_sampling_frequency() > 0
        assert len(extractor.get_channel_names()) > 0
        assert extractor.get_num_planes() > 1  # Should be multiplane
        assert extractor.is_volumetric  # Should be volumetric

        # Test frame retrieval for volumetric data
        frames = extractor.get_frames([0, 1])
        assert frames.shape[0] == 2
        assert frames.shape[1:3] == extractor.get_image_shape()
        assert frames.shape[3] == extractor.get_num_planes()

        # Test video retrieval
        video = extractor.get_video(start_frame=0, end_frame=2)
        assert video.shape[0] == 2

        # Compare frames and video
        assert_array_equal(frames, video)

    def test_scanimageimagingextractor_multifile(self):
        """Test the ScanImageImagingExtractor with a ScanImage multifile series."""
        # For multifile, we only need to provide the first file
        file_path = SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif"
        extractor = ScanImageImagingExtractor(file_path=file_path)

        # Basic properties
        assert extractor.get_num_frames() > 0
        assert len(extractor.get_image_shape()) == 2
        assert extractor.get_sampling_frequency() > 0
        assert len(extractor.get_channel_names()) > 0

        # Check if multiple files were detected
        assert len(extractor.file_paths) > 1

        # Test frame retrieval
        frames = extractor.get_frames([0, 1])
        assert frames.shape[0] == 2
        assert frames.shape[1:] == extractor.get_image_shape()

        # Test video retrieval
        video = extractor.get_video(start_frame=0, end_frame=2)
        assert video.shape[0] == 2
        assert video.shape[1:] == extractor.get_image_shape()

        # Compare frames and video
        assert_array_equal(frames, video)

        # Test retrieving frames across file boundaries
        # Assuming each file has at least 5 frames
        if extractor.get_num_frames() > 10:
            # Get frames from different files
            cross_file_frames = extractor.get_frames([4, 8])
            assert cross_file_frames.shape[0] == 2
            assert cross_file_frames.shape[1:] == extractor.get_image_shape()

    def test_scanimageimagingextractor_channel_selection(self):
        """Test channel selection in ScanImageImagingExtractor."""
        # Use a file that has multiple channels
        file_path = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"

        # First, check available channels without specifying a channel
        extractor = ScanImageImagingExtractor(file_path=file_path)
        channel_names = extractor._channel_names

        # If there are multiple channels, test selecting a specific one
        if len(channel_names) > 1:
            # Select the second channel
            second_channel = channel_names[1]
            extractor_ch2 = ScanImageImagingExtractor(file_path=file_path, channel_name=second_channel)

            # Verify the selected channel
            assert extractor_ch2.channel_name == second_channel
            assert extractor_ch2.channel_index == 1

            # Get frames from both extractors and verify they're different
            frames_ch1 = extractor.get_frames([0])
            frames_ch2 = extractor_ch2.get_frames([0])

            # The frames should be different if they're from different channels
            assert not np.array_equal(frames_ch1, frames_ch2)

    def test_scanimageimagingextractor_invalid_channel(self):
        """Test that an invalid channel name raises a ValueError in ScanImageImagingExtractor."""
        file_path = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"

        with pytest.raises(ValueError, match="Channel name .* not found"):
            ScanImageImagingExtractor(file_path=file_path, channel_name="InvalidChannel")
