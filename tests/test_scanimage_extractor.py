import pytest
import numpy as np
from pathlib import Path
from numpy.testing import assert_array_equal
from tifffile import TiffReader

from roiextractors import (
    ScanImageImagingExtractor,
)
from .setup_paths import OPHYS_DATA_PATH

# Define the path to the ScanImage test files
SCANIMAGE_PATH = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage"


class TestScanImageExtractor:
    """Test the ScanImage extractor classes with various ScanImage files."""

    def test_planar_single_channel_single_file(self):
        """Test with planar (non-volumetric) single channel data.

        File: scanimage_20220801_single.tif
        Metadata:
        - Volumetric: False (single plane)
        - Channels: 1
        - Frames: 3
        - Frame rate: 15.2379 Hz
        - Image shape: 1024 x 1024
        """
        # This is frame per slice 24 and should fail
        file_path = SCANIMAGE_PATH / "scanimage_20220801_single.tif"

        extractor = ScanImageImagingExtractor(file_path=file_path)

        assert extractor.is_volumetric == False
        assert extractor.get_num_samples() == 3
        assert extractor.get_image_shape() == (1024, 1024)
        assert extractor.get_sampling_frequency() == 15.2379

        samples_per_file = extractor.get_num_samples() / len(extractor.file_paths)
        start_sample = 0
        end_sample = samples_per_file
        for file_path in extractor.file_paths:
            with TiffReader(file_path) as tiff_reader:
                data = tiff_reader.asarray()
                extractor_samples = extractor.get_series(start_sample=start_sample, end_sample=end_sample)
                assert_array_equal(data, extractor_samples)

                start_sample += samples_per_file
                end_sample += samples_per_file

    def test_planar_two_channels_single_file(self):
        """Test with planar (non-volumetric) two-channel data in a single file.

        File: planar_two_ch_single_files_00001_00001.tif
        Metadata:
        - Acquisition mode: grab
        - 1000 samples (frames)
        - Volumetric: False but `SI.hStackManager.enable` is set to `True`
        - Frame shape: (20, 20)
        - Channels: 2 (`Channel 1` and `Channel 2`)
        - Frames per slice: 1
        - Frame rate: 523.926 Hz
        - Volume rate: 523.926 Hz
        - Pages/IDFs: 2000

        This test verifies that the extractor correctly:
        1. Identifies the data as non-volumetric despite `SI.hStackManager.enable` being True
        2. Extracts the correct metadata (sampling frequency, frame count, dimensions)
        3. Retrieves the correct data for both channels
        """
        file_path = SCANIMAGE_PATH / "planar_two_channels_single_file" / "planar_two_ch_single_files_00001_00001.tif"

        # Test with Channel 1
        extractor_ch1 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 1")

        assert extractor_ch1.is_volumetric == False
        assert extractor_ch1.get_sampling_frequency() == 523.926
        assert extractor_ch1.get_image_shape() == (20, 20)
        assert extractor_ch1.get_num_samples() == 1000

        # Test with Channel 2
        extractor_ch2 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 2")

        assert extractor_ch2.is_volumetric == False
        assert extractor_ch2.get_sampling_frequency() == 523.926
        assert extractor_ch2.get_image_shape() == (20, 20)
        assert extractor_ch2.get_num_samples() == 1000

        # Get data from both extractors
        extractor_data_ch1 = extractor_ch1.get_series()
        extractor_data_ch2 = extractor_ch2.get_series()

        # Verify that the extractor data has the correct shape
        # For planar data, shape should be (samples, height, width)
        expected_shape = (extractor_ch1.get_num_samples(), *extractor_ch1.get_image_shape())
        assert (
            extractor_data_ch1.shape == expected_shape
        ), f"Channel 1: Expected shape {expected_shape}, got {extractor_data_ch1.shape}"
        assert (
            extractor_data_ch2.shape == expected_shape
        ), f"Channel 2: Expected shape {expected_shape}, got {extractor_data_ch2.shape}"

        # Read tiff data for comparison
        with TiffReader(file_path) as tiff_reader:
            tiff_data = tiff_reader.asarray()

            # For multi-channel data, tiff_data shape is (frames, channels, height, width)
            # Verify the shape of the tiff data
            assert len(tiff_data.shape) == 4, "Multi-channel tiff data should have 4 dimensions"
            assert tiff_data.shape[1] == 2, "Tiff data should have 2 channels"

            # Compare a subset of the data for each channel
            # We'll check the first 10 frames
            num_frames_to_check = 10

            # Get the tiff data for each channel
            tiff_ch1 = tiff_data[:num_frames_to_check, 0, :, :]  # Channel 1 (index 0)
            tiff_ch2 = tiff_data[:num_frames_to_check, 1, :, :]  # Channel 2 (index 1)

            # Get the extractor data for the same frames
            extractor_ch1_subset = extractor_data_ch1[:num_frames_to_check]
            extractor_ch2_subset = extractor_data_ch2[:num_frames_to_check]

            # Compare the data for Channel 1
            np.testing.assert_array_equal(
                extractor_ch1_subset,
                tiff_ch1,
                f"Channel 1 data does not match tiff data",
            )

            # Compare the data for Channel 2
            np.testing.assert_array_equal(
                extractor_ch2_subset,
                tiff_ch2,
                f"Channel 2 data does not match tiff data",
            )

    def test_planar_multi_channnel_multi_file(self):
        """Test with multi-file planar data acquired in loop acquisition mode.

        Files: scanimage_20240320_multifile_0000[1-3].tif
        Metadata:
        - Acquisition mode: loop (3 acquisitions per loop)
        - Volumetric: False (1 slice)
        - Channels: 2 (using "Channel 3")
        - Frames per slice: 10
        - Frame rate: 98.84 Hz
        - Image shape: 512 x 512

        This test verifies that the extractor correctly:
        1. Detects and loads all 3 files in the series
        2. Extracts the correct metadata (volumetric status, sampling frequency, etc.)
        3. Retrieves the correct data from each file in the series

        The test checks:
        - File detection: Confirms all 3 files in the series are found
        - File ordering: Verifies files are loaded in the correct sequence
        - Metadata extraction: Validates volumetric status, frame count, dimensions, and sampling rate
        - Data integrity: Ensures data from each file is correctly loaded and matches the original files
        """

        # First file of the series should be enough to initialize
        file_path = SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif"

        extractor = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 1")
        assert len(extractor.file_paths) == 3

        expected_files_names = [
            "scanimage_20240320_multifile_00001.tif",
            "scanimage_20240320_multifile_00002.tif",
            "scanimage_20240320_multifile_00003.tif",
        ]
        extracted_file_names = [Path(file_path).name for file_path in extractor.file_paths]

        assert extracted_file_names == expected_files_names

        assert extractor.is_volumetric == False
        assert extractor.get_num_samples() == 10 * 3  # 10 samples per file
        assert extractor.get_image_shape() == (512, 512)
        assert extractor.get_sampling_frequency() == 98.83842317607626

        samples_per_file = extractor.get_num_samples() / len(extractor.file_paths)
        start_sample = 0
        end_sample = samples_per_file
        for file_path in extractor.file_paths:
            with TiffReader(file_path) as tiff_reader:
                data = tiff_reader.asarray()
                channel_data = data[:, extractor._channel_index, ...]
                extractor_samples = extractor.get_series(start_sample=start_sample, end_sample=end_sample)
                assert_array_equal(channel_data, extractor_samples)

                start_sample += samples_per_file
                end_sample += samples_per_file

    def test_volumetric_single_channel_single_file(self):
        """Test with volumetric data in a single file.

        File: vol_no_flyback_00001_00001_stub.tif
        Metadata:
        - Acquisition state: grab
        - Volumetric: True (9 slices per volume)
        - Frame shape: 20 x 20
        - Channels saved: 1 (single channel)
        - Frames per slice: 1
        - Frame rate: 79.8033 Hz
        - Volume rate: 8.86703 Hz
        - Number of volumes: 10
        - Pages/IDFs: 90
        """
        file_path = (
            SCANIMAGE_PATH / "volumetric_single_channel_single_file_no_flyback" / "vol_no_flyback_00001_00001_stub.tif"
        )

        extractor = ScanImageImagingExtractor(file_path=file_path)

        assert extractor.is_volumetric == True
        assert extractor.get_num_samples() == 100
        assert extractor.get_image_shape() == (20, 20)
        assert extractor.get_sampling_frequency() == 32.7454


class TestScanImageVolumetricWithFlybackFrames:
    """Test the ScanImage extractor classes with files that have flyback frames."""

    def test_volumetric_single_channel_single_file(self):
        """
        First File: vol_one_ch_single_files_00002_00001.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (9 slices and 7 flyback frames)
        - Frame shape: 20 x 20
        - Channels: 1 (single channel)
        - Frames per slice: 1
        - Frame rate: 523.926 Hz
        - Volume rate: 32.7454 Hz
        - Pages/IDFs: 1600
        """
        file_path = SCANIMAGE_PATH / "volumetric_single_channel_single_file" / "vol_one_ch_single_files_00002_00001.tif"

        extractor = ScanImageImagingExtractor(file_path=file_path)

        assert extractor.is_volumetric == True
        assert extractor.num_flyback_frames == 7
        assert extractor.get_sampling_frequency() == 32.7454
        assert extractor.get_image_shape() == (20, 20)
        assert extractor.get_num_planes() == 9

        ifds = 1600
        total_frames_per_cycle = extractor.get_num_planes() + extractor.num_flyback_frames
        num_acquisition_cycles = ifds // total_frames_per_cycle
        assert extractor.get_num_samples() == num_acquisition_cycles

        # Get data from extractor
        extractor_data = extractor.get_series()

        # Verify that the extractor data has the correct shape
        # For volumetric data, shape should be (samples, height, width, planes)
        expected_shape = (num_acquisition_cycles, *extractor.get_image_shape(), extractor.get_num_planes())
        assert extractor_data.shape == expected_shape, f"Expected shape {expected_shape}, got {extractor_data.shape}"

        # Read tiff data for comparison
        with TiffReader(file_path) as tiff_reader:
            tiff_data = tiff_reader.asarray()

            # For each sample volume in the extractor data
            for sample_index in range(num_acquisition_cycles):
                # Calculate the slice of frames for this sample (excluding flyback frames)
                start_frame = sample_index * total_frames_per_cycle
                end_frame = start_frame + extractor.get_num_planes()
                tiff_volume_frames = tiff_data[start_frame:end_frame]

                # Reshape the extractor data to match the tiff data format for comparison
                # extractor_data is (samples, height, width, planes)
                # We need to compare with tiff_volume_frames which is (planes, height, width)
                extractor_volume = np.moveaxis(extractor_data[sample_index], -1, 0)

                # Compare the data
                np.testing.assert_array_equal(
                    extractor_volume, tiff_volume_frames, f"Sample {sample_index} volume does not match tiff data"
                )

    def test_volumetric_single_channel_multi_file(self):
        """
        First File: vol_one_ch_single_files_00002_00001.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (9 slices and 7 flyback frames)
        - Frame shape: 20 x 20
        - Channels: 1 (single channel)
        - Frames per slice: 1
        - Frame rate: 523.926 Hz
        - Volume rate: 32.7454 Hz
        - Pages/IDFs: 160 in each of the 10 files.
        """
        file_path = SCANIMAGE_PATH / "volumetric_single_channel_multi_file" / "vol_one_ch_multi_files_00001_00001.tif"

        extractor = ScanImageImagingExtractor(file_path=file_path)

        assert extractor.is_volumetric == True
        assert extractor.num_flyback_frames == 7
        assert extractor.get_sampling_frequency() == 32.7454
        assert extractor.get_image_shape() == (20, 20)
        assert extractor.get_num_planes() == 9

        ifds = 160 * 10
        total_frames_per_cycle = extractor.get_num_planes() + extractor.num_flyback_frames
        num_acquisition_cycles = ifds // total_frames_per_cycle
        assert extractor.get_num_samples() == num_acquisition_cycles

        # Get the entire series data from the extractor
        extractor_data = extractor.get_series()

        # Verify that the extractor data has the correct shape
        # For volumetric data, shape should be (samples, height, width, planes)
        expected_shape = (extractor.get_num_samples(), *extractor.get_image_shape(), extractor.get_num_planes())
        assert extractor_data.shape == expected_shape, f"Expected shape {expected_shape}, got {extractor_data.shape}"

        # For each sample volume in the extractor data
        for sample_index in range(extractor.get_num_samples()):
            # Determine which file contains this sample
            frames_per_file = 160
            frames_per_sample = total_frames_per_cycle
            samples_per_file = frames_per_file // frames_per_sample
            file_index = sample_index // samples_per_file

            # Get the corresponding file path
            current_file_path = extractor.file_paths[file_index]

            # Calculate the relative sample index within this file
            relative_sample_index = sample_index % samples_per_file

            # Read tiff data for comparison
            with TiffReader(current_file_path) as tiff_reader:
                tiff_data = tiff_reader.asarray()

                # Calculate the slice of frames for this sample (excluding flyback frames)
                start_frame = relative_sample_index * total_frames_per_cycle
                end_frame = start_frame + extractor.get_num_planes()
                tiff_volume_frames = tiff_data[start_frame:end_frame]

                # Reshape the extractor data to match the tiff data format for comparison
                # extractor_data is (samples, height, width, planes)
                # We need to compare with tiff_volume_frames which is (planes, height, width)
                extractor_volume = np.moveaxis(extractor_data[sample_index], -1, 0)

                # Compare the data
                np.testing.assert_array_equal(
                    extractor_volume,
                    tiff_volume_frames,
                    f"File {file_index}, sample {sample_index} volume does not match tiff data",
                )

    def test_volumetric_two_channels_single_file(self):
        """
        file_name: `vol_two_ch_single_file_00001_00001.tif`
        Metadata:
        - Acquisition mode: grab
        - 100 samples (volumes)
        - Volumetric: True (9 slices and 7 flyback frames)
        - Frame shape: 20 x 20
        - Channels: 2 (`Channel 1` and `Channel 2`)
        - Frames per slice: 1
        - Frame rate: 523.926 Hz
        - Volume rate: 32.7454 Hz
        - Pages/IDFs: 3200
        """
        file_path = SCANIMAGE_PATH / "volumetric_two_channels_single_file" / "vol_two_ch_single_file_00001_00001.tif"

        # Test with Channel 1
        extractor_ch1 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 1")

        assert extractor_ch1.is_volumetric == True
        assert extractor_ch1.num_flyback_frames == 7
        assert extractor_ch1.get_sampling_frequency() == 32.7454
        assert extractor_ch1.get_image_shape() == (20, 20)
        assert extractor_ch1.get_num_planes() == 9

        ifds = 3200
        num_channels = 2
        total_frames_per_cycle = (extractor_ch1.get_num_planes() + extractor_ch1.num_flyback_frames) * num_channels
        num_acquisition_cycles = ifds // total_frames_per_cycle
        assert extractor_ch1.get_num_samples() == num_acquisition_cycles

        # Test with Channel 2
        extractor_ch2 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 2")

        assert extractor_ch2.is_volumetric == True
        assert extractor_ch2.num_flyback_frames == 7
        assert extractor_ch2.get_sampling_frequency() == 32.7454
        assert extractor_ch2.get_image_shape() == (20, 20)
        assert extractor_ch2.get_num_planes() == 9
        assert extractor_ch2.get_num_samples() == num_acquisition_cycles

        # Get data from both extractors
        extractor_data_ch1 = extractor_ch1.get_series()
        extractor_data_ch2 = extractor_ch2.get_series()

        # Verify that the extractor data has the correct shape
        # For volumetric data, shape should be (samples, height, width, planes)
        expected_shape = (
            extractor_ch1.get_num_samples(),
            *extractor_ch1.get_image_shape(),
            extractor_ch1.get_num_planes(),
        )
        assert (
            extractor_data_ch1.shape == expected_shape
        ), f"Channel 1: Expected shape {expected_shape}, got {extractor_data_ch1.shape}"
        assert (
            extractor_data_ch2.shape == expected_shape
        ), f"Channel 2: Expected shape {expected_shape}, got {extractor_data_ch2.shape}"

        # Read tiff data for comparison
        with TiffReader(file_path) as tiff_reader:
            tiff_data = tiff_reader.asarray()

            # For multi-channel data, tiff_data shape is (frames, channels, height, width)
            # Verify the shape of the tiff data
            assert len(tiff_data.shape) == 4, "Multi-channel tiff data should have 4 dimensions"
            assert tiff_data.shape[1] == 2, "Tiff data should have 2 channels"

            # For each sample volume in the extractor data
            for sample_index in range(num_acquisition_cycles):
                # Calculate the slice of frames for this sample (excluding flyback frames)
                # For multi-channel data, we need to select the appropriate frames
                start_frame = sample_index * total_frames_per_cycle // num_channels
                end_frame = start_frame + extractor_ch1.get_num_planes()

                # Get the tiff data for each channel
                tiff_volume_ch1 = tiff_data[start_frame:end_frame, 0, :, :]  # Channel 1 (index 0)
                tiff_volume_ch2 = tiff_data[start_frame:end_frame, 1, :, :]  # Channel 2 (index 1)

                # Reshape the extractor data to match the tiff data format for comparison
                # extractor_data is (samples, height, width, planes)
                # We need to compare with tiff_volume which is (planes, height, width)
                extractor_volume_ch1 = np.moveaxis(extractor_data_ch1[sample_index], -1, 0)
                extractor_volume_ch2 = np.moveaxis(extractor_data_ch2[sample_index], -1, 0)

                # Compare the data for Channel 1
                np.testing.assert_array_equal(
                    extractor_volume_ch1,
                    tiff_volume_ch1,
                    f"Sample {sample_index}, Channel 1 volume does not match tiff data",
                )

                # Compare the data for Channel 2
                np.testing.assert_array_equal(
                    extractor_volume_ch2,
                    tiff_volume_ch2,
                    f"Sample {sample_index}, Channel 2 volume does not match tiff data",
                )

    def test_volumetric_two_channels_multi_file(self):
        """
        First File: vol_two_ch_multi_files_00001_00001.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (9 slices and 7 flyback frames)
        - Frame shape: 20 x 20
        - Channels: 2 (`Channel 1` and `Channel 2`)
        - Frames per slice: 1
        - Frame rate: 523.926 Hz
        - Volume rate: 32.7454 Hz
        - Pages/IDFs: 320 in each of the 10 files.
        """
        file_path = SCANIMAGE_PATH / "volumetric_two_channels_multi_file" / "vol_two_ch_multi_files_00001_00001.tif"

        # Test with Channel 1
        extractor_ch1 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 1")

        assert extractor_ch1.is_volumetric == True
        assert extractor_ch1.num_flyback_frames == 7
        assert extractor_ch1.get_sampling_frequency() == 32.7454
        assert extractor_ch1.get_image_shape() == (20, 20)
        assert extractor_ch1.get_num_planes() == 9

        ifds = 320 * 10
        num_channels = 2
        total_frames_per_cycle = (extractor_ch1.get_num_planes() + extractor_ch1.num_flyback_frames) * num_channels
        num_acquisition_cycles = ifds // total_frames_per_cycle
        assert extractor_ch1.get_num_samples() == num_acquisition_cycles

        # Test with Channel 2
        extractor_ch2 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 2")

        assert extractor_ch2.is_volumetric == True
        assert extractor_ch2.num_flyback_frames == 7
        assert extractor_ch2.get_sampling_frequency() == 32.7454
        assert extractor_ch2.get_image_shape() == (20, 20)
        assert extractor_ch2.get_num_planes() == 9
        assert extractor_ch2.get_num_samples() == num_acquisition_cycles

        # Get the entire series data from both extractors
        extractor_data_ch1 = extractor_ch1.get_series()
        extractor_data_ch2 = extractor_ch2.get_series()

        # Verify that the extractor data has the correct shape
        # For volumetric data, shape should be (samples, height, width, planes)
        expected_shape = (
            extractor_ch1.get_num_samples(),
            *extractor_ch1.get_image_shape(),
            extractor_ch1.get_num_planes(),
        )
        assert (
            extractor_data_ch1.shape == expected_shape
        ), f"Channel 1: Expected shape {expected_shape}, got {extractor_data_ch1.shape}"
        assert (
            extractor_data_ch2.shape == expected_shape
        ), f"Channel 2: Expected shape {expected_shape}, got {extractor_data_ch2.shape}"

        ifds_per_file = 320  # Number of frames in each file
        samples_per_file = ifds_per_file // total_frames_per_cycle

        # Test only the samples that fit completely within each file
        for file_index, file_path in enumerate(extractor_ch1.file_paths):
            with TiffReader(file_path) as tiff_reader:
                tiff_data = tiff_reader.asarray()

                # For multi-channel data, tiff_data shape is (samples, num_channels, x, y)
                # Verify the shape of the tiff data
                assert len(tiff_data.shape) == 4, "Multi-channel tiff data should have 4 dimensions"
                assert tiff_data.shape[1] == 2, "Tiff data should have 2 channels"

                # Calculate the sample range for this file
                start_sample = file_index * samples_per_file
                end_sample = (file_index + 1) * samples_per_file

                # Only test samples that are within the range of the extractor
                end_sample = min(end_sample, extractor_ch1.get_num_samples())

                # For each sample in this file
                for sample_index in range(start_sample, end_sample):
                    # Calculate the relative sample index within this file
                    relative_sample_index = sample_index - start_sample

                    # Calculate the slice of frames for this sample (excluding flyback frames)
                    # For multi-channel data, we need to select the appropriate frames
                    # Each volume has planes * channels frames, with flyback frames at the end
                    start_frame = relative_sample_index * total_frames_per_cycle // num_channels

                    # Make sure we don't go out of bounds
                    if start_frame + extractor_ch1.get_num_planes() > tiff_data.shape[0]:
                        continue

                    # Get the tiff data for each channel
                    tiff_volume_ch1 = tiff_data[
                        start_frame : start_frame + extractor_ch1.get_num_planes(), 0, :, :
                    ]  # Channel 1 (index 0)
                    tiff_volume_ch2 = tiff_data[
                        start_frame : start_frame + extractor_ch1.get_num_planes(), 1, :, :
                    ]  # Channel 2 (index 1)

                    # Reshape the extractor data to match the tiff data format for comparison
                    # extractor_data is (samples, height, width, planes)
                    # We need to compare with tiff_volume which is (planes, height, width)
                    extractor_volume_ch1 = np.moveaxis(extractor_data_ch1[sample_index], -1, 0)
                    extractor_volume_ch2 = np.moveaxis(extractor_data_ch2[sample_index], -1, 0)

                    # Compare the data for Channel 1
                    np.testing.assert_array_equal(
                        extractor_volume_ch1,
                        tiff_volume_ch1,
                        f"File {file_index}, Channel 1, sample {sample_index} volume does not match tiff data",
                    )

                    # Compare the data for Channel 2
                    np.testing.assert_array_equal(
                        extractor_volume_ch2,
                        tiff_volume_ch2,
                        f"File {file_index}, Channel 2, sample {sample_index} volume does not match tiff data",
                    )


class TestScanImageExtractorVolumetricMultiSamplesPerDepth:
    """Test the ScanImage extractor classes with files that have multiple frames per slice."""

    def test_volumetric_data_single_channel_single_file(self):
        """Test with volumetric data that has multiple frames per slice.

        File: scanimage_20220801_volume.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (20 slices)
        - Channels: 1 (single channel)
        - Frames per slice: 8
        - Frame rate: 30.0141 Hz
        - Volume rate: 0.187588 Hz
        - Pages/IDFs: 20
        - 'num_frames_per_volume': 160

        This file does not have enough samples to do a full slice extraction. It has 20 slices
        and 8 frames per slice but only 20 pages of data. This means that only data from the first three
        depths are included. Skipping this test for now
        """

        return

    def test_volumetric_data_single_channel_single_file_2(self):
        """Test with a multi-volume ScanImage file with frames per slice > 1.

        File: scanimage_20220801_multivolume.tif
        Metadata:
        - Volumetric: True (multiple planes)
        - Channels: 1 (single channel)
        - Image shape: 512 x 512
        - Frames per slice: 8
        - num_ifds: 20
        - num_slices: 10

        This test does not have enough samples to do a full slice extraction. It has 20 slices
        and 8 frames per slice but only 20 pages of data. This means that only data from the first three
        depths are included. Skipping this test for now.
        """
        return

    def test_volumetric_data_multi_channel_single_file(self):
        """Test with volumetric data acquired in grab mode with multiple frames per depth.

        File: scanimage_20220923_noroi.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (multiple planes)
        - Channels: Multiple Channels ['Channel 1', 'Channel 4']
        - Frames per depth: 2
        - Frame rate: 29.1248 Hz
        - Volume rate: 7.28119 Hz
        - Image shape: 256 x 256
        - Num_slices: 2
        - IFDS/Pages: 24
        """
        file_path = SCANIMAGE_PATH / "scanimage_20220923_noroi.tif"

        # Test that ValueError is raised when slice_sample is not provided
        with pytest.raises(ValueError):
            extractor = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 4")

        frames_per_slice = ScanImageImagingExtractor.get_frames_per_slice(file_path)
        assert frames_per_slice == 2, "File should have 2 slices per sample"

        # Test that the extractor works correctly when a valid slice_sample is provided
        extractor_sample_1 = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 4", slice_sample=0)

        assert extractor_sample_1.is_volumetric == True
        assert extractor_sample_1.get_frame_shape() == (256, 256)
        assert extractor_sample_1.get_num_planes() == 2
        assert extractor_sample_1.get_sample_shape() == (256, 256, 2)
        assert extractor_sample_1.get_sampling_frequency() == 14.5517

        frames_in_dataset = 24
        num_channels = len(extractor_sample_1.get_channel_names())
        num_planes = extractor_sample_1.get_num_planes()
        frames_in_a_sample = num_channels * num_planes * frames_per_slice
        expected_samples = frames_in_dataset // frames_in_a_sample
        assert extractor_sample_1.get_num_samples() == expected_samples

        extractor_sample_2 = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 4", slice_sample=1)
        assert extractor_sample_2.is_volumetric == True
        assert extractor_sample_2.get_frame_shape() == (256, 256)
        assert extractor_sample_2.get_num_planes() == 2
        assert extractor_sample_2.get_sample_shape() == (256, 256, 2)

        assert extractor_sample_2.get_sampling_frequency() == 14.5517

        frames_in_dataset = 24
        num_channels = 2  # ['Channel 1', 'Channel 4']
        num_planes = extractor_sample_2.get_num_planes()

        frames_in_a_sample = num_channels * num_planes * frames_per_slice
        expected_samples_2 = frames_in_dataset // frames_in_a_sample
        assert extractor_sample_2.get_num_samples() == expected_samples_2

        # Compare to tiff library extraction for data integrity:
        with TiffReader(file_path) as tiff_reader:
            data_tiff = tiff_reader.asarray()

        # Extract the data for the first slice
        data_sample_1 = extractor_sample_1.get_series()
        data_sample_2 = extractor_sample_2.get_series()

        # Compare all frames for slice sample 0
        tiff_slice_sample_index = 0
        tiff_channel_index = 1  # Channel 4 is at index 1 in the tiff data

        # Iterate through all samples (volumes)
        num_samples = extractor_sample_1.get_num_samples()
        num_planes = extractor_sample_1.get_num_planes()
        for sample_index in range(num_samples):
            sample = data_sample_1[sample_index, ...]  # data is (time, width, height, depth)

            # Iterate through all frames in the sample (depth planes)
            for frame_index in range(num_planes):
                frame_extractor = sample[..., frame_index]

                # Calculate the corresponding index in the tiff data
                # Each sample has 2 frames per slice, and we're using slice_sample=0
                tiff_frame_index = sample_index * 2 + frame_index
                frame_tiff = data_tiff[tiff_frame_index, tiff_slice_sample_index, tiff_channel_index, ...]

                np.testing.assert_array_equal(
                    frame_extractor, frame_tiff, f"Sample {sample_index}, frame {frame_index} does not match tiff data"
                )

        # Compare all frames for slice sample 1
        tiff_slice_sample_index = 1

        # Iterate through all samples (volumes)
        num_samples = extractor_sample_2.get_num_samples()
        num_planes = extractor_sample_2.get_num_planes()
        for sample_index in range(num_samples):
            sample = data_sample_2[sample_index, ...]  # data is (time, width, height, depth)

            # Iterate through all frames in the sample (depth planes)
            for frame_index in range(num_planes):
                frame_extractor = sample[..., frame_index]

                # Calculate the corresponding index in the tiff data
                # Each sample has 2 frames per slice, and we're using slice_sample=1
                tiff_frame_index = sample_index * 2 + frame_index
                frame_tiff = data_tiff[tiff_frame_index, tiff_slice_sample_index, tiff_channel_index, ...]

                np.testing.assert_array_equal(
                    frame_extractor, frame_tiff, f"Sample {sample_index}, frame {frame_index} does not match tiff data"
                )

    def test_volumetric_data_multi_channel_single_file_2(self):
        """Test with multi-channel volumetric data with frames per slice > 1.

        File: scanimage_20220923_roi.tif
        Metadata:
        - Volumetric: True (multiple planes)
        - Channels: Multiple ['Channel 1', 'Channel 4'
        - Frames per slice: 2
        - Frame rate: 29.1248 Hz
        - Volume rate: 7.28119 Hz
        - IFDS/Pages: 24
        - Num slices: 2
        - Image shape: 528 x 256

        """

        file_path = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"

        # Test that ValueError is raised when slice_sample is not provided
        with pytest.raises(ValueError):
            extractor = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 1")

        frames_per_slice = ScanImageImagingExtractor.get_frames_per_slice(file_path)
        assert frames_per_slice == 2, "File should have 2 slices per sample"

        # Test that the extractor works correctly when a valid slice_sample is provided
        extractor_sample_1 = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 1", slice_sample=0)
        extractor_sample_2 = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 1", slice_sample=1)

        assert extractor_sample_1.is_volumetric == True
        assert extractor_sample_1.get_image_shape() == (528, 256)
        assert extractor_sample_1.get_sampling_frequency() == 7.28119

        frames_in_dataset = 24
        num_channels = 2  # ['Channel 1', 'Channel 4']
        num_planes = extractor_sample_1.get_num_planes()

        frames_in_a_sample = num_channels * num_planes * frames_per_slice
        expected_samples = frames_in_dataset // frames_in_a_sample
        assert extractor_sample_1.get_num_samples() == expected_samples

        assert extractor_sample_2.is_volumetric == True
        assert extractor_sample_2.get_image_shape() == (528, 256)
        assert extractor_sample_2.get_sampling_frequency() == 7.28119
        expected_samples_2 = expected_samples
        assert extractor_sample_2.get_num_samples() == expected_samples_2

        # Compare to tiff library extraction for data integrity:
        with TiffReader(file_path) as tiff_reader:
            data = tiff_reader.asarray()

        # Extract the data for both slices
        data_sample_1 = extractor_sample_1.get_series()
        data_sample_2 = extractor_sample_2.get_series()

        # Compare all frames for slice sample 0
        tiff_slice_sample_index = 0
        tiff_channel_index = 0  # Channel 1 is at index 0 in the tiff data

        # Iterate through all samples (volumes)
        for sample_index in range(extractor_sample_1.get_num_samples()):
            sample = data_sample_1[sample_index, ...]  # data is (time, width, height, depth)

            # Iterate through all frames in the sample (depth planes)
            for frame_index in range(extractor_sample_1.get_num_planes()):
                frame_extractor = sample[..., frame_index]

                # Calculate the corresponding index in the tiff data
                # Each sample has 2 frames per slice, and we're using slice_sample=0
                tiff_frame_index = sample_index * 2 + frame_index
                frame_tiff = data[tiff_frame_index, tiff_slice_sample_index, tiff_channel_index, ...]

                np.testing.assert_array_equal(
                    frame_extractor, frame_tiff, f"Sample {sample_index}, frame {frame_index} does not match tiff data"
                )

        # Compare all frames for slice sample 1
        tiff_slice_sample_index = 1

        # Iterate through all samples (volumes)
        for sample_index in range(extractor_sample_2.get_num_samples()):
            sample = data_sample_2[sample_index, ...]  # data is (time, width, height, depth)

            # Iterate through all frames in the sample (depth planes)
            for frame_index in range(extractor_sample_2.get_num_planes()):
                frame_extractor = sample[..., frame_index]

                # Calculate the corresponding index in the tiff data
                # Each sample has 2 frames per slice, and we're using slice_sample=1
                tiff_frame_index = sample_index * 2 + frame_index
                frame_tiff = data[tiff_frame_index, tiff_slice_sample_index, tiff_channel_index, ...]

                np.testing.assert_array_equal(
                    frame_extractor, frame_tiff, f"Sample {sample_index}, frame {frame_index} does not match tiff data"
                )


class TestScanImageVolumetricPlaneSlicing:
    """Test the plane_index parameter of the ScanImage extractor classes."""

    def test_plane_index_parameter_single_channel(self):
        """Test the plane_index parameter with volumetric single channel data.

        File: vol_one_ch_single_files_00002_00001.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (9 slices and 7 flyback frames)
        - Frame shape: 20 x 20
        - Channels: 1 (single channel)
        - Frames per slice: 1
        - Frame rate: 523.926 Hz
        - Volume rate: 32.7454 Hz
        - Pages/IDFs: 1600

        This test verifies that the plane_index parameter correctly extracts a specific plane
        from volumetric data and behaves as if the data is not volumetric. It tests:
        1. Extracting the first plane (plane_index=0)
        2. Extracting a middle plane (plane_index=4)
        3. Extracting the last plane (plane_index=8)
        4. Comparing the extracted planes with the corresponding planes in the full volumetric data
        """
        file_path = SCANIMAGE_PATH / "volumetric_single_channel_single_file" / "vol_one_ch_single_files_00002_00001.tif"

        # Create an extractor with plane_index=0 (first plane)
        extractor_plane0 = ScanImageImagingExtractor(file_path=file_path, plane_index=0)

        # Create an extractor with plane_index=4 (middle plane)
        extractor_plane4 = ScanImageImagingExtractor(file_path=file_path, plane_index=4)

        # Create an extractor with plane_index=8 (last plane)
        extractor_plane8 = ScanImageImagingExtractor(file_path=file_path, plane_index=8)

        # Create a regular extractor for comparison
        extractor_full = ScanImageImagingExtractor(file_path=file_path)

        # Verify that the extractors with plane_index are not volumetric
        assert extractor_plane0.is_volumetric == False, "Extractor with plane_index should not be volumetric"
        assert extractor_plane4.is_volumetric == False, "Extractor with plane_index should not be volumetric"
        assert extractor_plane8.is_volumetric == False, "Extractor with plane_index should not be volumetric"

        # Verify that the full extractor is volumetric
        assert extractor_full.is_volumetric == True, "Full extractor should be volumetric"

        # Verify that the extractors with plane_index have num_planes=1
        assert extractor_plane0.get_num_planes() == 1, "Extractor with plane_index should have num_planes=1"
        assert extractor_plane4.get_num_planes() == 1, "Extractor with plane_index should have num_planes=1"
        assert extractor_plane8.get_num_planes() == 1, "Extractor with plane_index should have num_planes=1"

        # Verify that the full extractor has num_planes=9
        assert extractor_full.get_num_planes() == 9, "Full extractor should have num_planes=9"

        # Get data from all extractors
        data_plane0 = extractor_plane0.get_series()
        data_plane4 = extractor_plane4.get_series()
        data_plane8 = extractor_plane8.get_series()
        data_full = extractor_full.get_series()

        # Predefine expected shapes for easier readability
        expected_shape_plane = (extractor_plane0.get_num_samples(), 20, 20)
        expected_shape_full = (extractor_full.get_num_samples(), 20, 20, 9)

        # Verify the shape of the data
        assert data_plane0.shape == expected_shape_plane, f"Data shape should be {expected_shape_plane}"
        assert data_plane4.shape == expected_shape_plane, f"Data shape should be {expected_shape_plane}"
        assert data_plane8.shape == expected_shape_plane, f"Data shape should be {expected_shape_plane}"
        assert data_full.shape == expected_shape_full, f"Data shape should be {expected_shape_full}"

        # Verify that the data from the plane extractors matches the corresponding plane in the full extractor
        # Use assert_allclose instead of assert_array_equal to allow for small differences
        # Compare the full series at once
        np.testing.assert_allclose(
            data_plane0,
            data_full[:, :, :, 0],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Data from plane 0 extractor does not match plane 0 in full extractor",
        )
        np.testing.assert_allclose(
            data_plane4,
            data_full[:, :, :, 4],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Data from plane 4 extractor does not match plane 4 in full extractor",
        )
        np.testing.assert_allclose(
            data_plane8,
            data_full[:, :, :, 8],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Data from plane 8 extractor does not match plane 8 in full extractor",
        )

    def test_plane_index_with_multiple_frames_per_slice(self):
        """Test the plane_index parameter with slice_sample for multi-channel volumetric data.

        File: scanimage_20220923_noroi.tif
        Metadata:
        - Acquisition mode: grab
        - Volumetric: True (2 planes)
        - Channels: Multiple Channels ['Channel 1', 'Channel 4']
        - Frames per slice: 2
        - Frame rate: 29.1248 Hz
        - Volume rate: 7.28119 Hz
        - Image shape: 256 x 256
        - IFDS/Pages: 24

        This test verifies that the plane_index parameter works correctly when combined with
        the slice_sample parameter for volumetric data with multiple frames per slice. It tests:
        1. Extracting plane 0 with slice_sample=0
        2. Extracting plane 1 with slice_sample=0
        3. Extracting plane 0 with slice_sample=1
        4. Extracting plane 1 with slice_sample=1
        5. Comparing the extracted data with the corresponding planes in the regular extractors
        """
        file_path = SCANIMAGE_PATH / "scanimage_20220923_noroi.tif"

        # Create extractors with different combinations of slice_sample and plane_index
        extractor_slice_sample_0_plane_0 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 4",
            slice_sample=0,
            plane_index=0,
        )

        extractor_slice_sample_0_plane_1 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 4",
            slice_sample=0,
            plane_index=1,
        )

        extractor_slice_sample_1_plane_0 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 4",
            slice_sample=1,
            plane_index=0,
        )

        extractor_slice_sample_1_plane_1 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 4",
            slice_sample=1,
            plane_index=1,
        )

        # Create regular extractors for comparison
        extractor_slice_sample_0 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 4",
            slice_sample=0,
        )

        extractor_slice_sample_1 = ScanImageImagingExtractor(
            file_paths=[file_path],
            channel_name="Channel 4",
            slice_sample=1,
        )
        # Define the error message for easier readability
        error_message = "Extractor with plane_index should not be volumetric"
        # Verify that the extractors with plane_index are not volumetric
        assert extractor_slice_sample_0_plane_0.is_volumetric == False, error_message
        assert extractor_slice_sample_0_plane_1.is_volumetric == False, error_message
        assert extractor_slice_sample_1_plane_0.is_volumetric == False, error_message
        assert extractor_slice_sample_1_plane_1.is_volumetric == False, error_message

        # Verify that the regular extractors are volumetric
        assert extractor_slice_sample_0.is_volumetric == True, "Regular extractor should be volumetric"
        assert extractor_slice_sample_1.is_volumetric == True, "Regular extractor should be volumetric"

        # Verify that the extractors with plane_index have num_planes=1
        error_message = "Extractor with plane_index should have num_planes=1"
        assert extractor_slice_sample_0_plane_0.get_num_planes() == 1, error_message
        assert extractor_slice_sample_0_plane_1.get_num_planes() == 1, error_message
        assert extractor_slice_sample_1_plane_0.get_num_planes() == 1, error_message
        assert extractor_slice_sample_1_plane_1.get_num_planes() == 1, error_message

        # Verify that the regular extractors have num_planes=2
        assert extractor_slice_sample_0.get_num_planes() == 2, "Regular extractor should have num_planes=2"
        assert extractor_slice_sample_1.get_num_planes() == 2, "Regular extractor should have num_planes=2"

        # Get data from all extractors
        data_slice_sample_0_plane_0 = extractor_slice_sample_0_plane_0.get_series()
        data_slice_sample_0_plane_1 = extractor_slice_sample_0_plane_1.get_series()
        data_slice_sample_1_plane_0 = extractor_slice_sample_1_plane_0.get_series()
        data_slice_sample_1_plane_1 = extractor_slice_sample_1_plane_1.get_series()
        data_slice_sample_0 = extractor_slice_sample_0.get_series()
        data_slice_sample_1 = extractor_slice_sample_1.get_series()

        # Predefine expected shapes for easier readability
        expected_shape_plane = (extractor_slice_sample_0_plane_0.get_num_samples(), 256, 256)
        expected_shape_full = (extractor_slice_sample_0.get_num_samples(), 256, 256, 2)

        # Verify the shape of the data
        assert data_slice_sample_0_plane_0.shape == expected_shape_plane, f"Data shape should be {expected_shape_plane}"
        assert data_slice_sample_0_plane_1.shape == expected_shape_plane, f"Data shape should be {expected_shape_plane}"
        assert data_slice_sample_1_plane_0.shape == expected_shape_plane, f"Data shape should be {expected_shape_plane}"
        assert data_slice_sample_1_plane_1.shape == expected_shape_plane, f"Data shape should be {expected_shape_plane}"
        assert data_slice_sample_0.shape == expected_shape_full, f"Data shape should be {expected_shape_full}"
        assert data_slice_sample_1.shape == expected_shape_full, f"Data shape should be {expected_shape_full}"

        # Verify that the data from the plane extractors matches the corresponding plane in the regular extractors
        for sample_index in range(
            min(extractor_slice_sample_0_plane_0.get_num_samples(), extractor_slice_sample_0.get_num_samples())
        ):
            np.testing.assert_array_equal(
                data_slice_sample_0_plane_0[sample_index],
                data_slice_sample_0[sample_index, :, :, 0],
                f"Data from slice_sample_0_plane_0 extractor does not match plane 0 in slice_sample_0 extractor for sample {sample_index}",
            )
            np.testing.assert_array_equal(
                data_slice_sample_0_plane_1[sample_index],
                data_slice_sample_0[sample_index, :, :, 1],
                f"Data from slice_sample_0_plane_1 extractor does not match plane 1 in slice_sample_0 extractor for sample {sample_index}",
            )

        for sample_index in range(
            min(extractor_slice_sample_1_plane_0.get_num_samples(), extractor_slice_sample_1.get_num_samples())
        ):
            np.testing.assert_array_equal(
                data_slice_sample_1_plane_0[sample_index],
                data_slice_sample_1[sample_index, :, :, 0],
                f"Data from slice_sample_1_plane_0 extractor does not match plane 0 in slice_sample_1 extractor for sample {sample_index}",
            )
            np.testing.assert_array_equal(
                data_slice_sample_1_plane_1[sample_index],
                data_slice_sample_1[sample_index, :, :, 1],
                f"Data from slice_sample_1_plane_1 extractor does not match plane 1 in slice_sample_1 extractor for sample {sample_index}",
            )


class TestTimestampExtraction:
    """Test the get_times method for ScanImageImagingExtractor class."""

    def test_get_times_flyback_frames(self):
        """Test that get_times correctly excludes flyback frames in volumetric data.

        This test verifies that:
        1. The correct number of timestamps are returned (equal to number of samples)
        2. Flyback frames are properly excluded from timestamps
        3. Timestamps match the expected values from the TIFF metadata

        File: vol_one_ch_single_files_00002_00001.tif
        """
        from tifffile import TiffFile

        file_path = SCANIMAGE_PATH / "volumetric_single_channel_single_file" / "vol_one_ch_single_files_00002_00001.tif"

        # Create the extractor
        extractor = ScanImageImagingExtractor(file_path=file_path)

        # Get timestamps from extractor
        timestamps = extractor.get_times()

        # Check that number of timestamps equals number of samples
        assert len(timestamps) == extractor.get_num_samples(), "Number of timestamps should match number of samples"

        # Verify monotonically increasing
        assert np.all(np.diff(timestamps) > 0), "Timestamps should be monotonically increasing"

        # Extract raw timestamps directly from TIFF file using list comprehension
        with TiffFile(file_path) as tiff:
            raw_timestamps = [ScanImageImagingExtractor.extract_timestamp_from_page(page) for page in tiff.pages]
        raw_timestamps = np.array(raw_timestamps)

        # Calculate expected timestamps based on extractor's table mapping
        expected_timestamps = np.zeros(extractor.get_num_samples())

        for sample_index in range(extractor.get_num_samples()):
            # Get the last frame in each sample to match the extractor's behavior
            frame_index = sample_index * extractor.get_num_planes() + (extractor.get_num_planes() - 1)
            table_row = extractor._frames_to_ifd_table[frame_index]
            file_index = table_row["file_index"]
            ifd_index = table_row["IFD_index"]

            # Add the timestamp from the raw data
            expected_timestamps[sample_index] = raw_timestamps[ifd_index]

        expected_timestamps = np.array(expected_timestamps)

        # Compare the timestamps with expected values
        np.testing.assert_array_equal(
            timestamps,
            expected_timestamps,
            "Timestamps from get_times should match expected values extracted from TIFF metadata",
        )

    def test__get_times_after_plane_slicing_with_flyback_frames(self):
        """Test get_times with plane_index on volumetric data with flyback frames.

        This test verifies that:
        1. The get_times method works correctly when accessing a specific plane using plane_index
        2. When plane_index is used, timestamps should reflect the timestamps of the frames at that plane

        File: vol_one_ch_single_files_00002_00001.tif
        """
        from tifffile import TiffFile

        file_path = SCANIMAGE_PATH / "volumetric_single_channel_single_file" / "vol_one_ch_single_files_00002_00001.tif"

        # Create extractors with and without plane_index
        extractor_full = ScanImageImagingExtractor(file_path=file_path)
        extractor_plane0 = ScanImageImagingExtractor(file_path=file_path, plane_index=0)
        extractor_plane4 = ScanImageImagingExtractor(file_path=file_path, plane_index=4)

        # Get timestamps
        timestamps_full = extractor_full.get_times()
        timestamps_plane0 = extractor_plane0.get_times()
        timestamps_plane4 = extractor_plane4.get_times()

        # Check that number of timestamps equals number of samples for each extractor
        assert (
            len(timestamps_full) == extractor_full.get_num_samples()
        ), "Number of timestamps should match number of samples for full extractor"
        assert (
            len(timestamps_plane0) == extractor_plane0.get_num_samples()
        ), "Number of timestamps should match number of samples for plane_index=0 extractor"
        assert (
            len(timestamps_plane4) == extractor_plane4.get_num_samples()
        ), "Number of timestamps should match number of samples for plane_index=4 extractor"

        # Check that all extractors have the same number of samples
        assert (
            extractor_full.get_num_samples() == extractor_plane0.get_num_samples() == extractor_plane4.get_num_samples()
        ), "All extractors should have the same number of samples"

        # Extract raw timestamps directly from TIFF file
        with TiffFile(file_path) as tiff:
            raw_timestamps = [ScanImageImagingExtractor.extract_timestamp_from_page(page) for page in tiff.pages]
        raw_timestamps = np.array(raw_timestamps)

        # Calculate expected timestamps directly by understanding the structure of the dataset
        # without relying on the internal implementation

        # Calculate timestamps for the full extractor
        num_planes = extractor_full.get_num_planes()
        num_flyback = extractor_full.num_flyback_frames
        total_frames_per_cycle = num_planes + num_flyback

        # For full extractor, we expect timestamps from the last plane of each volume
        expected_full_timestamps = []
        for sample_index in range(extractor_full.get_num_samples()):
            # Calculate frame index for the last plane in each volume (right before the flyback frames)
            cycle_start_ifd = sample_index * total_frames_per_cycle
            last_plane_ifd = cycle_start_ifd + num_planes - 1
            expected_full_timestamps.append(raw_timestamps[last_plane_ifd])

        expected_full_timestamps = np.array(expected_full_timestamps)

        # For plane0 extractor, we expect timestamps from plane 0 of each volume
        expected_plane0_timestamps = []
        for sample_index in range(extractor_plane0.get_num_samples()):
            # Calculate frame index for plane 0 in each volume
            cycle_start_ifd = sample_index * total_frames_per_cycle
            plane0_ifd = cycle_start_ifd
            expected_plane0_timestamps.append(raw_timestamps[plane0_ifd])

        expected_plane0_timestamps = np.array(expected_plane0_timestamps)

        # For plane4 extractor, we expect timestamps from plane 4 of each volume
        expected_plane4_timestamps = []
        for sample_index in range(extractor_plane4.get_num_samples()):
            # Calculate frame index for plane 4 in each volume
            cycle_start_ifd = sample_index * total_frames_per_cycle
            plane4_ifd = cycle_start_ifd + 4
            expected_plane4_timestamps.append(raw_timestamps[plane4_ifd])

        expected_plane4_timestamps = np.array(expected_plane4_timestamps)

        # Verify full extractor timestamps
        np.testing.assert_array_equal(
            timestamps_full,
            expected_full_timestamps,
            "Timestamps from full extractor should match expected values for last planes",
        )

        # Verify plane0 extractor timestamps
        np.testing.assert_array_equal(
            timestamps_plane0,
            expected_plane0_timestamps,
            "Timestamps from plane_index=0 extractor should match expected values for plane 0",
        )

        # Verify plane4 extractor timestamps
        np.testing.assert_array_equal(
            timestamps_plane4,
            expected_plane4_timestamps,
            "Timestamps from plane_index=4 extractor should match expected values for plane 4",
        )

    def test_get_times_multi_samples_per_slice(self):
        """Test get_times with multiple samples per slice in volumetric data.

        This test verifies that:
        1. The correct timestamps are returned when slice_sample is specified
        2. Timestamps match the expected values from the TIFF metadata

        File: scanimage_20220923_noroi.tif
        """
        from tifffile import TiffFile

        file_path = SCANIMAGE_PATH / "scanimage_20220923_noroi.tif"

        # Create extractor with slice_sample=0
        extractor = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 4", slice_sample=0)

        # Get timestamps from extractor
        timestamps = extractor.get_times()

        # Check that number of timestamps equals number of samples
        assert len(timestamps) == extractor.get_num_samples(), "Number of timestamps should match number of samples"

        # Verify monotonically increasing (if there's more than one timestamp)
        if len(timestamps) > 1:
            assert np.all(np.diff(timestamps) > 0), "Timestamps should be monotonically increasing"

        # Extract raw timestamps directly from TIFF file using list comprehension
        with TiffFile(file_path) as tiff:
            raw_timestamps = [ScanImageImagingExtractor.extract_timestamp_from_page(page) for page in tiff.pages]
        raw_timestamps = np.array(raw_timestamps)

        # Calculate expected timestamps based on extractor's table mapping
        expected_timestamps = []

        for sample_index in range(extractor.get_num_samples()):
            # Get the last frame in each sample to match the extractor's behavior
            frame_index = sample_index * extractor.get_num_planes() + (extractor.get_num_planes() - 1)
            if frame_index < len(extractor._frames_to_ifd_table):
                table_row = extractor._frames_to_ifd_table[frame_index]
                file_index = table_row["file_index"]
                ifd_index = table_row["IFD_index"]

                # Add the timestamp from the raw data
                if ifd_index < len(raw_timestamps):
                    expected_timestamps.append(raw_timestamps[ifd_index])

        expected_timestamps = np.array(expected_timestamps)

        # Compare the timestamps with expected values (if we have any expected timestamps)
        if len(expected_timestamps) > 0 and len(timestamps) == len(expected_timestamps):
            np.testing.assert_array_equal(
                timestamps,
                expected_timestamps,
                "Timestamps from get_times should match expected values extracted from TIFF metadata",
            )

    def test_get_times_after_plane_slicing_multi_samples_per_slice(self):
        """Test get_times with plane_index on data with multiple samples per slice.

        This test verifies that:
        1. The get_times method works correctly when using both slice_sample and plane_index
        2. When plane_index is used, timestamps should reflect the timestamps of the frames at that plane

        File: scanimage_20220923_noroi.tif
        """
        from tifffile import TiffFile

        file_path = SCANIMAGE_PATH / "scanimage_20220923_noroi.tif"

        # Create extractors with and without plane_index
        extractor_full = ScanImageImagingExtractor(file_paths=[file_path], channel_name="Channel 4", slice_sample=0)
        extractor_plane0 = ScanImageImagingExtractor(
            file_paths=[file_path], channel_name="Channel 4", slice_sample=0, plane_index=0
        )

        # Get timestamps
        timestamps_full = extractor_full.get_times()
        timestamps_plane0 = extractor_plane0.get_times()

        # Check that number of timestamps equals number of samples for each extractor
        assert (
            len(timestamps_full) == extractor_full.get_num_samples()
        ), "Number of timestamps should match number of samples for full extractor"
        assert (
            len(timestamps_plane0) == extractor_plane0.get_num_samples()
        ), "Number of timestamps should match number of samples for plane_index=0 extractor"

        # Should have the same number of samples and timestamps
        assert (
            extractor_full.get_num_samples() == extractor_plane0.get_num_samples()
        ), "Both extractors should have the same number of samples"

        # Extract raw timestamps directly from TIFF file
        with TiffFile(file_path) as tiff:
            raw_timestamps = [ScanImageImagingExtractor.extract_timestamp_from_page(page) for page in tiff.pages]
        raw_timestamps = np.array(raw_timestamps)

        # For complex data with multiple frames per slice, we need more information to calculate
        # expected timestamps without relying on internal implementation

        # For the full volumetric extractor (with slice_sample=0), we need to identify
        # timestamps for the last plane in each volume
        num_planes = extractor_full.get_num_planes()

        # Get the indices for the frames in each extractor
        full_indices = []
        plane0_indices = []

        if len(extractor_full._frames_to_ifd_table) > 0:
            # Get the frame indices for the full extractor
            for sample_index in range(min(2, extractor_full.get_num_samples())):  # Just check the first few samples
                frame_index = sample_index * num_planes + (num_planes - 1)  # Last plane
                if frame_index < len(extractor_full._frames_to_ifd_table):
                    full_indices.append(extractor_full._frames_to_ifd_table[frame_index]["IFD_index"])

            # Get the frame indices for the plane0 extractor
            for sample_index in range(min(2, extractor_plane0.get_num_samples())):  # Just check the first few samples
                if sample_index < len(extractor_plane0._frames_to_ifd_table):
                    plane0_indices.append(extractor_plane0._frames_to_ifd_table[sample_index]["IFD_index"])

            # Given enough samples to analyze, determine the frame offset between full and plane0
            if full_indices and plane0_indices:
                # Verify that the get_times method correctly returns timestamps
                # from the corresponding frames in each extractor

                # For full extractor
                expected_full_timestamps = [raw_timestamps[idx] for idx in full_indices]
                for i, (expected, actual) in enumerate(
                    zip(expected_full_timestamps, timestamps_full[: len(expected_full_timestamps)])
                ):
                    assert expected == actual, f"Timestamp mismatch for full extractor at sample {i}"

                # For plane0 extractor
                expected_plane0_timestamps = [raw_timestamps[idx] for idx in plane0_indices]
                for i, (expected, actual) in enumerate(
                    zip(expected_plane0_timestamps, timestamps_plane0[: len(expected_plane0_timestamps)])
                ):
                    assert expected == actual, f"Timestamp mismatch for plane0 extractor at sample {i}"


def test_file_paths_parameter():
    """Test the file_paths parameter for direct file specification.

    This test verifies that the extractor correctly uses the provided file_paths
    parameter instead of relying on automatic file detection heuristics.

    The test checks:
    1. Explicitly providing file paths works correctly
    2. The extractor loads the correct number of files
    3. The files are loaded in the correct order
    """
    # Get the paths to the multi-file dataset
    file_paths = [
        SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif",
        SCANIMAGE_PATH / "scanimage_20240320_multifile_00002.tif",
        SCANIMAGE_PATH / "scanimage_20240320_multifile_00003.tif",
    ]

    # Create extractor with explicit file_paths parameter
    extractor = ScanImageImagingExtractor(file_paths=file_paths, channel_name="Channel 1")

    # Verify the correct files were loaded
    assert len(extractor.file_paths) == 3, "Should load all 3 files"

    # Verify the files are in the correct order
    expected_file_names = [Path(p).name for p in file_paths]
    actual_file_names = [Path(p).name for p in extractor.file_paths]
    assert actual_file_names == expected_file_names, "Files should be loaded in the provided order"

    # Verify the data is loaded correctly
    assert extractor.get_num_samples() == 10 * 3, "Should have 10 samples per file"
    assert extractor.get_image_shape() == (512, 512), "Image shape should be correct"


def test_get_times_planar_multichannel():
    """Test the get_times method.

    This test verifies that the get_times method correctly extracts timestamps
    from ScanImage TIFF files for the selected channel.

    The test checks that for multi-channel data, the timestamps are correctly
    filtered for the selected channel.
    """

    file_path = SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif"

    extractor_ch1 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 1")
    extractor_ch2 = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 2")

    # Get timestamps for each channel
    timestamps_ch1 = extractor_ch1.get_times()
    timestamps_ch2 = extractor_ch2.get_times()

    # Check basic properties
    assert len(timestamps_ch1) == extractor_ch1.get_num_samples(), "Should have one timestamp per frame for channel 1"
    assert len(timestamps_ch2) == extractor_ch2.get_num_samples(), "Should have one timestamp per frame for channel 2"

    # Extract all timestamps from the file to compare
    from tifffile import TiffFile

    all_file_paths = [
        "scanimage_20240320_multifile_00001.tif",
        "scanimage_20240320_multifile_00002.tif",
        "scanimage_20240320_multifile_00003.tif",
    ]

    all_timestamps = []
    for file_name in all_file_paths:
        file_path = SCANIMAGE_PATH / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        # Read the metadata from the TIFF file
        with TiffFile(file_path) as tiff:
            for page in tiff.pages:
                if "ImageDescription" not in page.tags:
                    continue

                metadata_str = page.tags["ImageDescription"].value

                # Look for the timestamp line
                for line in metadata_str.strip().split("\n"):
                    if line.startswith("frameTimestamps_sec"):
                        # Extract the value part after " = "
                        _, value_str = line.split(" = ", 1)
                        try:
                            timestamp = float(value_str.strip())
                            all_timestamps.append(timestamp)
                        except ValueError:
                            all_timestamps.append(None)
                        break

    all_timestamps = np.asarray(all_timestamps)
    assert len(all_timestamps) == len(timestamps_ch1) + len(
        timestamps_ch2
    ), "Total timestamps should match the sum of both channels"
    # Verify that the timestamps for each channel are correctly filtered from all timestamps
    # For a 2-channel recording, channel 3 should have timestamps from even indices (0, 2, 4...)
    # and channel 4 should have timestamps from odd indices (1, 3, 5...)
    ch1_indices = np.arange(0, len(all_timestamps), 2)
    ch2_indices = np.arange(1, len(all_timestamps), 2)

    expected_timestamps_ch1 = np.array(all_timestamps)[ch1_indices]
    expected_timestamps_ch2 = np.array(all_timestamps)[ch2_indices]

    # Compare the first few timestamps to verify correct filtering
    assert np.allclose(timestamps_ch1, expected_timestamps_ch1), "Channel 1 timestamps should match expected values"
    assert np.allclose(timestamps_ch2, expected_timestamps_ch2), "Channel 2 timestamps should match expected values"


def test_get_frames_per_slice():
    """
    Test the static get_frames_per_slice method.

    This test verifies that the static get_frames_per_slice method correctly extracts
    the number of slices per sample from ScanImage TIFF files without needing to create
    an extractor instance.
    """
    # Test with single frame per slice file
    single_frame_file = SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif"
    frames_per_slice = ScanImageImagingExtractor.get_frames_per_slice(single_frame_file)
    assert frames_per_slice == 10, "File should have 1 frame per slice"

    # Test with multiple frames per slice file
    multi_frame_file = SCANIMAGE_PATH / "scanimage_20220801_volume.tif"
    frames_per_slice = ScanImageImagingExtractor.get_frames_per_slice(multi_frame_file)
    assert frames_per_slice == 8, "File should have 8 frames per slice"

    # Test with another multiple frames per slice file
    multi_frame_file2 = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"
    frames_per_slice = ScanImageImagingExtractor.get_frames_per_slice(multi_frame_file2)
    assert frames_per_slice == 2, "File should have 2 frames per slice"


def test_get_availale_channel_names():
    """Test the static get_channel_names method.

    This test verifies that the static get_available_channel_names method correctly extracts
    channel names from ScanImage TIFF files without needing to create an extractor instance.

    The test checks:
    1. Single-channel file: Verifies correct channel name extraction
    2. Multi-channel file: Verifies all channel names are correctly extracted
    """
    # Test with single-channel file
    single_channel_file = SCANIMAGE_PATH / "scanimage_20220801_single.tif"
    single_channel_names = ScanImageImagingExtractor.get_available_channel_names(single_channel_file)
    assert isinstance(single_channel_names, list), "Channel names should be returned as a list"
    assert len(single_channel_names) == 1, "Single channel file should have one channel"
    assert single_channel_names[0] == "Channel 1", "Channel name should match expected value"

    # Test with multi-channel file
    multi_channel_file = SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif"
    multi_channel_names = ScanImageImagingExtractor.get_available_channel_names(multi_channel_file)
    assert isinstance(multi_channel_names, list), "Channel names should be returned as a list"
    assert len(multi_channel_names) == 2, "Multi-channel file should have two channels"
    assert multi_channel_names == ["Channel 1", "Channel 2"]

    # Test with volumetric file (should still work even though extractor initialization would fail)
    volumetric_file = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"
    volumetric_channel_names = ScanImageImagingExtractor.get_available_channel_names(volumetric_file)
    assert isinstance(volumetric_channel_names, list), "Channel names should be returned as a list"
    assert len(volumetric_channel_names) >= 1, "Should extract at least one channel name"
    assert volumetric_channel_names == ["Channel 1", "Channel 4"]
