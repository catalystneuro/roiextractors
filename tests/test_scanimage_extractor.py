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


class TestScanImageExtractors:
    """Test the ScanImage extractor classes with various ScanImage files."""

    def test_volumetric_data(self):
        """
        Test for a file acquired with ScanImage in multifile mode.
        This is single channel data in grab acquisition
        This has more than one frame per depth so it should fail.

        """
        # For multifile, we only need to provide the first file
        file_path = SCANIMAGE_PATH / "scanimage_20220801_volume.tif"

        with pytest.raises(ValueError):
            extractor = ScanImageImagingExtractor(file_path=file_path)

        # Uncomment when adding support
        # Basic properties
        # assert extractor.is_volumetric == True
        # assert extractor.get_num_frames() == 160
        # assert extractor.get_image_shape() == (512, 512)
        # assert extractor.get_sampling_frequency()  == 0.187588

        # # Check if multiple files were detected
        # assert len(extractor.file_paths)  == 1

    def test_scanimage_multifile(self):
        """
        Test for data acquired in loop acquisition so it will be multi file
        this is planar file
        there are two channels
        """

        # First file of the series should be enough to initialize
        file_path = SCANIMAGE_PATH / "scanimage_20240320_multifile_00001.tif"

        extractor = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 3")
        assert len(extractor.file_paths) == 3

        expected_files_names = [
            "scanimage_20240320_multifile_00001.tif",
            "scanimage_20240320_multifile_00002.tif",
            "scanimage_20240320_multifile_00003.tif",
        ]
        extracted_file_names = [Path(file_path).name for file_path in extractor.file_paths]

        assert extracted_file_names == expected_files_names

        assert extractor.is_volumetric == False
        samples_per_file = 10
        assert extractor.get_num_frames() == 10 * 3  # 10 frames per file
        assert extractor.get_image_shape() == (512, 512)
        assert extractor.get_sampling_frequency() == 98.83842317607626

        # for file_path in extractor.file_paths:
        #     with TiffReader(file_path) as tiff_reader:
        #         data = tiff_reader.asarray()

        #         extractor_frames = extractor.get_frames(frame_idxs=range(10))
        #         assert_array_equal(data, extractor_frames)

    def test_scanimage_noroi(self):
        """
        Data acquired in grab mode, volumetric data multi channel
        This has multiple frames per depth so it should throw an error

        """
        file_path = SCANIMAGE_PATH / "scanimage_20220923_noroi.tif"
        with pytest.raises(ValueError):
            extractor = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 3")

        # assert extractor.is_volumetric == True
        # assert extractor.get_num_frames() == 12
        # assert extractor.get_image_shape() == (256, 256)
        # assert extractor.get_sampling_frequency()  == 14.5517

    def test_scanimage_roi(self):
        """
        Multi channel data
        Volumetric data
        frames per slice 2 so it should throw an error
        """

        file_path = SCANIMAGE_PATH / "scanimage_20220923_roi.tif"
        with pytest.raises(ValueError):
            extractor = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 3")

    def test_scanimage_single(self):
        """
        Planar data
        Single channel data

        """
        # This is frame per slice 24 and should fail
        file_path = SCANIMAGE_PATH / "scanimage_20220801_single.tif"

        extractor = ScanImageImagingExtractor(file_path=file_path, channel_name="Channel 3")

        assert extractor.is_volumetric == False
        assert extractor.get_num_frames() == 3
        assert extractor.get_image_shape() == (1024, 1024)
        assert extractor.get_sampling_frequency() == 15.2379

    def test_scanimage_multivolume(self):
        """
        Test with a ScanImage multivolume file.
        Volumetric data
        Single channel
        frames per slice 8 so it should fail

        """
        file_path = SCANIMAGE_PATH / "scanimage_20220801_multivolume.tif"

        # Create multi-plane extractor
        with pytest.raises(ValueError):
            extractor = ScanImageImagingExtractor(file_path=file_path)
