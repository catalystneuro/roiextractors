import pytest
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree, copy

from ScanImageTiffReader import ScanImageTiffReader
from roiextractors import ScanImageTiffSinglePlaneImagingExtractor, ScanImageTiffMultiPlaneImagingExtractor

from .setup_paths import OPHYS_DATA_PATH


@pytest.fixture(scope="module")
def scan_image_tiff_single_plane_imaging_extractor():
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage" / "scanimage_20220801_volume.tif"
    return ScanImageTiffSinglePlaneImagingExtractor(file_path=file_path)


def test_get_video(scan_image_tiff_single_plane_imaging_extractor):
    video = scan_image_tiff_single_plane_imaging_extractor.get_video()
    assert video.shape == (1, 512, 512)
