import pytest
from numpy.testing import assert_array_equal
from ScanImageTiffReader import ScanImageTiffReader
from roiextractors.extractors.tiffimagingextractors.scanimagetiff_utils import (
    _get_scanimage_reader,
    extract_extra_metadata,
    parse_matlab_vector,
    parse_metadata,
    parse_metadata_v3_8,
    extract_timestamps_from_file,
)

from .setup_paths import OPHYS_DATA_PATH


def test_get_scanimage_reader():
    ScanImageTiffReader = _get_scanimage_reader()
    assert ScanImageTiffReader is not None


@pytest.mark.parametrize(
    "filename, expected_key, expected_value",
    [
        ("sample_scanimage_version_3_8.tiff", "state.software.version", "3.8"),
        ("scanimage_20220801_single.tif", "SI.VERSION_MAJOR", "2022"),
        ("scanimage_20220923_roi.tif", "SI.VERSION_MAJOR", "2023"),
    ],
)
def test_extract_extra_metadata(filename, expected_key, expected_value):
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage" / filename
    metadata = extract_extra_metadata(file_path)
    assert metadata[expected_key] == expected_value


@pytest.mark.parametrize(
    "matlab_vector, expected_vector",
    [
        ("[1 2 3]", [1, 2, 3]),
        ("[1,2,3]", [1, 2, 3]),
        ("[1, 2, 3]", [1, 2, 3]),
        ("[1;2;3]", [1, 2, 3]),
        ("[1; 2; 3]", [1, 2, 3]),
    ],
)
def test_parse_matlab_vector(matlab_vector, expected_vector):
    vector = parse_matlab_vector(matlab_vector)
    assert vector == expected_vector


@pytest.mark.parametrize(
    "filename, expected_metadata",
    [
        (
            "scanimage_20220801_single.tif",
            {
                "sampling_frequency": 15.2379,
                "num_channels": 1,
                "num_planes": 20,
                "frames_per_slice": 24,
                "channel_names": ["Channel 1"],
            },
        ),
        (
            "scanimage_20220923_roi.tif",
            {
                "sampling_frequency": 29.1248,
                "num_channels": 2,
                "num_planes": 2,
                "frames_per_slice": 2,
                "channel_names": ["Channel 1", "Channel 4"],
            },
        ),
    ],
)
def test_parse_metadata(filename, expected_metadata):
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage" / filename
    metadata = extract_extra_metadata(file_path)
    metadata = parse_metadata(metadata)
    assert metadata == expected_metadata


def test_parse_metadata_v3_8():
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage" / "sample_scanimage_version_3_8.tiff"
    metadata = extract_extra_metadata(file_path)
    metadata = parse_metadata_v3_8(metadata)
    expected_metadata = {"sampling_frequency": 3.90625, "num_channels": 1, "num_planes": 1}
    assert metadata == expected_metadata


@pytest.mark.parametrize(
    "filename, expected_timestamps",
    [
        ("scanimage_20220801_single.tif", [0.45951611, 0.98468446, 1.50985974]),
        (
            "scanimage_20220923_roi.tif",
            [
                0.0,
                0.0,
                0.03433645,
                0.03433645,
                1.04890375,
                1.04890375,
                1.08324025,
                1.08324025,
                2.12027815,
                2.12027815,
                2.15461465,
                2.15461465,
                2.7413649,
                2.7413649,
                2.7757014,
                2.7757014,
                3.23987545,
                3.23987545,
                3.27421195,
                3.27421195,
                3.844804,
                3.844804,
                3.87914055,
                3.87914055,
            ],
        ),
    ],
)
def test_extract_timestamps_from_file(filename, expected_timestamps):
    file_path = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage" / filename
    timestamps = extract_timestamps_from_file(file_path)
    assert_array_equal(timestamps, expected_timestamps)
