import pytest
from numpy.testing import assert_array_equal
from roiextractors.extractors.tiffimagingextractors.scanimagetiff_utils import (
    _get_scanimage_reader,
    extract_extra_metadata,
    parse_matlab_vector,
    parse_metadata,
    parse_metadata_v3_8,
    extract_timestamps_from_file,
)

from .setup_paths import OPHYS_DATA_PATH

import platform

is_m_series_mac = platform.system() == "darwin" and platform.machine() == "arm64"
if (
    is_m_series_mac
):  # Remove this check once scanimage tiff reader is available on ARM -- see https://gitlab.com/vidriotech/scanimagetiffreader-python/-/issues/31
    pytest.skip("ScanImageTiffReader does not support M-series Macs", allow_module_level=True)


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


def test_parse_matlab_vector_invalid():
    with pytest.raises(ValueError):
        parse_matlab_vector("Invalid")


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
                "roi_metadata": {
                    "imagingRoiGroup": {
                        "ver": 1,
                        "classname": "scanimage.mroi.RoiGroup",
                        "name": "Default Imaging ROI Group",
                        "UserData": None,
                        "roiUuid": "E9CD2A60E29A5EDE",
                        "roiUuiduint64": 1.684716838e19,
                        "rois": {
                            "ver": 1,
                            "classname": "scanimage.mroi.Roi",
                            "name": "Default Imaging Roi",
                            "UserData": {
                                "imagingSystem": "Imaging_RGG",
                                "fillFractionSpatial": 0.9,
                                "forceSquarePixelation": 1,
                                "forceSquarePixels": 1,
                                "scanZoomFactor": 1,
                                "scanAngleShiftFast": 0,
                                "scanAngleMultiplierSlow": 1,
                                "scanAngleShiftSlow": 0,
                                "scanRotation": 0,
                                "pixelsPerLine": 1024,
                                "linesPerFrame": 1024,
                            },
                            "roiUuid": "1B54BED0B8A25D87",
                            "roiUuiduint64": 1.969408741e18,
                            "zs": 0,
                            "scanfields": {
                                "ver": 1,
                                "classname": "scanimage.mroi.scanfield.fields.RotatedRectangle",
                                "name": "Default Imaging Scanfield",
                                "UserData": None,
                                "roiUuid": "4309FD6B19453539",
                                "roiUuiduint64": 4.830670712e18,
                                "centerXY": [0, 0],
                                "sizeXY": [18, 18],
                                "rotationDegrees": 0,
                                "enable": 1,
                                "pixelResolutionXY": [1024, 1024],
                                "pixelToRefTransform": [
                                    [0.017578125, 0, -9.008789063],
                                    [0, 0.017578125, -9.008789063],
                                    [0, 0, 1],
                                ],
                                "affine": [[18, 0, -9], [0, 18, -9], [0, 0, 1]],
                            },
                            "discretePlaneMode": 0,
                            "powers": None,
                            "pzAdjust": [],
                            "Lzs": None,
                            "interlaceDecimation": None,
                            "interlaceOffset": None,
                            "enable": 1,
                        },
                    },
                    "photostimRoiGroups": None,
                    "integrationRoiGroup": {
                        "ver": 1,
                        "classname": "scanimage.mroi.RoiGroup",
                        "name": "",
                        "UserData": None,
                        "roiUuid": "9FC266E57D28670D",
                        "roiUuiduint64": 1.151187673e19,
                        "rois": {"_ArrayType_": "double", "_ArraySize_": [1, 0], "_ArrayData_": None},
                    },
                },
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
                "roi_metadata": {
                    "imagingRoiGroup": {
                        "ver": 1,
                        "classname": "scanimage.mroi.RoiGroup",
                        "name": "MROI Imaging ROI Group",
                        "UserData": None,
                        "roiUuid": "4118A30BD7393EFF",
                        "roiUuiduint64": 4.690678283e18,
                        "rois": [
                            {
                                "ver": 1,
                                "classname": "scanimage.mroi.Roi",
                                "name": "ROI 1",
                                "UserData": None,
                                "roiUuid": "8C08C657736FBC6C",
                                "roiUuiduint64": 1.009053304e19,
                                "zs": -11178.45,
                                "scanfields": {
                                    "ver": 1,
                                    "classname": "scanimage.mroi.scanfield.fields.RotatedRectangle",
                                    "name": "",
                                    "UserData": None,
                                    "roiUuid": "2B42EE3A0B039B9E",
                                    "roiUuiduint64": 3.117315825e18,
                                    "centerXY": [0.2141430948, -6.019800333],
                                    "sizeXY": [3.616638935, 3.521464226],
                                    "rotationDegrees": 0,
                                    "enable": 1,
                                    "pixelResolutionXY": [256, 256],
                                    "pixelToRefTransform": [
                                        [0.01412749584, 0, -1.601240121],
                                        [0, 0.01375571963, -7.787410306],
                                        [0, 0, 1],
                                    ],
                                    "affine": [
                                        [3.616638935, 0, -1.594176373],
                                        [0, 3.521464226, -7.780532446],
                                        [0, 0, 1],
                                    ],
                                },
                                "discretePlaneMode": 0,
                                "powers": None,
                                "pzAdjust": [],
                                "Lzs": None,
                                "interlaceDecimation": None,
                                "interlaceOffset": None,
                                "enable": 1,
                            },
                            {
                                "ver": 1,
                                "classname": "scanimage.mroi.Roi",
                                "name": "ROI 2",
                                "UserData": None,
                                "roiUuid": "7C9E605DC6951B29",
                                "roiUuiduint64": 8.979720663e18,
                                "zs": -11178.45,
                                "scanfields": {
                                    "ver": 1,
                                    "classname": "scanimage.mroi.scanfield.fields.RotatedRectangle",
                                    "name": "",
                                    "UserData": None,
                                    "roiUuid": "A02889BA5E5501AB",
                                    "roiUuiduint64": 1.154062548e19,
                                    "centerXY": [2.664891847, 6.376705491],
                                    "sizeXY": [3.616638935, 3.759400998],
                                    "rotationDegrees": 0,
                                    "enable": 1,
                                    "pixelResolutionXY": [256, 256],
                                    "pixelToRefTransform": [
                                        [0.01412749584, 0, 0.8495086314],
                                        [0, 0.01468516015, 4.489662412],
                                        [0, 0, 1],
                                    ],
                                    "affine": [
                                        [3.616638935, 0, 0.8565723794],
                                        [0, 3.759400998, 4.497004992],
                                        [0, 0, 1],
                                    ],
                                },
                                "discretePlaneMode": 0,
                                "powers": None,
                                "pzAdjust": [],
                                "Lzs": None,
                                "interlaceDecimation": None,
                                "interlaceOffset": None,
                                "enable": 1,
                            },
                        ],
                    },
                    "photostimRoiGroups": None,
                    "integrationRoiGroup": {
                        "ver": 1,
                        "classname": "scanimage.mroi.RoiGroup",
                        "name": "",
                        "UserData": None,
                        "roiUuid": "1B4D989071535CF3",
                        "roiUuiduint64": 1.967396358e18,
                        "rois": {"_ArrayType_": "double", "_ArraySize_": [1, 0], "_ArrayData_": None},
                    },
                },
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
