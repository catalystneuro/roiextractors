import platform

import pytest
from numpy.testing import assert_array_equal

from roiextractors import (
    ScanImageTiffMultiPlaneImagingExtractor,
    ScanImageTiffMultiPlaneMultiFileImagingExtractor,
    ScanImageTiffSinglePlaneImagingExtractor,
    ScanImageTiffSinglePlaneMultiFileImagingExtractor,
)
from roiextractors.extractors.tiffimagingextractors.scanimagetiff_utils import (
    _get_scanimage_reader,
    extract_extra_metadata,
    parse_metadata,
)

from .setup_paths import OPHYS_DATA_PATH

is_m_series_mac = platform.system() == "Darwin" and platform.machine() == "arm64"
if (
    is_m_series_mac
):  # Remove this check once scanimage tiff reader is available on ARM -- see https://gitlab.com/vidriotech/scanimagetiffreader-python/-/issues/31
    pytest.skip("ScanImageTiffReader does not support M-series Macs", allow_module_level=True)


@pytest.fixture(scope="module")
def file_path():
    return OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage" / "scanimage_20220923_roi.tif"


@pytest.fixture(scope="module")
def expected_properties():
    return dict(
        sampling_frequency=7.28119,
        num_channels=2,
        num_planes=2,
        frames_per_slice=2,
        channel_names=["Channel 1", "Channel 4"],
        plane_names=["0", "1"],
        image_size=(528, 256),
        num_frames=6,
        dtype="int16",
    )


@pytest.fixture(
    scope="module",
    params=[
        dict(channel_name="Channel 1", plane_name="0"),
        dict(channel_name="Channel 1", plane_name="1"),
        dict(channel_name="Channel 4", plane_name="0"),
        dict(channel_name="Channel 4", plane_name="1"),
    ],
)
def scan_image_tiff_single_plane_imaging_extractor(request, file_path):
    return ScanImageTiffSinglePlaneImagingExtractor(file_path=file_path, **request.param)


@pytest.mark.parametrize("channel_name, plane_name", [("Invalid Channel", "0"), ("Channel 1", "Invalid Plane")])
def test_ScanImageTiffSinglePlaneImagingExtractor__init__invalid(file_path, channel_name, plane_name):
    with pytest.raises(ValueError):
        ScanImageTiffSinglePlaneImagingExtractor(file_path=file_path, channel_name=channel_name, plane_name=plane_name)


def test_ScanImageTiffSinglePlaneImagingExtractor__init__metadata_provided(file_path):
    metadata = extract_extra_metadata(file_path)
    parsed_metadata = parse_metadata(metadata)
    extractor = ScanImageTiffSinglePlaneImagingExtractor(
        file_path=file_path,
        channel_name="Channel 1",
        plane_name="0",
        metadata=metadata,
        parsed_metadata=parsed_metadata,
    )
    assert extractor.metadata == metadata
    assert extractor.parsed_metadata == parsed_metadata


def test_ScanImageTiffSinglePlaneImagingExtractor__init__invalid_metadata_provided(file_path):
    metadata = {"invalid_key": "invalid_value"}
    parsed_metadata = {"invalid_key": "invalid_value"}
    with pytest.raises(KeyError):
        ScanImageTiffSinglePlaneImagingExtractor(
            file_path=file_path,
            channel_name="Channel 1",
            plane_name="0",
            metadata=metadata,
            parsed_metadata=parsed_metadata,
        )


def test_ScanImageTiffSinglePlaneImagingExtractor__init__parsed_metadata_not_provided(file_path):
    metadata = extract_extra_metadata(file_path)
    with pytest.raises(AssertionError):
        ScanImageTiffSinglePlaneImagingExtractor(
            file_path=file_path,
            channel_name="Channel 1",
            plane_name="0",
            metadata=metadata,
        )


@pytest.mark.parametrize("frame_idxs", (0, [0, 1, 2], [0, 2, 5]))
def test_get_frames(scan_image_tiff_single_plane_imaging_extractor, frame_idxs, expected_properties):
    ScanImageTiffReader = _get_scanimage_reader()

    frames = scan_image_tiff_single_plane_imaging_extractor.get_frames(frame_idxs=frame_idxs)
    file_path = str(scan_image_tiff_single_plane_imaging_extractor.file_path)
    plane = scan_image_tiff_single_plane_imaging_extractor.plane
    channel = scan_image_tiff_single_plane_imaging_extractor.channel
    num_planes = expected_properties["num_planes"]
    num_channels = expected_properties["num_channels"]
    frames_per_slice = expected_properties["frames_per_slice"]
    if isinstance(frame_idxs, int):
        frame_idxs = [frame_idxs]

    raw_idxs = []
    for idx in frame_idxs:
        cycle = idx // frames_per_slice
        frame_in_cycle = idx % frames_per_slice
        raw_idx = (
            cycle * num_planes * num_channels * frames_per_slice
            + plane * num_channels * frames_per_slice
            + num_channels * frame_in_cycle
            + channel
        )
        raw_idxs.append(raw_idx)

    with ScanImageTiffReader(file_path) as io:
        assert_array_equal(frames, io.data()[raw_idxs])


@pytest.mark.parametrize("frame_idxs", ([-1], [50]))
def test_get_frames_invalid(scan_image_tiff_single_plane_imaging_extractor, frame_idxs):
    with pytest.raises(ValueError):
        scan_image_tiff_single_plane_imaging_extractor.get_frames(frame_idxs=frame_idxs)


@pytest.mark.parametrize("frame_idx", (1, 3, 5))
def test_get_single_frame(scan_image_tiff_single_plane_imaging_extractor, expected_properties, frame_idx):
    ScanImageTiffReader = _get_scanimage_reader()

    frame = scan_image_tiff_single_plane_imaging_extractor._get_single_frame(frame=frame_idx)
    file_path = str(scan_image_tiff_single_plane_imaging_extractor.file_path)
    plane = scan_image_tiff_single_plane_imaging_extractor.plane
    channel = scan_image_tiff_single_plane_imaging_extractor.channel
    num_planes = expected_properties["num_planes"]
    num_channels = expected_properties["num_channels"]
    frames_per_slice = expected_properties["frames_per_slice"]
    cycle = frame_idx // frames_per_slice
    frame_in_cycle = frame_idx % frames_per_slice
    raw_idx = (
        cycle * num_planes * num_channels * frames_per_slice
        + plane * num_channels * frames_per_slice
        + num_channels * frame_in_cycle
        + channel
    )
    with ScanImageTiffReader(file_path) as io:
        assert_array_equal(frame, io.data()[raw_idx : raw_idx + 1])


@pytest.mark.parametrize("frame", (-1, 50))
def test_get_single_frame_invalid(scan_image_tiff_single_plane_imaging_extractor, frame):
    with pytest.raises(ValueError):
        scan_image_tiff_single_plane_imaging_extractor._get_single_frame(frame=frame)


@pytest.mark.parametrize("start_sample, end_sample", [(0, None), (None, 6), (1, 4), (0, 6)])
def test_get_series(
    scan_image_tiff_single_plane_imaging_extractor,
    expected_properties,
    start_sample,
    end_sample,
):
    ScanImageTiffReader = _get_scanimage_reader()

    series = scan_image_tiff_single_plane_imaging_extractor.get_series(start_sample=start_sample, end_sample=end_sample)
    if start_sample is None:
        start_sample = 0
    if end_sample is None:
        end_sample = expected_properties["num_frames"]
    file_path = str(scan_image_tiff_single_plane_imaging_extractor.file_path)
    plane = scan_image_tiff_single_plane_imaging_extractor.plane
    channel = scan_image_tiff_single_plane_imaging_extractor.channel
    num_planes = expected_properties["num_planes"]
    num_channels = expected_properties["num_channels"]
    frames_per_slice = expected_properties["frames_per_slice"]

    raw_idxs = []
    for idx in range(start_sample, end_sample):
        cycle = idx // frames_per_slice
        frame_in_cycle = idx % frames_per_slice
        raw_idx = (
            cycle * num_planes * num_channels * frames_per_slice
            + plane * num_channels * frames_per_slice
            + num_channels * frame_in_cycle
            + channel
        )
        raw_idxs.append(raw_idx)

    with ScanImageTiffReader(file_path) as io:
        assert_array_equal(series, io.data()[raw_idxs])


@pytest.mark.parametrize("start_sample, end_sample", [(-1, 2), (0, 50)])
def test_get_series_invalid(
    scan_image_tiff_single_plane_imaging_extractor,
    start_sample,
    end_sample,
):
    with pytest.raises(ValueError):
        scan_image_tiff_single_plane_imaging_extractor.get_series(start_sample=start_sample, end_sample=end_sample)


def test_get_image_size(scan_image_tiff_single_plane_imaging_extractor, expected_properties):
    image_size = scan_image_tiff_single_plane_imaging_extractor.get_image_size()
    assert image_size == expected_properties["image_size"]


def test_get_num_frames(scan_image_tiff_single_plane_imaging_extractor, expected_properties):
    num_frames = scan_image_tiff_single_plane_imaging_extractor.get_num_frames()
    assert num_frames == expected_properties["num_frames"]


def test_get_sampling_frequency(scan_image_tiff_single_plane_imaging_extractor, expected_properties):
    sampling_frequency = scan_image_tiff_single_plane_imaging_extractor.get_sampling_frequency()
    assert sampling_frequency == expected_properties["sampling_frequency"]


def test_get_num_channels(scan_image_tiff_single_plane_imaging_extractor, expected_properties):
    num_channels = scan_image_tiff_single_plane_imaging_extractor.get_num_channels()
    assert num_channels == expected_properties["num_channels"]


def test_get_available_planes(scan_image_tiff_single_plane_imaging_extractor, expected_properties):
    file_path = str(scan_image_tiff_single_plane_imaging_extractor.file_path)
    plane_names = ScanImageTiffSinglePlaneImagingExtractor.get_available_planes(file_path)
    assert plane_names == expected_properties["plane_names"]


def test_get_available_channels(scan_image_tiff_single_plane_imaging_extractor, expected_properties):
    file_path = str(scan_image_tiff_single_plane_imaging_extractor.file_path)
    channel_names = ScanImageTiffSinglePlaneImagingExtractor.get_available_channels(file_path)
    assert channel_names == expected_properties["channel_names"]


def test_get_num_planes(scan_image_tiff_single_plane_imaging_extractor, expected_properties):
    num_planes = scan_image_tiff_single_plane_imaging_extractor.get_num_planes()
    assert num_planes == expected_properties["num_planes"]


def test_get_dtype(scan_image_tiff_single_plane_imaging_extractor, expected_properties):
    dtype = scan_image_tiff_single_plane_imaging_extractor.get_dtype()
    assert dtype == expected_properties["dtype"]


def test_check_frame_inputs_valid(scan_image_tiff_single_plane_imaging_extractor):
    scan_image_tiff_single_plane_imaging_extractor.check_frame_inputs(frame=0)


def test_check_frame_inputs_invalid(scan_image_tiff_single_plane_imaging_extractor, expected_properties):
    num_frames = expected_properties["num_frames"]
    with pytest.raises(ValueError):
        scan_image_tiff_single_plane_imaging_extractor.check_frame_inputs(frame=num_frames + 1)


@pytest.mark.parametrize("frame", (0, 3, 5))
def test_frame_to_raw_index(
    scan_image_tiff_single_plane_imaging_extractor,
    frame,
    expected_properties,
):
    raw_index = scan_image_tiff_single_plane_imaging_extractor.frame_to_raw_index(frame=frame)
    plane = scan_image_tiff_single_plane_imaging_extractor.plane
    channel = scan_image_tiff_single_plane_imaging_extractor.channel
    num_planes = expected_properties["num_planes"]
    num_channels = expected_properties["num_channels"]
    frames_per_slice = expected_properties["frames_per_slice"]
    cycle = frame // frames_per_slice
    frame_in_cycle = frame % frames_per_slice
    expected_index = (
        cycle * num_planes * num_channels * frames_per_slice
        + plane * num_channels * frames_per_slice
        + num_channels * frame_in_cycle
        + channel
    )
    assert raw_index == expected_index


def test_ScanImageTiffMultiPlaneImagingExtractor__init__(file_path):
    extractor = ScanImageTiffMultiPlaneImagingExtractor(file_path=file_path)
    assert extractor.file_path == file_path


def test_ScanImageTiffMultiPlaneImagingExtractor__init__invalid(file_path):
    with pytest.raises(ValueError):
        ScanImageTiffMultiPlaneImagingExtractor(file_path=file_path, channel_name="Invalid Channel")


def test_ScanImageTiffMultiPlaneImagingExtractor__init__metadata_provided(file_path):
    metadata = extract_extra_metadata(file_path)
    parsed_metadata = parse_metadata(metadata)
    extractor = ScanImageTiffMultiPlaneImagingExtractor(
        file_path=file_path, metadata=metadata, parsed_metadata=parsed_metadata
    )
    assert extractor.metadata == metadata
    assert extractor.parsed_metadata == parsed_metadata


def test_ScanImageTiffMultiPlaneImagingExtractor__init__invalid_metadata_provided(file_path):
    metadata = {"invalid_key": "invalid_value"}
    parsed_metadata = {"invalid_key": "invalid_value"}
    with pytest.raises(KeyError):
        ScanImageTiffMultiPlaneImagingExtractor(file_path=file_path, metadata=metadata, parsed_metadata=parsed_metadata)


@pytest.fixture(scope="module")
def scanimage_folder_path():
    return OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage"


@pytest.fixture(scope="module")
def multifile_file_pattern():
    return "scanimage_20240320_multifile_*.tif"


@pytest.fixture(scope="module")
def expected_file_names():
    return [
        "scanimage_20240320_multifile_00001.tif",
        "scanimage_20240320_multifile_00002.tif",
        "scanimage_20240320_multifile_00003.tif",
    ]


@pytest.mark.parametrize("extract_all_metadata", [True, False])
def test_ScanImageTiffSinglePlaneMultiFileImagingExtractor__init__(
    scanimage_folder_path, multifile_file_pattern, expected_file_names, extract_all_metadata
):
    extractor = ScanImageTiffSinglePlaneMultiFileImagingExtractor(
        folder_path=scanimage_folder_path,
        file_pattern=multifile_file_pattern,
        channel_name="Channel 1",
        plane_name="0",
        extract_all_metadata=extract_all_metadata,
    )
    file_names = [imaging_extractor.file_path.name for imaging_extractor in extractor._imaging_extractors]
    assert file_names == expected_file_names


def test_ScanImageTiffSinglePlaneMultiFileImagingExtractor__init__invalid(scanimage_folder_path):
    with pytest.raises(ValueError):
        ScanImageTiffSinglePlaneMultiFileImagingExtractor(
            folder_path=scanimage_folder_path,
            file_pattern="invalid_pattern",
            channel_name="Channel 1",
            plane_name="0",
        )


@pytest.mark.parametrize("extract_all_metadata", [True, False])
def test_ScanImageTiffMultiPlaneMultiFileImagingExtractor__init__(
    scanimage_folder_path, multifile_file_pattern, expected_file_names, extract_all_metadata
):
    extractor = ScanImageTiffMultiPlaneMultiFileImagingExtractor(
        folder_path=scanimage_folder_path,
        file_pattern=multifile_file_pattern,
        channel_name="Channel 1",
        extract_all_metadata=extract_all_metadata,
    )
    file_names = [imaging_extractor.file_path.name for imaging_extractor in extractor._imaging_extractors]
    assert file_names == expected_file_names


def test_ScanImageTiffMultiPlaneMultiFileImagingExtractor__init__invalid(scanimage_folder_path):
    with pytest.raises(ValueError):
        ScanImageTiffMultiPlaneMultiFileImagingExtractor(
            folder_path=scanimage_folder_path,
            file_pattern="invalid_pattern",
            channel_name="Channel 1",
        )
