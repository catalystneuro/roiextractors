import pytest
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree, copy
from numpy.testing import assert_array_equal

from ScanImageTiffReader import ScanImageTiffReader
from roiextractors import ScanImageTiffSinglePlaneImagingExtractor, ScanImageTiffMultiPlaneImagingExtractor
from roiextractors.extractors.tiffimagingextractors.scanimagetiffimagingextractor import (
    extract_extra_metadata,
    parse_metadata,
    parse_metadata_v3_8,
)

from .setup_paths import OPHYS_DATA_PATH

scan_image_path = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage"
test_files = [
    "scanimage_20220801_volume.tif",
    "scanimage_20220801_multivolume.tif",
    "scanimage_20230119_adesnik_00001.tif",
]
file_paths = [scan_image_path / test_file for test_file in test_files]


def metadata_string_to_dict(metadata_string):
    metadata_dict = {
        x.split("=")[0].strip(): x.split("=")[1].strip()
        for x in metadata_string.replace("\n", "\r").split("\r")
        if "=" in x
    }
    return metadata_dict


@pytest.fixture(scope="module", params=file_paths)
def scan_image_tiff_single_plane_imaging_extractor(request):
    return ScanImageTiffSinglePlaneImagingExtractor(file_path=request.param, channel_name="Channel 1", plane_name="0")


@pytest.fixture(
    scope="module",
    params=[
        dict(channel_name="Channel 1", plane_name="0"),
        dict(channel_name="Channel 1", plane_name="1"),
        dict(channel_name="Channel 1", plane_name="2"),
        dict(channel_name="Channel 2", plane_name="0"),
        dict(channel_name="Channel 2", plane_name="1"),
        dict(channel_name="Channel 2", plane_name="2"),
    ],
)  # Only the adesnik file has many (>2) frames per plane and multiple (2) channels.
def scan_image_tiff_single_plane_imaging_extractor_adesnik(request):
    file_path = scan_image_path / "scanimage_20230119_adesnik_00001.tif"
    return ScanImageTiffSinglePlaneImagingExtractor(file_path=file_path, **request.param)


@pytest.fixture(scope="module")
def num_planes_adesnik():
    return 3


@pytest.fixture(scope="module")
def num_channels_adesnik():
    return 2


@pytest.mark.parametrize("frame_idxs", (0, [0]))
def test_get_frames(scan_image_tiff_single_plane_imaging_extractor, frame_idxs):
    frames = scan_image_tiff_single_plane_imaging_extractor.get_frames(frame_idxs=frame_idxs)
    file_path = str(scan_image_tiff_single_plane_imaging_extractor.file_path)
    with ScanImageTiffReader(file_path) as io:
        if isinstance(frame_idxs, int):
            frame_idxs = [frame_idxs]
        assert_array_equal(frames, io.data()[frame_idxs])


@pytest.mark.parametrize("frame_idxs", ([0, 1, 2], [1, 3, 31]))  # 31 is the last frame in the adesnik file
def test_get_frames_adesnik(
    scan_image_tiff_single_plane_imaging_extractor_adesnik, num_planes_adesnik, num_channels_adesnik, frame_idxs
):
    frames = scan_image_tiff_single_plane_imaging_extractor_adesnik.get_frames(frame_idxs=frame_idxs)
    file_path = str(scan_image_tiff_single_plane_imaging_extractor_adesnik.file_path)
    plane = scan_image_tiff_single_plane_imaging_extractor_adesnik.plane
    channel = scan_image_tiff_single_plane_imaging_extractor_adesnik.channel
    raw_idxs = [
        idx * num_planes_adesnik * num_channels_adesnik + plane * num_channels_adesnik + channel for idx in frame_idxs
    ]
    with ScanImageTiffReader(file_path) as io:
        assert_array_equal(frames, io.data()[raw_idxs])


def test_get_single_frame(scan_image_tiff_single_plane_imaging_extractor):
    frame = scan_image_tiff_single_plane_imaging_extractor._get_single_frame(frame=0)
    file_path = str(scan_image_tiff_single_plane_imaging_extractor.file_path)
    with ScanImageTiffReader(file_path) as io:
        assert_array_equal(frame, io.data()[:1])


def test_get_video(scan_image_tiff_single_plane_imaging_extractor):
    video = scan_image_tiff_single_plane_imaging_extractor.get_video()
    file_path = str(scan_image_tiff_single_plane_imaging_extractor.file_path)
    num_channels = scan_image_tiff_single_plane_imaging_extractor.get_num_channels()
    num_planes = scan_image_tiff_single_plane_imaging_extractor.get_num_planes()
    with ScanImageTiffReader(file_path) as io:
        assert_array_equal(video, io.data()[:: num_planes * num_channels])


@pytest.mark.parametrize("start_frame, end_frame", [(0, 2), (5, 10), (20, 32)])
def test_get_video_adesnik(
    scan_image_tiff_single_plane_imaging_extractor_adesnik,
    num_planes_adesnik,
    num_channels_adesnik,
    start_frame,
    end_frame,
):
    video = scan_image_tiff_single_plane_imaging_extractor_adesnik.get_video(
        start_frame=start_frame, end_frame=end_frame
    )
    file_path = str(scan_image_tiff_single_plane_imaging_extractor_adesnik.file_path)
    plane = scan_image_tiff_single_plane_imaging_extractor_adesnik.plane
    channel = scan_image_tiff_single_plane_imaging_extractor_adesnik.channel
    raw_idxs = [
        idx * num_planes_adesnik * num_channels_adesnik + plane * num_channels_adesnik + channel
        for idx in range(start_frame, end_frame)
    ]
    with ScanImageTiffReader(file_path) as io:
        assert_array_equal(video, io.data()[raw_idxs])


def test_get_image_size(scan_image_tiff_single_plane_imaging_extractor):
    image_size = scan_image_tiff_single_plane_imaging_extractor.get_image_size()
    file_path = str(scan_image_tiff_single_plane_imaging_extractor.file_path)
    with ScanImageTiffReader(file_path) as io:
        assert image_size == tuple(io.shape()[1:])


def test_get_num_frames(scan_image_tiff_single_plane_imaging_extractor):
    num_frames = scan_image_tiff_single_plane_imaging_extractor.get_num_frames()
    num_channels = scan_image_tiff_single_plane_imaging_extractor.get_num_channels()
    num_planes = scan_image_tiff_single_plane_imaging_extractor.get_num_planes()
    file_path = str(scan_image_tiff_single_plane_imaging_extractor.file_path)
    with ScanImageTiffReader(file_path) as io:
        assert num_frames == io.shape()[0] // (num_channels * num_planes)


def test_get_sampling_frequency(scan_image_tiff_single_plane_imaging_extractor):
    sampling_frequency = scan_image_tiff_single_plane_imaging_extractor.get_sampling_frequency()
    file_path = str(scan_image_tiff_single_plane_imaging_extractor.file_path)
    with ScanImageTiffReader(file_path) as io:
        metadata_string = io.metadata()
        metadata_dict = metadata_string_to_dict(metadata_string)
        assert sampling_frequency == float(metadata_dict["SI.hRoiManager.scanVolumeRate"])


def test_get_num_channels(scan_image_tiff_single_plane_imaging_extractor):
    num_channels = scan_image_tiff_single_plane_imaging_extractor.get_num_channels()
    file_path = str(scan_image_tiff_single_plane_imaging_extractor.file_path)
    with ScanImageTiffReader(file_path) as io:
        metadata_string = io.metadata()
        metadata_dict = metadata_string_to_dict(metadata_string)
        assert num_channels == len(metadata_dict["SI.hChannels.channelsActive"].split(";"))


def test_get_channel_names(scan_image_tiff_single_plane_imaging_extractor):
    channel_names = scan_image_tiff_single_plane_imaging_extractor.get_channel_names()
    num_channels = scan_image_tiff_single_plane_imaging_extractor.get_num_channels()
    file_path = str(scan_image_tiff_single_plane_imaging_extractor.file_path)
    with ScanImageTiffReader(file_path) as io:
        metadata_string = io.metadata()
        metadata_dict = metadata_string_to_dict(metadata_string)
        assert channel_names == metadata_dict["SI.hChannels.channelName"].split("'")[1::2][:num_channels]


def test_get_num_planes(scan_image_tiff_single_plane_imaging_extractor):
    num_planes = scan_image_tiff_single_plane_imaging_extractor.get_num_planes()
    file_path = str(scan_image_tiff_single_plane_imaging_extractor.file_path)
    with ScanImageTiffReader(file_path) as io:
        metadata_string = io.metadata()
        metadata_dict = metadata_string_to_dict(metadata_string)
        assert num_planes == int(metadata_dict["SI.hStackManager.numSlices"])


def test_get_dtype(scan_image_tiff_single_plane_imaging_extractor):
    dtype = scan_image_tiff_single_plane_imaging_extractor.get_dtype()
    file_path = str(scan_image_tiff_single_plane_imaging_extractor.file_path)
    with ScanImageTiffReader(file_path) as io:
        assert dtype == io.data().dtype


def test_check_frame_inputs_valid(scan_image_tiff_single_plane_imaging_extractor):
    scan_image_tiff_single_plane_imaging_extractor.check_frame_inputs(frame=0)


def test_check_frame_inputs_invalid(scan_image_tiff_single_plane_imaging_extractor):
    num_frames = scan_image_tiff_single_plane_imaging_extractor.get_num_frames()
    with pytest.raises(ValueError):
        scan_image_tiff_single_plane_imaging_extractor.check_frame_inputs(frame=num_frames + 1)


@pytest.mark.parametrize("frame", (0, 10, 31))
def test_frame_to_raw_index_adesnik(
    scan_image_tiff_single_plane_imaging_extractor_adesnik, num_channels_adesnik, num_planes_adesnik, frame
):
    raw_index = scan_image_tiff_single_plane_imaging_extractor_adesnik.frame_to_raw_index(frame=frame)
    plane = scan_image_tiff_single_plane_imaging_extractor_adesnik.plane
    channel = scan_image_tiff_single_plane_imaging_extractor_adesnik.channel
    assert raw_index == (frame * num_planes_adesnik * num_channels_adesnik) + (plane * num_channels_adesnik) + channel


@pytest.mark.parametrize("file_path", file_paths)
def test_extract_extra_metadata(file_path):
    metadata = extract_extra_metadata(file_path)
    io = ScanImageTiffReader(str(file_path))
    extra_metadata = {}
    for metadata_string in (io.description(iframe=0), io.metadata()):
        metadata_dict = {
            x.split("=")[0].strip(): x.split("=")[1].strip()
            for x in metadata_string.replace("\n", "\r").split("\r")
            if "=" in x
        }
        extra_metadata = dict(**extra_metadata, **metadata_dict)
    assert metadata == extra_metadata


@pytest.mark.parametrize("file_path", file_paths)
def test_parse_metadata(file_path):
    metadata = extract_extra_metadata(file_path)
    parsed_metadata = parse_metadata(metadata)
    sampling_frequency = float(metadata["SI.hRoiManager.scanVolumeRate"])
    num_channels = len(metadata["SI.hChannels.channelsActive"].split(";"))
    num_planes = int(metadata["SI.hStackManager.numSlices"])
    frames_per_slice = int(metadata["SI.hStackManager.framesPerSlice"])
    channel_names = metadata["SI.hChannels.channelName"].split("'")[1::2][:num_channels]
    assert parsed_metadata == dict(
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        num_planes=num_planes,
        frames_per_slice=frames_per_slice,
        channel_names=channel_names,
    )


def test_parse_metadata_v3_8():
    file_path = scan_image_path / "sample_scanimage_version_3_8.tiff"
    metadata = extract_extra_metadata(file_path)
    parsed_metadata = parse_metadata_v3_8(metadata)
    sampling_frequency = float(metadata["state.acq.frameRate"])
    num_channels = int(metadata["state.acq.numberOfChannelsSave"])
    num_planes = int(metadata["state.acq.numberOfZSlices"])
    assert parsed_metadata == dict(
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        num_planes=num_planes,
    )


@pytest.mark.parametrize("file_path", file_paths)
def test_ScanImageTiffMultiPlaneImagingExtractor__init__(file_path):
    extractor = ScanImageTiffMultiPlaneImagingExtractor(file_path=file_path)
    assert extractor.file_path == file_path
