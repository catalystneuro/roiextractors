import pytest
from numpy import dtype
from numpy.testing import assert_array_equal

from roiextractors import CxdImagingExtractor
from tests.setup_paths import OPHYS_DATA_PATH

# Skip all tests in this module if aicsimageio is not installed
aicsimageio = pytest.importorskip("aicsimageio")

try:
    from aicsimageio.readers.bioformats_reader import BioFile
except ImportError:
    pytest.skip(
        "aicsimageio.readers.bioformats_reader.BioFile is required for these tests but not available.",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def file_path():
    return OPHYS_DATA_PATH / "imaging_datasets" / "BioFormats" / "Data00676_stubbed.cxd"


@pytest.fixture(scope="module")
def expected_properties():
    return dict(
        num_frames=10,
        num_channels=1,
        num_planes=1,
        channel_names=["Channel:0:0"],
        plane_names=["0"],
        image_size=(350, 350),
        sampling_frequency=1.0,
        dtype="uint16",
    )


@pytest.fixture(scope="module")
def cxdimagingextractor(file_path, expected_properties):
    return CxdImagingExtractor(file_path=file_path, sampling_frequency=expected_properties["sampling_frequency"])


@pytest.mark.parametrize(
    "channel_name, plane_name, expected_error_message",
    [
        ("Invalid Channel", "0", "The selected channel 'Invalid Channel' is not a valid channel name."),
        ("Channel:0:0", "Invalid Plane", "The selected plane 'Invalid Plane' is not a valid plane name."),
    ],
)
def test_cxdimagingextractor__init__invalid(file_path, channel_name, plane_name, expected_error_message):
    with pytest.raises(ValueError, match=expected_error_message):
        CxdImagingExtractor(
            file_path=file_path,
            channel_name=channel_name,
            plane_name=plane_name,
            sampling_frequency=1.0,
        )


def test_cxdimagingextractor__init__sampling_frequency_not_provided_when_missing_from_metadata(file_path):
    expected_error_message = "Sampling frequency is not found in the metadata. Please provide it manually with the 'sampling_frequency' argument."
    with pytest.raises(ValueError, match=expected_error_message):
        CxdImagingExtractor(file_path=file_path)


def test_get_num_frames(cxdimagingextractor, expected_properties):
    num_frames = cxdimagingextractor.get_num_frames()
    assert num_frames == expected_properties["num_frames"]


def test_get_image_size(cxdimagingextractor, expected_properties):
    image_size = cxdimagingextractor.get_image_size()
    assert image_size == expected_properties["image_size"]


def test_get_dtype(cxdimagingextractor, expected_properties):
    frames_dtype = cxdimagingextractor.get_dtype()
    assert frames_dtype == dtype(expected_properties["dtype"])


def test_get_sampling_frequency(cxdimagingextractor, expected_properties):
    sampling_frequency = cxdimagingextractor.get_sampling_frequency()
    assert sampling_frequency == expected_properties["sampling_frequency"]


def test_get_channel_names(cxdimagingextractor, expected_properties):
    channel_names = cxdimagingextractor.get_channel_names()
    assert channel_names == expected_properties["channel_names"]


def test_get_num_channels(cxdimagingextractor, expected_properties):
    num_channels = cxdimagingextractor.get_num_channels()
    assert num_channels == expected_properties["num_channels"]


@pytest.mark.parametrize(
    "start_frame, end_frame, num_frames",
    [(0, None, 10), (None, 6, 6), (1, 4, 3), (0, 6, 6)],
)
def test_get_video(file_path, cxdimagingextractor, expected_properties, start_frame, end_frame, num_frames):
    video = cxdimagingextractor.get_video(start_frame=start_frame, end_frame=end_frame)
    assert video.shape == (num_frames, *expected_properties["image_size"])

    with aicsimageio.readers.bioformats_reader.BioFile(file_path) as reader:
        test_video = reader.to_numpy()

    assert_array_equal(video, test_video[start_frame:end_frame, 0, 0, ...])


@pytest.mark.parametrize("start_frame, end_frame", [(-1, 2), (0, 50)])
def test_get_video_invalid(cxdimagingextractor, start_frame, end_frame):
    with pytest.raises(ValueError):
        cxdimagingextractor.get_video(start_frame=start_frame, end_frame=end_frame)
