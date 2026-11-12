import shutil
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tifffile import TiffFile

from roiextractors import MicroManagerTiffImagingExtractor
from tests.extractor_mixins import ImagingExtractorTestMixin, check_metadata
from tests.setup_paths import OPHYS_DATA_PATH

MICROMANAGER_PATH = OPHYS_DATA_PATH / "imaging_datasets" / "MicroManagerTif"
MMSTACK_PATH = MICROMANAGER_PATH / "mmstack"
BURST_PATH = (
    MMSTACK_PATH / "micromanager_version_2_0_0" / "time_structure" / "burst_with_metadata_txt_and_display_settings"
)
OUT_OF_ORDER_PATH = MMSTACK_PATH / "micromanager_version_2_0_0_beta3" / "time_structure" / "out_of_order_pages"
STOPPED_PATH = MMSTACK_PATH / "micromanager_version_1_4_15" / "time_structure" / "stopped_index_map_after_images"


def _read_frames(folder_path):
    frames = {}
    for file_path in sorted(folder_path.glob("*.ome.tif")):
        with TiffFile(file_path) as tiff:
            channel_names = tiff.micromanager_metadata["Summary"]["ChNames"]
            for page in tiff.pages:
                metadata = page.tags[51123].value
                coordinates = tuple(
                    int(metadata[key]) for key in ("PositionIndex", "ChannelIndex", "FrameIndex", "SliceIndex")
                )
                frames[coordinates] = page.asarray(), float(metadata["ElapsedTime-ms"]) / 1000
    return frames, channel_names


def _read_series(frames, position_index, channel_index):
    selected = {key: value for key, value in frames.items() if key[:2] == (position_index, channel_index)}
    time_indices = sorted({key[2] for key in selected})
    depth_indices = sorted({key[3] for key in selected})
    series = np.stack(
        [
            np.stack(
                [selected[position_index, channel_index, time_index, depth][0] for depth in depth_indices], axis=-1
            )
            for time_index in time_indices
        ]
    )
    timestamps = np.array(
        [
            [selected[position_index, channel_index, time_index, depth][1] for depth in depth_indices]
            for time_index in time_indices
        ]
    )
    if len(depth_indices) == 1:
        series = series[..., 0]
    return series, timestamps


class _MicroManagerMixin(ImagingExtractorTestMixin):
    extractor_cls = MicroManagerTiffImagingExtractor
    expected_sample_shape = (64, 64)
    expected_dtype = np.dtype("uint16")

    def check_series(self, series):
        frames, names = _read_frames(self.extractor_kwargs["file_path"].parent)
        channel_index = names.index(self.extractor_kwargs.get("channel_name", names[0]))
        expected, _ = _read_series(frames, position_index=0, channel_index=channel_index)
        assert_array_equal(series, expected)


class TestMicroManagerBurst(_MicroManagerMixin):
    extractor_kwargs = dict(file_path=BURST_PATH / "beads_1_MMStack.ome.tif")
    expected_num_samples = 20
    expected_channel_names = ["Default"]
    # Plane/@DeltaT in milliseconds, also stored as ElapsedTime-ms in each page.
    expected_native_timestamps = [1.527, 1.62802, 1.72801]
    expected_sampling_frequency = 1 / 0.10167
    expected_metadata = dict(
        acquisition_start_datetime=datetime(2024, 8, 5, 15, 37, 23, 653000, tzinfo=timezone(timedelta(hours=2))),
        channel_name="Default",
        exposure_time=0.1,
        acquisition_software_name="Micro-Manager",
        acquisition_software_version="2.0.0",
    )


class TestMicroManagerStoppedAcquisition(_MicroManagerMixin):
    extractor_kwargs = dict(
        file_path=STOPPED_PATH / "170703_100x15x15_100pct_laser_10nM_2h_incubation_04_MMStack.ome.tif"
    )
    expected_num_samples = 5
    expected_channel_names = ["Default"]
    # Plane/@DeltaT in seconds. SizeT still says 500, while TiffData/PlaneCount is 5.
    expected_native_timestamps = [0.361, 5.368, 10.344]
    expected_sampling_frequency = 1 / 5.007
    expected_metadata = dict(
        acquisition_start_datetime=datetime(2017, 7, 3, 15, 54, 23),
        channel_name="Default",
        exposure_time=0.2,
        acquisition_software_name="Micro-Manager",
        acquisition_software_version="1.4.15",
    )


class TestMicroManagerPagesOutOfOrder(_MicroManagerMixin):
    extractor_kwargs = dict(file_path=OUT_OF_ORDER_PATH / "MMStack_Pos0.ome.tif", channel_name="FITC")
    expected_num_samples = 2
    expected_channel_names = ["FITC", "DAPI"]
    # FITC Plane/@DeltaT in milliseconds. Its pages are 1 and 2, rather than 0 and 2.
    expected_native_timestamps = [0.164, 0.385]
    expected_sampling_frequency = 1 / 0.221
    expected_metadata = dict(
        channel_name="FITC",
        pixel_height_um=1.0,
        pixel_width_um=1.0,
        field_of_view_height_um=64.0,
        field_of_view_width_um=64.0,
        exposure_time=0.01,
        acquisition_software_name="Micro-Manager",
        acquisition_software_version="2.0.0-beta3-20160124",
    )


@pytest.mark.parametrize(
    "relative_folder",
    [
        "micromanager_version_2_0_0/time_structure/burst_with_metadata_txt_and_display_settings",
        "micromanager_version_2_0_0_beta3/time_structure/out_of_order_pages",
        "micromanager_version_2_0_1/time_structure/set_interval_single_channel_with_metadata_txt",
        "micromanager_version_1_4_22/time_structure/interval_shorter_than_exposure",
        "micromanager_version_1_4_22/time_structure/set_interval_two_channels_with_metadata_txt",
        "micromanager_version_1_4_22/time_structure/stopped",
        "micromanager_version_1_4_15/time_structure/stopped_index_map_after_images",
        "micromanager_version_1_4_22/acquisition_setup/grid_position_names",
        "micromanager_version_2_0_0_beta3/acquisition_setup/positions_in_one_file",
        "micromanager_version_2_0_1/acquisition_setup/plate_well_names",
        "micromanager_version_1_4_23/binary_data_layout/split_files",
        "micromanager_version_2_0_0/binary_data_layout/split_files",
        "micromanager_version_2_0_0_beta3/binary_data_layout/split_files",
    ],
)
def test_mmstack_pixels_and_times(relative_folder):
    folder_path = MMSTACK_PATH / relative_folder
    frames, channel_names = _read_frames(folder_path)
    for position_index in sorted({key[0] for key in frames}):
        for channel_index, channel_name in enumerate(channel_names):
            expected_series, expected_times = _read_series(frames, position_index, channel_index)
            for file_path in sorted(folder_path.glob("*.ome.tif")):
                kwargs = dict(file_path=file_path, channel_name=channel_name, position_index=position_index)
                if len(expected_series) == 1:
                    kwargs["sampling_frequency"] = 1.0
                extractor = MicroManagerTiffImagingExtractor(**kwargs)
                assert "_unified_metadata" not in extractor.__dict__
                assert_array_equal(extractor.get_series(), expected_series)
                assert_array_equal(extractor.get_series(1, 3), expected_series[1:3])
                assert_allclose(extractor.get_native_timestamps(), expected_times[:, 0])
                assert_allclose(extractor.get_native_timestamps(1, 3), expected_times[1:3, 0])
                if len(expected_times) > 1:
                    expected_rate = 1 / np.median(np.diff(expected_times[:, 0]))
                    assert extractor.get_sampling_frequency() == pytest.approx(expected_rate)
                if extractor.is_volumetric:
                    for plane_index in range(expected_series.shape[-1]):
                        plane = extractor.select_plane(plane_index)
                        assert "_unified_metadata" not in plane.__dict__
                        assert_array_equal(plane.get_series(), expected_series[..., plane_index])
                        assert_allclose(plane.get_native_timestamps(), expected_times[:, plane_index])


def test_position_selection():
    folder = MMSTACK_PATH / "micromanager_version_2_0_0_beta3" / "acquisition_setup" / "positions_in_one_file"
    file_path = folder / "181003_multi_pos_time_course_1_MMStack.ome.tif"
    with pytest.raises(ValueError, match="position_index must be specified"):
        MicroManagerTiffImagingExtractor(file_path=file_path)
    with pytest.raises(ValueError, match="position_index must be between 0 and 1"):
        MicroManagerTiffImagingExtractor(file_path=file_path, position_index=2)

    folder = MMSTACK_PATH / "micromanager_version_1_4_22" / "acquisition_setup" / "grid_position_names"
    file_path = folder / "mm1422_4p_2t_3c_512_1_MMStack_2-Pos_001_000.ome.tif"
    extractor = MicroManagerTiffImagingExtractor(file_path=file_path, channel_name="DAPI")
    frames, _ = _read_frames(folder)
    expected, times = _read_series(frames, position_index=1, channel_index=1)
    assert_array_equal(extractor.get_series(), expected)
    assert_allclose(extractor.get_native_timestamps(), times[:, 0])


def test_folder_path_deprecation():
    with pytest.warns(FutureWarning, match="Passing a folder is deprecated"):
        extractor = MicroManagerTiffImagingExtractor(folder_path=BURST_PATH)
    assert extractor.get_num_samples() == 20
    with pytest.warns(FutureWarning, match="Passing a folder is deprecated"):
        positional_extractor = MicroManagerTiffImagingExtractor(BURST_PATH)
    assert_array_equal(positional_extractor.get_series(), extractor.get_series())


def test_missing_files(tmp_path):
    folder = MMSTACK_PATH / "micromanager_version_1_4_23" / "binary_data_layout" / "split_files"
    file_path = folder / "MMStack_Pos0.ome.tif"
    shutil.copy(file_path, tmp_path / file_path.name)
    with pytest.raises(AssertionError, match="The list of files that are missing: .*MMStack_Pos0_1.ome.tif"):
        MicroManagerTiffImagingExtractor(file_path=tmp_path / file_path.name)


def test_damaged_legacy_fixture():
    file_path = (
        MICROMANAGER_PATH / "TS12_20220407_20hz_noteasy_1" / "TS12_20220407_20hz_noteasy_1_MMStack_Default.ome.tif"
    )
    with pytest.raises(ValueError, match="repeated TIFF page references"):
        MicroManagerTiffImagingExtractor(file_path=file_path)


def test_sampling_frequency_override():
    file_path = BURST_PATH / "beads_1_MMStack.ome.tif"
    extractor = MicroManagerTiffImagingExtractor(file_path=file_path, sampling_frequency=30.0)
    assert extractor.get_sampling_frequency() == 30.0
    assert_allclose(extractor.get_native_timestamps()[:3], [1.527, 1.62802, 1.72801])


def test_single_sample_requires_sampling_frequency():
    folder = MMSTACK_PATH / "micromanager_version_2_0_1" / "acquisition_setup" / "plate_well_names"
    with pytest.raises(ValueError, match="sampling_frequency must be provided"):
        MicroManagerTiffImagingExtractor(file_path=folder / "phase_1_MMStack_A1-Site_0.ome.tif", channel_name="State0")


@pytest.mark.filterwarnings("ignore:The metadata API is experimental:UserWarning")
def test_volumetric_metadata_and_views():
    folder = MMSTACK_PATH / "micromanager_version_2_0_1" / "acquisition_setup" / "plate_well_names"
    extractor = MicroManagerTiffImagingExtractor(
        file_path=folder / "phase_1_MMStack_A1-Site_2.ome.tif", channel_name="State1", sampling_frequency=1.0
    )
    # Summary/StartTime, Pixels/PhysicalSizeX/Y, and Plane/PositionZ and ExposureTime.
    expected = dict(
        acquisition_start_datetime=datetime(2023, 8, 24, 11, 5, 21, 678000, tzinfo=timezone(timedelta(hours=-7))),
        channel_name="State1",
        pixel_height_um=0.325,
        pixel_width_um=0.325,
        field_of_view_height_um=20.8,
        field_of_view_width_um=20.8,
        plane_z_offsets_um=[0.0, 2.0],
        exposure_time=0.01,
        acquisition_software_name="Micro-Manager",
        acquisition_software_version="2.0.1 20230426",
    )
    plane = extractor.select_plane(1)
    assert "_unified_metadata" not in extractor.__dict__
    assert "_unified_metadata" not in plane.__dict__
    check_metadata(plane.metadata, {**expected, "plane_z_offsets_um": [2.0]})
    check_metadata(extractor.metadata, expected)
    assert extractor.metadata is extractor.metadata
    cached_plane = extractor.select_plane(1)
    assert "_unified_metadata" not in cached_plane.__dict__
    check_metadata(cached_plane.metadata, {**expected, "plane_z_offsets_um": [2.0]})
