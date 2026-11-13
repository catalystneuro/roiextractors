import re
import shutil
from datetime import datetime

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tifffile import TiffReader

from roiextractors import ScanImageMultiROIImagingExtractor

from .extractor_mixins import ImagingExtractorTestMixin
from .setup_paths import OPHYS_DATA_PATH

SCANIMAGE_PATH = OPHYS_DATA_PATH / "imaging_datasets" / "ScanImage"
ACQUISITION_SETUP_PATH = SCANIMAGE_PATH / "tiff_format_version_4" / "acquisition_setup"


def read_pages(*file_paths):
    """Read every page of the files with `tifffile`, in the order of the files, each page holding all the ROIs."""
    pages = []
    for file_path in file_paths:
        with TiffReader(file_path) as tiff_reader:
            pages.extend(page.asarray() for page in tiff_reader.pages)
    return np.stack(pages)


class TestScanImageMultiROIImagingExtractorVersion2023(ImagingExtractorTestMixin):
    """The seventh of eight ROIs of 64 x 64, separated by gaps of 4 rows, so it starts at row 6 * (64 + 4) = 408."""

    FILE_PATH = ACQUISITION_SETUP_PATH / "multi_roi_planar_eight_rois_version_2023_1" / "planar_00000_00001.tif"
    extractor_cls = ScanImageMultiROIImagingExtractor
    extractor_kwargs = dict(file_path=FILE_PATH, roi_index=6)
    expected_num_samples = 3
    expected_sample_shape = (64, 64)
    expected_dtype = np.dtype("int16")
    expected_channel_names = ["Channel 2"]
    # frameTimestamps_sec of pages 0 to 2, plus linePeriod for each of the 408 rows above the ROI
    expected_native_timestamps = (np.array([0.0, 0.20519988, 0.41039981]) + 408 * 4.154514711136592e-05).tolist()
    expected_sampling_frequency = 4.884374999999999  # scanFrameRate, with logAverageFactor 1
    expected_metadata = dict(
        acquisition_start_datetime=datetime(2025, 4, 8, 11, 20, 56, 264000),  # epoch = [2025  4  8 11 20 56.264]
        # sizeXY of the ROI's scanfield in degrees, times objectiveResolution, over its pixelResolutionXY
        pixel_height_um=0.5833125 * 150 / 64,
        pixel_width_um=0.5833125 * 150 / 64,
        field_of_view_height_um=0.5833125 * 150,
        field_of_view_width_um=0.5833125 * 150,
        channel_name="Channel 2",
        scan_line_rate=1 / 4.154514711136592e-05,  # 1 / linePeriod
        acquisition_software_name="ScanImage",
        acquisition_software_version="2023.1.0",  # VERSION_MAJOR, VERSION_MINOR and VERSION_UPDATE
    )

    def check_series(self, series):
        """Each sample is rows 408 to 472 of its page, the seventh ROI.

        The README of the fixture gives eight ROIs of 64 x 64 stacked on a page of 540 rows with gaps of 4 rows, so the
        six ROIs above `roi_index` 6 and their gaps take 6 * (64 + 4) = 408 rows.
        """
        pages = read_pages(self.FILE_PATH)
        assert_array_equal(series, pages[:, 408:472, :])

    def test_get_available_rois(self):
        """The ROIs are listed in their order on the page, with the names of the file, out of order and repeated."""
        rois = ScanImageMultiROIImagingExtractor.get_available_rois(self.FILE_PATH)
        # `name`, `roiUuid` and the scanfield `centerXY` in degrees of each ROI in `RoiGroups.imagingRoiGroup.rois`
        names_uuids_and_centers = [
            ("ROI 5", "C8E4ACA7528D832A", (-8.942879464, -7.556308036)),
            ("ROI 7", "AEF0E4BD9382753E", (-11.82978423, -2.889808036)),
            ("ROI 6", "6A895E8EA9797606", (-7.163284226, -2.889808036)),
            ("ROI 4", "85C2409BB89682E6", (-4.276379464, -9.282498512)),
            ("ROI 2", "87C4A3828680E72C", (0.390120536, -10.56226042)),
            ("ROI 3", "AEF4579E650B8C92", (5.056620536, -10.56226042)),
            ("ROI 3", "F30594C7CA58209F", (5.413763393, -5.895760417)),
            ("ROI 1", "88B86D39518ACE11", (4.491144345, 4.610191964)),
        ]
        micrometers_per_degree = 150  # SI.objectiveResolution
        size_xy_degrees = (0.5833125, 0.5833125)  # the scanfield sizeXY of every ROI, 64 pixels of 1.37 um
        assert [roi["roi_index"] for roi in rois] == list(range(8))
        assert [(roi["name"], roi["uuid"]) for roi in rois] == [
            (name, uuid) for name, uuid, _ in names_uuids_and_centers
        ]
        for roi, (_, _, center_xy_degrees) in zip(rois, names_uuids_and_centers):
            assert_allclose(roi["center_xy_um"], np.multiply(center_xy_degrees, micrometers_per_degree))
            assert_allclose(roi["size_xy_um"], np.multiply(size_xy_degrees, micrometers_per_degree))

    def test_cropped_roi_has_its_own_timestamps(self, setup_extractor):
        """A crop starting at row 10 of the ROI is scanned 10 rows later, also when the ROI timestamps were read before."""
        roi_timestamps = self.extractor.get_timestamps()
        cropped_extractor = self.extractor.slice_field_of_view(row_start=10, row_end=20)
        line_period = 4.154514711136592e-05
        expected_timestamps = roi_timestamps + 10 * line_period
        assert_allclose(cropped_extractor.get_native_timestamps(), expected_timestamps)
        assert_allclose(cropped_extractor.get_timestamps(), expected_timestamps)


class TestScanImageMultiROIImagingExtractorVersion2020(ImagingExtractorTestMixin):
    """The last of eight ROIs of 64 x 64, separated by gaps of 4 rows, so it ends at the bottom of the page."""

    FILE_PATH = ACQUISITION_SETUP_PATH / "multi_roi_planar_eight_rois_version_2020_1" / "planar_00000_00001.tif"
    extractor_cls = ScanImageMultiROIImagingExtractor
    extractor_kwargs = dict(file_path=FILE_PATH, roi_index=7)
    expected_num_samples = 3
    expected_sample_shape = (64, 64)
    expected_dtype = np.dtype("int16")
    expected_channel_names = ["Channel 2"]
    # frameTimestamps_sec of pages 0 to 2, plus linePeriod for each of the 7 * (64 + 4) = 476 rows above the ROI
    expected_native_timestamps = (np.array([0.0, 0.19726124, 0.394522505]) + 476 * 4.16601e-05).tolist()
    expected_sampling_frequency = 5.06837  # scanFrameRate, with logAverageFactor 1
    expected_metadata = dict(
        acquisition_start_datetime=datetime(2024, 6, 14, 18, 18, 53, 244000),  # epoch = [2024  6 14 18 18 53.244]
        # sizeXY of the ROI's scanfield in degrees, times objectiveResolution, over its pixelResolutionXY
        pixel_height_um=0.5833125 * 150 / 64,
        pixel_width_um=0.5833125 * 150 / 64,
        field_of_view_height_um=0.5833125 * 150,
        field_of_view_width_um=0.5833125 * 150,
        channel_name="Channel 2",
        scan_line_rate=1 / 4.16601e-05,  # 1 / linePeriod
        acquisition_software_name="ScanImage",
        acquisition_software_version="2020.1.2",  # VERSION_MAJOR, VERSION_MINOR and VERSION_UPDATE
    )

    def check_series(self, series):
        """Each sample is rows 476 to 540 of its page, the last ROI.

        The README of the fixture gives eight ROIs of 64 x 64 stacked on a page of 540 rows with gaps of 4 rows, so the
        seven ROIs above `roi_index` 7 and their gaps take 7 * (64 + 4) = 476 rows, and the ROI ends at the bottom.
        """
        pages = read_pages(self.FILE_PATH)
        assert_array_equal(series, pages[:, 476:540, :])


class TestScanImageMultiROIImagingExtractorDifferentHeightsVersion2018(ImagingExtractorTestMixin):
    """The last of three ROIs of 502, 502 and 420 rows, which starts below the other two and their gaps of 50 rows."""

    FILE_PATH = (
        SCANIMAGE_PATH
        / "tiff_format_version_3"
        / "acquisition_setup"
        / "multi_roi_planar_three_rois_of_different_heights_version_2018b"
        / "planar_00001.tif"
    )
    extractor_cls = ScanImageMultiROIImagingExtractor
    extractor_kwargs = dict(file_path=FILE_PATH, roi_index=2)
    expected_num_samples = 3
    expected_sample_shape = (420, 120)
    expected_dtype = np.dtype("int16")
    expected_channel_names = ["Channel 1"]
    # frameTimestamps_sec of pages 0 to 2, plus linePeriod for each of the 502 + 50 + 502 + 50 = 1104 rows above the ROI
    expected_native_timestamps = (np.array([0.0, 0.065530685, 0.1310615]) + 1104 * 4.16349e-05).tolist()
    expected_sampling_frequency = 15.2594  # scanFrameRate, with logAverageFactor 1
    expected_metadata = dict(
        acquisition_start_datetime=datetime(2021, 3, 22, 13, 28, 7, 573000),  # epoch = [2021  3 22 13 28 7.573]
        # sizeXY of the ROI's scanfield in degrees, times objectiveResolution, over its pixelResolutionXY
        pixel_height_um=14 * 150 / 420,
        pixel_width_um=4 * 150 / 120,
        field_of_view_height_um=14 * 150,
        field_of_view_width_um=4 * 150,
        channel_name="Channel 1",
        scan_line_rate=1 / 4.16349e-05,  # 1 / linePeriod
        acquisition_software_name="ScanImage",
        acquisition_software_version="2018b.1",  # VERSION_MAJOR and VERSION_MINOR, 2018b has no VERSION_UPDATE
    )

    def check_series(self, series):
        """Each sample is rows 1104 to 1524 of its page, the last ROI.

        The README of the fixture gives three ROIs of 502, 502 and 420 rows on a page of 1524 rows, which leaves
        1524 - 1424 = 100 rows, a gap of 50 rows at each of the two boundaries. The ROI starts below the first two ROIs
        and their gaps, at 502 + 50 + 502 + 50 = 1104.
        """
        pages = read_pages(self.FILE_PATH)
        assert_array_equal(series, pages[:, 1104:1524, :])


class TestScanImageMultiROIImagingExtractorDifferentHeightsTwoChannelsVersion2018(ImagingExtractorTestMixin):
    """The second channel of the middle of three ROIs of 540, 520 and 480 rows, separated by gaps of 50 rows."""

    FILE_PATH = (
        SCANIMAGE_PATH
        / "tiff_format_version_3"
        / "acquisition_setup"
        / "multi_roi_planar_two_channels_three_rois_of_different_heights_version_2018b"
        / "planar_00001.tif"
    )
    extractor_cls = ScanImageMultiROIImagingExtractor
    extractor_kwargs = dict(file_path=FILE_PATH, channel_name="Channel 2", roi_index=1)
    expected_num_samples = 2
    expected_sample_shape = (520, 120)
    expected_dtype = np.dtype("int16")
    expected_channel_names = ["Channel 1", "Channel 2"]
    # frameTimestamps_sec of pages 1 and 3, plus linePeriod for each of the 540 + 50 = 590 rows above the ROI
    expected_native_timestamps = (np.array([0.0, 0.0703365]) + 590 * 4.16504e-05).tolist()
    expected_sampling_frequency = 14.2067  # scanFrameRate, with logAverageFactor 1
    expected_metadata = dict(
        acquisition_start_datetime=datetime(2020, 7, 31, 14, 38, 0, 292000),  # epoch = [2020  7 31 14 38 0.292]
        # sizeXY of the ROI's scanfield in degrees, times objectiveResolution, over its pixelResolutionXY
        pixel_height_um=17.33333333 * 150 / 520,
        pixel_width_um=4 * 150 / 120,
        field_of_view_height_um=17.33333333 * 150,
        field_of_view_width_um=4 * 150,
        channel_name="Channel 2",
        scan_line_rate=1 / 4.16504e-05,  # 1 / linePeriod
        acquisition_software_name="ScanImage",
        acquisition_software_version="2018b.1",  # VERSION_MAJOR and VERSION_MINOR, 2018b has no VERSION_UPDATE
    )

    def check_series(self, series):
        """Each sample is rows 590 to 1110 of page `2 * sample + 1`, the middle ROI of the second channel.

        The two channels are saved in turn, so the second channel is in the odd pages. The README of the fixture gives
        three ROIs of 540, 520 and 480 rows on a page of 1640 rows, which leaves 1640 - 1540 = 100 rows, a gap of 50 rows
        at each of the two boundaries. The ROI starts below the first ROI and its gap, at 540 + 50 = 590.
        """
        pages = read_pages(self.FILE_PATH)
        assert_array_equal(series, pages[1::2, 590:1110, :])


class TestScanImageMultiROIImagingExtractorSplitOverFilesVersion2018(ImagingExtractorTestMixin):
    """The middle of three ROIs of 64 x 64, separated by gaps of 4 rows, in an acquisition saved as 5 files."""

    FILE_PATH = (
        SCANIMAGE_PATH
        / "tiff_format_version_3"
        / "binary_data_layout"
        / "multi_roi_planar_three_rois_split_over_files_version_2018b"
        / "planar_00002_00001.tif"
    )
    extractor_cls = ScanImageMultiROIImagingExtractor
    extractor_kwargs = dict(file_path=FILE_PATH, roi_index=1)
    expected_num_samples = 9
    expected_sample_shape = (64, 64)
    expected_dtype = np.dtype("int16")
    expected_channel_names = ["Channel 1"]
    # frameTimestamps_sec of pages 0 and 1 of the first file and page 0 of the second, plus linePeriod for each of the
    # 64 + 4 = 68 rows above the ROI
    expected_native_timestamps = (np.array([0.0, 0.056171685, 0.1123435]) + 68 * 4.16189e-05).tolist()
    expected_sampling_frequency = 17.7982  # scanFrameRate, with logAverageFactor 1
    expected_metadata = dict(
        acquisition_start_datetime=datetime(2022, 7, 21, 12, 11, 19, 901000),  # epoch = [2022  7 21 12 11 19.901]
        # sizeXY of the ROI's scanfield in degrees, times objectiveResolution, over its pixelResolutionXY
        pixel_height_um=1.6 * 150 / 64,
        pixel_width_um=2.133333333 * 150 / 64,
        field_of_view_height_um=1.6 * 150,
        field_of_view_width_um=2.133333333 * 150,
        channel_name="Channel 1",
        scan_line_rate=1 / 4.16189e-05,  # 1 / linePeriod
        acquisition_software_name="ScanImage",
        acquisition_software_version="2018b.1",  # VERSION_MAJOR and VERSION_MINOR, 2018b has no VERSION_UPDATE
    )

    def check_series(self, series):
        """Each sample is rows 68 to 132 of its page, the middle ROI, with the pages of the 5 files in order.

        The README of the fixture gives three ROIs of 64 x 64 on a page of 200 rows with gaps of 4 rows, so the ROI
        starts below the first ROI and its gap, at 64 + 4 = 68.
        """
        file_paths = sorted(self.FILE_PATH.parent.glob("planar_00002_*.tif"))
        pages = read_pages(*file_paths)
        assert_array_equal(series, pages[:, 68:132, :])


class TestScanImageMultiROIImagingExtractorSplitOverFilesVersion2020(ImagingExtractorTestMixin):
    """The first of three ROIs of 64 x 64, at the top of the page, in an acquisition saved as 3 files."""

    FILE_PATH = (
        SCANIMAGE_PATH
        / "tiff_format_version_4"
        / "binary_data_layout"
        / "multi_roi_planar_three_rois_split_over_files_no_frame_count_version_2020_1"
        / "planar_00001_00001.tif"
    )
    extractor_cls = ScanImageMultiROIImagingExtractor
    extractor_kwargs = dict(file_path=FILE_PATH, roi_index=0)
    expected_num_samples = 5
    expected_sample_shape = (64, 64)
    expected_dtype = np.dtype("int16")
    expected_channel_names = ["Channel 2"]
    # frameTimestamps_sec of pages 0 and 1 of the first file and page 0 of the second. No rows are above the ROI.
    expected_native_timestamps = [0.0, 0.26712276, 0.53424556]
    expected_sampling_frequency = 3.74462  # scanFrameRate, with logAverageFactor 1
    expected_metadata = dict(
        acquisition_start_datetime=datetime(2024, 7, 23, 18, 31, 10, 786000),  # epoch = [2024  7 23 18 31 10.786]
        # sizeXY of the ROI's scanfield in degrees, times objectiveResolution, over its pixelResolutionXY
        pixel_height_um=0.5833125 * 150 / 64,
        pixel_width_um=0.5833125 * 150 / 64,
        field_of_view_height_um=0.5833125 * 150,
        field_of_view_width_um=0.5833125 * 150,
        channel_name="Channel 2",
        scan_line_rate=1 / 4.16354e-05,  # 1 / linePeriod
        acquisition_software_name="ScanImage",
        acquisition_software_version="2020.1.2",  # VERSION_MAJOR, VERSION_MINOR and VERSION_UPDATE
    )

    def check_series(self, series):
        """Each sample is rows 0 to 64 of its page, the first ROI, with the pages of the 3 files in order."""
        file_paths = sorted(self.FILE_PATH.parent.glob("planar_00001_*.tif"))
        pages = read_pages(*file_paths)
        assert_array_equal(series, pages[:, 0:64, :])


def test_gap_of_three_rois():
    """The last of three ROIs starts at row 2488, below two ROIs and the two gaps after them.

    The gap is the rows ScanImage saves between two ROIs while the scanner flies from one to the next. Three ROIs of 1002
    rows on a page of 3490 rows leave 3490 - 3 * 1002 = 484 rows, a gap of 242 rows at each of the two boundaries, so
    the last ROI starts at 2 * (1002 + 242) = 2488. The header gives the same gap: `flytoTimePerScanfield` of 0.01 s over
    a `linePeriod` of 4.15866e-05 s is 240.5 lines, rounded up to the even 242.
    """
    file_path = ACQUISITION_SETUP_PATH / "multi_roi_planar_three_rois_version_2021_1" / "planar_00002_00001.tif"
    extractor = ScanImageMultiROIImagingExtractor(file_path=file_path, roi_index=2)
    pages = read_pages(file_path)
    assert_array_equal(extractor.get_series(), pages[:, 2488:3490, :])


def test_get_available_rois_without_uuids():
    """The ROIs of a ScanImage 2018a file are listed with None as their `uuid`, as 2018a records no UUID.

    The file is volumetric, which the extractor does not read, but its ROIs can still be listed.
    """
    file_path = (
        SCANIMAGE_PATH
        / "tiff_format_version_3"
        / "binary_data_layout"
        / "multi_roi_volumetric_two_rois_of_different_heights_unidirectional_version_2018a"
        / "volume.tif"
    )
    rois = ScanImageMultiROIImagingExtractor.get_available_rois(file_path)
    assert [(roi["roi_index"], roi["name"], roi["uuid"]) for roi in rois] == [(0, "", None), (1, "", None)]


class TestScanImageMultiROIImagingExtractorErrors:
    """Errors of the constructor, on files it does not read or with arguments that do not select a ROI."""

    FILE_PATH = ACQUISITION_SETUP_PATH / "multi_roi_planar_eight_rois_version_2023_1" / "planar_00000_00001.tif"

    def test_roi_index_is_required_with_several_rois(self):
        expected_message = (
            "roi_index must be specified for data with more than one ROI. The file has 8 ROIs, use "
            "ScanImageMultiROIImagingExtractor.get_available_rois(file_path) to list them."
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            ScanImageMultiROIImagingExtractor(file_path=self.FILE_PATH)

    def test_roi_index_out_of_range_raises(self):
        with pytest.raises(ValueError, match=re.escape("roi_index (8) must be between 0 and 7.")):
            ScanImageMultiROIImagingExtractor(file_path=self.FILE_PATH, roi_index=8)

    def test_gaps_that_disagree_with_the_flyto_time_raise(self, tmp_path):
        """A file whose fly-to time gives another gap than the size of the page raises instead of being cropped by a guess.

        No such file exists on gin, so the fly-to time of the 2023.1 file is changed from 3.5 to 5.9 lines, which gives
        a gap of 6 rows instead of 4. The replacement has the same length, so the offsets in the file stay valid.
        """
        file_path = tmp_path / "planar_00000_00001.tif"
        shutil.copy(self.FILE_PATH, file_path)
        file_bytes = file_path.read_bytes()
        original_header_line = b"SI.hScan2D.flytoTimePerScanfield = 0.0001454080148897807"
        changed_header_line = b"SI.hScan2D.flytoTimePerScanfield = 0.0002454080148897807"
        file_path.write_bytes(file_bytes.replace(original_header_line, changed_header_line))

        expected_message = (
            "The gap between ROIs is 4 rows from the size of the page but 6 rows from SI.hScan2D.flytoTimePerScanfield "
            "and SI.hRoiManager.linePeriod."
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            ScanImageMultiROIImagingExtractor(file_path=file_path, roi_index=0)

    def test_volumetric_data_raises(self):
        file_path = (
            ACQUISITION_SETUP_PATH / "multi_roi_slow_stack_two_rois_channels_one_and_four_version_2023_0" / "stack.tif"
        )
        expected_message = "Volumetric multi-ROI data is not supported yet."
        with pytest.raises(NotImplementedError, match=re.escape(expected_message)):
            ScanImageMultiROIImagingExtractor(file_path=file_path, channel_name="Channel 1", roi_index=0)

    def test_file_without_multi_roi_raises(self):
        file_path = (
            SCANIMAGE_PATH
            / "tiff_format_version_4"
            / "time_structure"
            / "planar_frames_averaged_version_2022_0"
            / "planar.tif"
        )
        expected_message = (
            "This file is not a multi-ROI ScanImage acquisition (SI.hRoiManager.mroiEnable is not set). "
            "Use ScanImageImagingExtractor instead."
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            ScanImageMultiROIImagingExtractor(file_path=file_path)
