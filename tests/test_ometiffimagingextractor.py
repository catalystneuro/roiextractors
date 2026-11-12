"""Tests for the OMETiffImagingExtractor using real OME-TIFF test data."""

import re
import shutil
from datetime import datetime

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tifffile import TiffFile

from roiextractors import OMETiffImagingExtractor
from tests.extractor_mixins import ImagingExtractorTestMixin, check_metadata
from tests.setup_paths import OPHYS_DATA_PATH

OME_TIFF_PATH = OPHYS_DATA_PATH / "imaging_datasets" / "OME-TIFF"


class TestOMETiffImagingExtractor:
    """Test the OMETiffImagingExtractor with various OME-TIFF datasets."""

    def test_multi_channel_single_plane_multi_file(self):
        """Test with multi-channel, planar time series data across multiple files.

        Files: planar_two_channels_multi_file/tubhiswt_C{0,1}.ome.tif
        Source: tubhiswt-3D, stubbed from 20 to 5 timepoints, cropped to 64x64.
        - Samples (T): 5
        - Channels (C): 2 (one per file)
        - Depth planes (Z): 1
        - Frame shape: 64 x 64
        - Volumetric: False
        """
        file_path = OME_TIFF_PATH / "planar_two_channels_multi_file" / "tubhiswt_C0.ome.tif"

        extractor_ch0 = OMETiffImagingExtractor(file_path, sampling_frequency=1.0, channel_name="0")

        with TiffFile(file_path) as tiff:
            tiff_data_ch0 = np.stack([page.asarray() for page in tiff.pages])
        assert_array_equal(extractor_ch0.get_series(), tiff_data_ch0)

        ch1_path = OME_TIFF_PATH / "planar_two_channels_multi_file" / "tubhiswt_C1.ome.tif"
        extractor_ch1 = OMETiffImagingExtractor(file_path, sampling_frequency=1.0, channel_name="1")

        with TiffFile(ch1_path) as tiff:
            tiff_data_ch1 = np.stack([page.asarray() for page in tiff.pages])
        assert_array_equal(extractor_ch1.get_series(), tiff_data_ch1)

    def test_binary_only_companion_file(self):
        """Test a dataset whose Pixels metadata lives in a `*.companion.ome` sidecar.

        File: volumetric_single_channel_multifile_companion/multifile-Z1.ome.tiff
        Each TIFF carries only a `BinaryOnly` pointer to `multifile.companion.ome`, which holds the
        single Pixels block describing all five files.
        - Samples (T): 1
        - Channels (C): 1
        - Depth planes (Z): 5, one per file
        - Frame shape: 24 x 18
        - Volumetric: True
        """
        folder_path = OME_TIFF_PATH / "volumetric_single_channel_multifile_companion"
        file_path = folder_path / "multifile-Z1.ome.tiff"

        extractor = OMETiffImagingExtractor(file_path, sampling_frequency=1.0)

        expected_metadata = {"acquisition_start_datetime": datetime(2010, 3, 2, 15, 8, 59)}
        with pytest.warns(UserWarning, match="The metadata API is experimental"):
            check_metadata(extractor.metadata, expected_metadata)

        assert extractor.get_num_samples() == 1
        assert extractor.get_image_shape() == (24, 18)
        assert extractor.get_num_planes() == 5
        assert extractor.is_volumetric is True

        # Each plane must come from the file the companion names for that Z index
        extractor_data = extractor.get_series()
        for plane_index in range(5):
            plane_file_path = folder_path / f"multifile-Z{plane_index + 1}.ome.tiff"
            with TiffFile(plane_file_path) as tiff:
                expected = tiff.pages[0].asarray()
            assert_array_equal(extractor_data[0, :, :, plane_index], expected)

    def test_missing_companion_file_raises(self, tmp_path):
        """A BinaryOnly file whose companion is absent should say so, not fail on missing Pixels."""
        folder_path = OME_TIFF_PATH / "volumetric_single_channel_multifile_companion"
        for source_path in folder_path.glob("*.ome.tiff"):
            shutil.copy(source_path, tmp_path / source_path.name)

        file_path = tmp_path / "multifile-Z1.ome.tiff"
        companion_file_path = tmp_path / "multifile.companion.ome"
        expected_message = (
            f"The OME-XML of {file_path} points to the companion metadata file "
            f"'multifile.companion.ome', which was not found at '{companion_file_path}'. "
            "Companion files should be moved to the same folder for this to work."
        )

        with pytest.raises(FileNotFoundError, match=re.escape(expected_message)):
            OMETiffImagingExtractor(file_path, sampling_frequency=1.0)

    def test_sampling_frequency_from_time_increment(self):
        """Test that sampling_frequency is derived from TimeIncrement when not provided.

        File: planar_single_channel_single_file_with_time_increment/00001_01.ome.tiff
        Source: Same as planar_single_channel_single_file but with TimeIncrement="0.1" added
        to the Pixels element (0.1 seconds = 10 Hz).
        - Samples (T): 5
        - Channels (C): 1
        - Depth planes (Z): 1
        - Frame shape: 64 x 64
        - TimeIncrement: 0.1 s
        """
        file_path = OME_TIFF_PATH / "planar_single_channel_single_file_with_time_increment" / "00001_01.ome.tiff"

        extractor = OMETiffImagingExtractor(file_path)

        assert extractor.get_sampling_frequency() == 10.0

    def test_sampling_frequency_required_when_no_time_increment(self):
        """Test that ValueError is raised when neither sampling_frequency nor TimeIncrement is available."""
        file_path = OME_TIFF_PATH / "planar_single_channel_single_file" / "00001_01.ome.tiff"

        with pytest.raises(ValueError, match="sampling_frequency must be provided"):
            OMETiffImagingExtractor(file_path)

    def test_sampling_frequency_overrides_time_increment(self):
        """Test that an explicit sampling_frequency takes precedence over TimeIncrement."""
        file_path = OME_TIFF_PATH / "planar_single_channel_single_file_with_time_increment" / "00001_01.ome.tiff"

        extractor = OMETiffImagingExtractor(file_path, sampling_frequency=30.0)

        assert extractor.get_sampling_frequency() == 30.0

    def test_native_timestamps_from_delta_t(self):
        """Test that native timestamps are extracted from Plane/@DeltaT attributes.

        File: planar_single_channel_single_file_with_delta_t/00001_01.ome.tiff
        Source: Same as planar_single_channel_single_file but with 5 Plane elements added,
        each with non-uniform DeltaT values.
        - Samples (T): 5
        - Channels (C): 1
        - Depth planes (Z): 1
        - Frame shape: 64 x 64
        - DeltaT values: [0.0, 0.0651, 0.1337, 0.1982, 0.2671] seconds
        """
        file_path = OME_TIFF_PATH / "planar_single_channel_single_file_with_delta_t" / "00001_01.ome.tiff"

        extractor = OMETiffImagingExtractor(file_path, sampling_frequency=1.0)

        assert extractor._has_native_timestamps()
        timestamps = extractor.get_native_timestamps()
        assert timestamps is not None
        expected = np.array([0.0, 0.0651, 0.1337, 0.1982, 0.2671])
        np.testing.assert_allclose(timestamps, expected)

    def test_channel_names_from_ome_xml(self):
        """Test that real channel names from OME-XML Channel/@Name are used when present.

        Uses the Bruker dual-channel stub which has Channel/@Name="Ch1" and "Ch2".
        """
        bruker_path = OPHYS_DATA_PATH / "imaging_datasets" / "BrukerTif" / "TSeries-20240527-001"
        ome_file = bruker_path / "TSeries-20240527-001_Cycle00001_Ch1_000001.ome.tif"

        names = OMETiffImagingExtractor.get_available_channel_names(ome_file)
        assert names == ["Ch1", "Ch2"]

        extractor = OMETiffImagingExtractor(ome_file, sampling_frequency=1.0, channel_name="Ch1")
        assert extractor.get_num_samples() == 5

        expected_metadata = {
            "acquisition_start_datetime": datetime(2024, 5, 27, 19, 46, 12),
            "pixel_height_um": 1.129400083410842,
            "pixel_width_um": 1.1245597973390813,
            "field_of_view_height_um": 72.28160533829389,
            "field_of_view_width_um": 71.9718270297012,
            "channel_name": "Ch1",
        }
        with pytest.warns(UserWarning, match="The metadata API is experimental"):
            check_metadata(extractor.metadata, expected_metadata)

    def test_invalid_channel_name_shows_available(self):
        """Test that error message shows available channel names."""
        bruker_path = OPHYS_DATA_PATH / "imaging_datasets" / "BrukerTif" / "TSeries-20240527-001"
        ome_file = bruker_path / "TSeries-20240527-001_Cycle00001_Ch1_000001.ome.tif"

        with pytest.raises(ValueError, match="Channel 'bad' not found.*Available channels.*Ch1.*Ch2"):
            OMETiffImagingExtractor(ome_file, sampling_frequency=1.0, channel_name="bad")

    def test_file_not_found(self):
        """Test that a FileNotFoundError is raised for a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            OMETiffImagingExtractor("/nonexistent/path.ome.tif", sampling_frequency=1.0)


def read_ome_series(file_path, channel_index):
    """Read a channel with `tifffile` from the OME-XML, as (time, rows, columns) or (time, rows, columns, depth).

    `tifffile` builds the array from the OME-XML, independently of the extractor, across files when the OME-XML names
    several.
    """
    with TiffFile(file_path) as tiff_file:
        series = tiff_file.series[0]
        axes = series.axes
        data = series.asarray()
    assert set(axes) <= set("TCZYX"), f"Unexpected axes {axes}"
    # Add the axes the OME-XML leaves out, then order them as time, channel, rows, columns, depth
    missing_axes = "".join(axis for axis in "TCZ" if axis not in axes)
    data = data.reshape(data.shape + (1,) * len(missing_axes))
    axes += missing_axes
    data = np.transpose(data, [axes.index(axis) for axis in "TCYXZ"])
    channel_data = data[:, channel_index]
    return channel_data[..., 0] if channel_data.shape[-1] == 1 else channel_data


class TestOMETiffImagingExtractorPlanarSingleChannel(ImagingExtractorTestMixin):
    extractor_cls = OMETiffImagingExtractor
    extractor_kwargs = dict(
        file_path=OME_TIFF_PATH / "planar_single_channel_single_file" / "00001_01.ome.tiff", sampling_frequency=1.0
    )
    expected_num_samples = 5
    expected_sample_shape = (64, 64)
    expected_dtype = np.dtype("uint8")
    expected_channel_names = ["0"]
    expected_native_timestamps = None
    expected_sampling_frequency = 1.0  # passed to the constructor
    expected_metadata = {}

    def check_series(self, series):
        """The series is the channel of the array that `tifffile` builds from the OME-XML."""
        channel_name = self.extractor_kwargs.get("channel_name", self.expected_channel_names[0])
        channel_index = self.expected_channel_names.index(channel_name)
        expected_series = read_ome_series(self.extractor_kwargs["file_path"], channel_index=channel_index)
        assert_array_equal(series, expected_series)


class TestOMETiffImagingExtractorPlanarSingleChannelWithDeltaT(ImagingExtractorTestMixin):
    extractor_cls = OMETiffImagingExtractor
    extractor_kwargs = dict(
        file_path=OME_TIFF_PATH / "planar_single_channel_single_file_with_delta_t" / "00001_01.ome.tiff",
        sampling_frequency=1.0,
    )
    expected_num_samples = 5
    expected_sample_shape = (64, 64)
    expected_dtype = np.dtype("uint8")
    expected_channel_names = ["0"]
    # DeltaT of the first planes in the OME-XML
    expected_native_timestamps = [0.0, 0.0651, 0.1337]
    expected_sampling_frequency = 1.0  # passed to the constructor
    expected_metadata = {}

    def check_series(self, series):
        """The series is the channel of the array that `tifffile` builds from the OME-XML."""
        channel_name = self.extractor_kwargs.get("channel_name", self.expected_channel_names[0])
        channel_index = self.expected_channel_names.index(channel_name)
        expected_series = read_ome_series(self.extractor_kwargs["file_path"], channel_index=channel_index)
        assert_array_equal(series, expected_series)


class TestOMETiffImagingExtractorPlanarSingleChannelWithTimeIncrement(ImagingExtractorTestMixin):
    extractor_cls = OMETiffImagingExtractor
    extractor_kwargs = dict(
        file_path=OME_TIFF_PATH / "planar_single_channel_single_file_with_time_increment" / "00001_01.ome.tiff"
    )
    expected_num_samples = 5
    expected_sample_shape = (64, 64)
    expected_dtype = np.dtype("uint8")
    expected_channel_names = ["0"]
    expected_native_timestamps = None
    expected_sampling_frequency = 1 / 0.1  # 1 / TimeIncrement, 0.1 s in the OME-XML
    expected_metadata = {}

    def check_series(self, series):
        """The series is the channel of the array that `tifffile` builds from the OME-XML."""
        channel_name = self.extractor_kwargs.get("channel_name", self.expected_channel_names[0])
        channel_index = self.expected_channel_names.index(channel_name)
        expected_series = read_ome_series(self.extractor_kwargs["file_path"], channel_index=channel_index)
        assert_array_equal(series, expected_series)


class TestOMETiffImagingExtractorPlanarMultiChannel(ImagingExtractorTestMixin):
    extractor_cls = OMETiffImagingExtractor
    extractor_kwargs = dict(
        file_path=OME_TIFF_PATH / "planar_multi_channel_single_file" / "multi-channel-time-series.ome.tiff",
        sampling_frequency=1.0,
        channel_name="0",
    )
    expected_num_samples = 7
    expected_sample_shape = (64, 64)
    expected_dtype = np.dtype("int8")
    expected_channel_names = ["0", "1", "2"]  # numeric, the OME-XML has no Channel/@Name
    expected_native_timestamps = None
    expected_sampling_frequency = 1.0  # passed to the constructor
    expected_metadata = {}

    def check_series(self, series):
        """The series is the channel of the array that `tifffile` builds from the OME-XML."""
        channel_name = self.extractor_kwargs.get("channel_name", self.expected_channel_names[0])
        channel_index = self.expected_channel_names.index(channel_name)
        expected_series = read_ome_series(self.extractor_kwargs["file_path"], channel_index=channel_index)
        assert_array_equal(series, expected_series)


class TestOMETiffImagingExtractorPlanarTwoChannelsMultiFile(ImagingExtractorTestMixin):
    extractor_cls = OMETiffImagingExtractor
    extractor_kwargs = dict(
        file_path=OME_TIFF_PATH / "planar_two_channels_multi_file" / "tubhiswt_C0.ome.tif",
        sampling_frequency=1.0,
        channel_name="0",
    )
    expected_num_samples = 5
    expected_sample_shape = (64, 64)
    expected_dtype = np.dtype("uint8")
    expected_channel_names = ["0", "1"]
    expected_native_timestamps = None
    expected_sampling_frequency = 1.0  # passed to the constructor
    expected_metadata = {
        "acquisition_start_datetime": datetime(2013, 1, 15, 17, 2, 41),
        "experimenter": ["Maimoon Nasim"],
    }

    def check_series(self, series):
        """The series is the channel of the array that `tifffile` builds from the OME-XML."""
        channel_name = self.extractor_kwargs.get("channel_name", self.expected_channel_names[0])
        channel_index = self.expected_channel_names.index(channel_name)
        expected_series = read_ome_series(self.extractor_kwargs["file_path"], channel_index=channel_index)
        assert_array_equal(series, expected_series)


class TestOMETiffImagingExtractorVolumetricSingleChannel(ImagingExtractorTestMixin):
    extractor_cls = OMETiffImagingExtractor
    extractor_kwargs = dict(
        file_path=OME_TIFF_PATH / "volumetric_single_channel_single_file" / "4D-series.ome.tiff", sampling_frequency=1.0
    )
    expected_num_samples = 7
    expected_sample_shape = (64, 64, 5)
    expected_dtype = np.dtype("int8")
    expected_channel_names = ["0"]
    expected_native_timestamps = None
    expected_sampling_frequency = 1.0  # passed to the constructor
    expected_metadata = {}

    def check_series(self, series):
        """The series is the channel of the array that `tifffile` builds from the OME-XML."""
        channel_name = self.extractor_kwargs.get("channel_name", self.expected_channel_names[0])
        channel_index = self.expected_channel_names.index(channel_name)
        expected_series = read_ome_series(self.extractor_kwargs["file_path"], channel_index=channel_index)
        assert_array_equal(series, expected_series)


class TestOMETiffImagingExtractorVolumetricMultiChannel(ImagingExtractorTestMixin):
    extractor_cls = OMETiffImagingExtractor
    extractor_kwargs = dict(
        file_path=OME_TIFF_PATH / "volumetric_multi_channel_single_file" / "multi-channel-4D-series.ome.tiff",
        sampling_frequency=1.0,
        channel_name="0",
    )
    expected_num_samples = 7
    expected_sample_shape = (64, 64, 5)
    expected_dtype = np.dtype("int8")
    expected_channel_names = ["0", "1", "2"]
    expected_native_timestamps = None
    expected_sampling_frequency = 1.0  # passed to the constructor
    expected_metadata = {}

    def check_series(self, series):
        """The series is the channel of the array that `tifffile` builds from the OME-XML."""
        channel_name = self.extractor_kwargs.get("channel_name", self.expected_channel_names[0])
        channel_index = self.expected_channel_names.index(channel_name)
        expected_series = read_ome_series(self.extractor_kwargs["file_path"], channel_index=channel_index)
        assert_array_equal(series, expected_series)


class TestOMETiffImagingExtractorVolumetricTwoChannelsMultiFile(ImagingExtractorTestMixin):
    extractor_cls = OMETiffImagingExtractor
    extractor_kwargs = dict(
        file_path=OME_TIFF_PATH / "volumetric_two_channels_multi_file" / "tubhiswt_C0_TP0.ome.tif",
        sampling_frequency=1.0,
        channel_name="0",
    )
    expected_num_samples = 3
    expected_sample_shape = (64, 64, 10)
    expected_dtype = np.dtype("uint8")
    expected_channel_names = ["0", "1"]
    expected_native_timestamps = None
    expected_sampling_frequency = 1.0  # passed to the constructor
    expected_metadata = {
        "acquisition_start_datetime": datetime(2013, 1, 15, 17, 2, 48),
        "experimenter": ["Maimoon Nasim"],
    }

    def check_series(self, series):
        """The series is the channel of the array that `tifffile` builds from the OME-XML."""
        channel_name = self.extractor_kwargs.get("channel_name", self.expected_channel_names[0])
        channel_index = self.expected_channel_names.index(channel_name)
        expected_series = read_ome_series(self.extractor_kwargs["file_path"], channel_index=channel_index)
        assert_array_equal(series, expected_series)
