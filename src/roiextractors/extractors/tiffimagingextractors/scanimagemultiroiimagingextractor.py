"""Extractor for one ROI of a ScanImage multi-ROI (mesoscope) TIFF file.

Classes
-------
ScanImageMultiROIImagingExtractor
    Extractor for one ROI of a ScanImage multi-ROI (mesoscope) TIFF file.
"""

import math
from dataclasses import replace
from functools import cached_property

import numpy as np

from .scanimagetiffimagingextractor import ScanImageImagingExtractor
from ..._metadata import Metadata
from ...extraction_tools import PathType


class ScanImageMultiROIImagingExtractor(ScanImageImagingExtractor):
    """Extractor for one ROI of a ScanImage multi-ROI (mesoscope) TIFF file.

    A multi-ROI acquisition scans several rectangular fields and saves them in one page per channel, stacked from top
    to bottom along the slow scan axis, with dead rows between consecutive ROIs while the scanner flies to the next
    field. This extractor exposes one ROI as a planar imaging series by cropping its rows out of each page as it is
    read. Construct one extractor per ROI.

    ::

        Sample (where the scanner goes)           Page (what is stored)
        +------------------------------+          +---------+
        | +-----+                      |          |  ROI 0  |
        | |ROI 0|          +-----+     |          +---------+
        | +-----+          |ROI 1|     |          |   gap   |  fly-to rows
        |                  +-----+     |          +---------+
        |    +-----+                   |          |  ROI 1  |
        |    |ROI 2|                   |          +---------+
        |    +-----+                   |          |   gap   |
        +------------------------------+          +---------+
                                                  |  ROI 2  |
                                                  +---------+

    Each scan line of a ROI becomes one row of the page, so the position of a ROI in the sample does not change where
    it lands on the page. Every ROI must have the width of the page.

    Notes
    -----
    Only planar recordings whose ROIs have the same width are supported. Volumetric multi-ROI recordings and ROIs of
    different widths raise ``NotImplementedError``.

    The native timestamps are the times at which the ROI is scanned: the ``frameTimestamps_sec`` of each page plus
    ``SI.hRoiManager.linePeriod`` times the rows above the ROI. Each row of the page is one scan line, so the scanner
    reaches a row ``SI.hRoiManager.linePeriod`` seconds after the row above it. ``ScanImageImagingExtractor`` on the
    same file returns the times of the pages.
    """

    extractor_name = "ScanImageMultiROIImagingExtractor"

    def __init__(
        self,
        file_path: PathType | None = None,
        channel_name: str | None = None,
        file_paths: list[PathType] | None = None,
        roi_index: int | None = None,
    ):
        """Initialize the ScanImageMultiROIImagingExtractor.

        Parameters
        ----------
        file_path : PathType, optional
            Path to the ScanImage TIFF file. If this is part of a multi-file series, this should be the first file.
            Either `file_path` or `file_paths` must be provided.
        channel_name : str, optional
            Name of the channel to extract. Required only when the file has several channels. Use
            `get_available_channel_names(file_path)` to see the available channels.
        file_paths : list[PathType], optional
            List of file paths in the temporal order of the frames. If provided, this overrides the automatic file
            detection.
        roi_index : int, optional
            Position of the ROI on the page, counted from the top and starting at 0. Required only when the file has
            several ROIs. Use `get_available_rois(file_path)` to see the available ROIs.
        """
        # A volumetric recording raises below. Without `slice_sample`, one with several frames per slice would
        # raise first, asking for a `slice_sample` this extractor does not take.
        super().__init__(file_path=file_path, channel_name=channel_name, file_paths=file_paths, slice_sample=0)

        if not self._metadata.get("SI.hRoiManager.mroiEnable", False):
            raise ValueError(
                "This file is not a multi-ROI ScanImage acquisition (SI.hRoiManager.mroiEnable is not set). "
                "Use ScanImageImagingExtractor instead."
            )
        if self.is_volumetric:
            raise NotImplementedError("Volumetric multi-ROI data is not supported yet.")

        rois = self._get_enabled_rois(self._general_metadata)
        num_rois = len(rois)
        if roi_index is None:
            if num_rois > 1:
                raise ValueError(
                    f"roi_index must be specified for data with more than one ROI. The file has {num_rois} ROIs, "
                    "use ScanImageMultiROIImagingExtractor.get_available_rois(file_path) to list them."
                )
            roi_index = 0
        if not 0 <= roi_index < num_rois:
            raise ValueError(f"roi_index ({roi_index}) must be between 0 and {num_rois - 1}.")

        scanfields = [roi["scanfields"] for roi in rois]
        if any(isinstance(scanfield, list) for scanfield in scanfields):
            raise NotImplementedError("Volumetric multi-ROI data (one scanfield per plane) is not supported yet.")
        roi_widths = [scanfield["pixelResolutionXY"][0] for scanfield in scanfields]
        roi_heights = [scanfield["pixelResolutionXY"][1] for scanfield in scanfields]
        if len(set(roi_widths)) != 1:
            raise NotImplementedError(
                f"The ROIs have different widths ({roi_widths}), only ROIs of the same width are supported."
            )

        page_num_rows = self._num_rows
        gap = self._get_inter_roi_gap(page_num_rows=page_num_rows, roi_heights=roi_heights)

        # The ROI starts below the ROIs above it and the gap after each of them. The crop window of the base class
        # selects its rows, so `get_series` crops each page as it reads it.
        roi_row_start = sum(roi_heights[:roi_index]) + roi_index * gap
        roi_row_end = roi_row_start + roi_heights[roi_index]

        # The time the scanner takes to fly to the next ROI gives a second estimate of the gap: the fly-to time over
        # the line period, rounded up to an even number of lines, as a resonant scanner scanning in both directions
        # inserts lines in pairs. A file where the two disagree raises instead of being cropped by a guess.
        flyto_time = self._metadata["SI.hScan2D.flytoTimePerScanfield"]
        line_period = self._metadata["SI.hRoiManager.linePeriod"]
        timing_gap = math.ceil(flyto_time / line_period / 2) * 2
        if num_rois > 1 and gap != timing_gap:
            channel_argument = f", channel_name={self.channel_name!r}" if self._num_channels > 1 else ""
            raise ValueError(
                f"The gap between ROIs is {gap} rows from the size of the page but {timing_gap} rows from "
                "SI.hScan2D.flytoTimePerScanfield and SI.hRoiManager.linePeriod. Please open an issue with an example "
                "file at https://github.com/catalystneuro/roiextractors/issues\n\n"
                f"In the meantime, you can read ROI {roi_index} from the whole page, assuming the {gap}-row gap:\n\n"
                "    from roiextractors import ScanImageImagingExtractor\n\n"
                f"    extractor = ScanImageImagingExtractor(file_path={str(self.file_path)!r}{channel_argument})\n"
                f"    roi = extractor.slice_field_of_view(row_start={roi_row_start}, row_end={roi_row_end})\n\n"
                "Its timestamps are those of the page, without the time it takes to scan the rows above the ROI."
            )

        self._roi_index = roi_index
        self._narrow_field_of_view(
            row_start=roi_row_start,
            row_end=roi_row_end,
            column_start=0,
            column_end=self._num_columns,
        )

    @staticmethod
    def _get_enabled_rois(scanimage_metadata: dict) -> list[dict]:
        """Return the enabled ROIs in their order on the page. ScanImage does not scan a disabled ROI."""
        rois = scanimage_metadata["RoiGroups"]["imagingRoiGroup"]["rois"]
        # A group with one ROI is read as a dict
        rois = [rois] if isinstance(rois, dict) else rois
        return [roi for roi in rois if roi.get("enable", True)]

    @staticmethod
    def _get_inter_roi_gap(page_num_rows: int, roi_heights: list[int]) -> int:
        """Return the number of dead rows between consecutive ROIs on the page.

        The gap is the rows of the page left over by the ROIs, spread evenly between them.
        """
        num_rois = len(roi_heights)
        if num_rois == 1:
            return 0

        num_gaps = num_rois - 1
        residual_rows = page_num_rows - sum(roi_heights)
        if residual_rows < 0 or residual_rows % num_gaps != 0:
            raise ValueError(
                f"The {page_num_rows} rows of the page do not fit ROIs of heights {roi_heights} separated by equal gaps."
            )
        return residual_rows // num_gaps

    @staticmethod
    def get_available_rois(file_path: PathType) -> list[dict]:
        """Get the ROIs of a ScanImage multi-ROI TIFF file.

        Disabled ROIs are left out, as ScanImage does not scan them. The names are given by the user in ScanImage and
        can repeat. The UUID is the identifier ScanImage gives each ROI, unique within the file. It is not a stable
        identity: the same ROI can have another UUID in another acquisition, even in the same session, so match ROIs
        across recordings by their position or their geometry, not by their UUID.

        The center and the size come from the ROI's scanfield, in ScanImage reference coordinates (scan angles in
        degrees), converted to micrometers with ``SI.objectiveResolution``, the micrometers per degree of the
        objective. The center is relative to the center of the scan, not to the stage or the sample.

        Parameters
        ----------
        file_path : PathType
            Path to the ScanImage TIFF file.

        Returns
        -------
        list[dict]
            One dict for each ROI, in their order on the page from the top, with its `roi_index`, the argument of the
            constructor, its `name`, its `uuid`, which is None in files that do not record it (ScanImage 2018a and
            earlier), and its `center_xy_um` and `size_xy_um`, which are None for a ROI with one scanfield per plane.

        Examples
        --------
        >>> rois = ScanImageMultiROIImagingExtractor.get_available_rois('path/to/file.tif')
        >>> extractor = ScanImageMultiROIImagingExtractor(file_path='path/to/file.tif', roi_index=rois[0]["roi_index"])
        """
        from tifffile import read_scanimage_metadata

        with open(file_path, "rb") as file_handle:
            frame_data, roi_group_metadata, _ = read_scanimage_metadata(file_handle)
        micrometers_per_degree = frame_data["SI.objectiveResolution"]

        rois = ScanImageMultiROIImagingExtractor._get_enabled_rois(roi_group_metadata)
        available_rois = []
        for roi_index, roi in enumerate(rois):
            scanfield = roi["scanfields"]
            center_xy_um, size_xy_um = None, None
            if isinstance(scanfield, dict):  # a list holds one scanfield per plane
                center_xy_um = tuple(float(value) * micrometers_per_degree for value in scanfield["centerXY"])
                size_xy_um = tuple(float(value) * micrometers_per_degree for value in scanfield["sizeXY"])
            available_roi = dict(
                roi_index=roi_index,
                name=roi["name"],
                uuid=roi.get("roiUuid"),
                center_xy_um=center_xy_um,
                size_xy_um=size_xy_um,
            )
            available_rois.append(available_roi)
        return available_rois

    @cached_property
    def _unified_metadata(self) -> Metadata:
        """The metadata of the recording, with the field of view and the pixel size of the ROI.

        The field of view is the ``sizeXY`` of the ROI's scanfield, in ScanImage reference coordinates (scan angles in
        degrees), converted to micrometers with ``SI.objectiveResolution``. The pixel size divides it by the
        ``pixelResolutionXY`` of the scanfield, the columns and the rows of the ROI.
        """
        metadata = ScanImageImagingExtractor._unified_metadata.func(self)
        # A view of this extractor takes the metadata of the ROI from it
        if self._metadata_parent is not None:
            return metadata

        scanfield = self._get_enabled_rois(self._general_metadata)[self._roi_index]["scanfields"]
        micrometers_per_degree = self._metadata["SI.objectiveResolution"]
        field_of_view_width_um = float(scanfield["sizeXY"][0]) * micrometers_per_degree
        field_of_view_height_um = float(scanfield["sizeXY"][1]) * micrometers_per_degree
        num_columns, num_rows = scanfield["pixelResolutionXY"]
        return replace(
            metadata,
            field_of_view_height_um=field_of_view_height_um,
            field_of_view_width_um=field_of_view_width_um,
            pixel_height_um=field_of_view_height_um / num_rows,
            pixel_width_um=field_of_view_width_um / num_columns,
        )

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        """Retrieve the times at which the ROI is scanned.

        Each row of the page is one scan line, so the scanner reaches a row ``SI.hRoiManager.linePeriod`` seconds after
        the row above it. The scanner sweeps the page from the top, so it reaches the first row of the ROI
        ``SI.hRoiManager.linePeriod`` seconds per row above it after the ``frameTimestamps_sec`` of the page.

        Parameters
        ----------
        start_sample : int, optional
            The starting sample index. If None, starts from the beginning.
        end_sample : int, optional
            The ending sample index. If None, goes to the end.

        Returns
        -------
        timestamps: numpy.ndarray
            The times in seconds at which the first row of the ROI is scanned.
        """
        page_timestamps = super().get_native_timestamps(start_sample=start_sample, end_sample=end_sample)
        line_period = self._metadata["SI.hRoiManager.linePeriod"]
        return page_timestamps + line_period * self._row_slice.start

    def slice_field_of_view(
        self,
        row_start: int | None = None,
        row_end: int | None = None,
        column_start: int | None = None,
        column_end: int | None = None,
    ) -> "ScanImageMultiROIImagingExtractor":
        """Return a spatially cropped view of the ROI as a new ScanImageMultiROIImagingExtractor.

        The rows are relative to the ROI. A crop that starts at a lower row is scanned later in each frame, so the
        view has its own native timestamps.

        Parameters
        ----------
        row_start : int, optional
            Starting row index (inclusive). Default is 0.
        row_end : int, optional
            Ending row index (exclusive). Default is the full current height.
        column_start : int, optional
            Starting column index (inclusive). Default is 0.
        column_end : int, optional
            Ending column index (exclusive). Default is the full current width.

        Returns
        -------
        ScanImageMultiROIImagingExtractor
            A new extractor exposing only the requested field of view.
        """
        sliced = super().slice_field_of_view(
            row_start=row_start, row_end=row_end, column_start=column_start, column_end=column_end
        )
        # The copy has the timestamps of the full ROI if they were read before
        sliced._native_timestamps_cache = None
        return sliced
