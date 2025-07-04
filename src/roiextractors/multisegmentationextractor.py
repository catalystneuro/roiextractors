"""Defines the MultiSegmentationExtractor class.

Classes
-------
MultiSegmentationExtractor
    This class is used to combine multiple SegmentationExtractor objects by frames.
"""

import warnings

import numpy as np

from .segmentationextractor import SegmentationExtractor


def concatenate_output(func):  # TODO: refactor to avoid magical behavior
    """Concatenate output of single SegmentationExtractor methods.

    Parameters
    ----------
    func: function
        function to be decorated

    Returns
    -------
    _get_from_roi_map: function
        decorated function
    """

    def _get_from_roi_map(self, roi_ids=None, **kwargs):
        """Call member function of each SegmentationExtractor specified by func and concatenate the output.

        Parameters
        ----------
        roi_ids: list
            list of roi ids to be used
        kwargs: dict
            keyword arguments to be passed to func

        Returns
        -------
        out: list
            list of outputs from each SegmentationExtractor
        """
        out = []
        if roi_ids is None:
            roi_ids = np.array(self._all_roi_ids)
        else:
            roi_ids = np.array(roi_ids)
        seg_id = np.array([self._roi_map[roi_id]["segmentation_id"] for roi_id in roi_ids])
        roi_id_segmentation = np.array([self._roi_map[roi_id]["roi_id"] for roi_id in roi_ids])
        for i in np.unique(seg_id):
            seg_roi_ids = roi_id_segmentation[seg_id == i]
            out.append(getattr(self._segmentations[i], func.__name__)(roi_ids=seg_roi_ids, **kwargs))
        return func(self)(out)

    return _get_from_roi_map


class MultiSegmentationExtractor(SegmentationExtractor):
    """Class is used to concatenate multi-plane recordings from the same device and session of experiment."""

    extractor_name = "MultiSegmentationExtractor"
    mode = "file"
    installation_mesg = ""  # error message when not installed

    def __init__(self, segmentatation_extractors_list, plane_names=None):  # TODO: Hungarian notation --> type hints
        """Initialize a MultiSegmentationExtractor object from a list of SegmentationExtractors.

        Parameters
        ----------
        segmentatation_extractors_list: list of SegmentationExtractor
            list of segmentation extractor objects (one for each plane)
        plane_names: list
            list of strings of names for the plane. Defaults to 'Plane0', 'Plane1' ...
        """
        SegmentationExtractor.__init__(self)
        if not isinstance(segmentatation_extractors_list, list):
            raise Exception("Enter a list of segmentation extractor objects as argument")
        self._no_planes = len(segmentatation_extractors_list)
        if plane_names:
            plane_names = list(plane_names)
            if len(plane_names) >= self._no_planes:
                plane_names = plane_names[: self._no_planes]
            else:
                plane_names.extend([f"Plane{i}" for i in range(self._no_planes - len(plane_names))])
        else:
            plane_names = [f"Plane{i}" for i in range(self._no_planes)]
        self._segmentations = segmentatation_extractors_list
        self._all_roi_ids = []
        self._roi_map = {}

        s_id = 0
        for s_i, segmentation in enumerate(self._segmentations):
            roi_ids = segmentation.get_roi_ids()
            for roi_id in roi_ids:
                self._all_roi_ids.append(s_id)
                self._roi_map[s_id] = {"segmentation_id": s_i, "roi_id": roi_id}
                s_id += 1
        self._plane_names = plane_names
        self._sampling_frequency = self._segmentations[0].get_sampling_frequency()
        self._raw_movie_file_location = self._segmentations[0]._raw_movie_file_location
        self._channel_names = []
        _ = [self._channel_names.extend(self._segmentations[i].get_channel_names()) for i in range(self._no_planes)]

    @property
    def no_planes(self) -> int:
        """Number of planes in the recording.

        Returns
        -------
        no_planes: int
            number of planes in the recording
        """
        return self._no_planes

    @property
    def segmentations(self) -> list[SegmentationExtractor]:
        """List of segmentation extractors (one for each plane).

        Returns
        -------
        segmentations: list
            list of segmentation extractors (one for each plane)
        """
        return self._segmentations

    def get_num_channels(self):
        return np.sum([self._segmentations[i].get_num_channels() for i in range(self._no_planes)])

    def get_num_rois(self) -> int:
        return len(self._all_roi_ids)

    def get_images(self, name="correlation_plane0"):  # TODO: add get_images to base SegmentationExtractor class
        """Get images from the imaging extractors.

        Parameters
        ----------
        name: str
            name of the image to get

        Returns
        -------
        images: numpy.ndarray
            Array of images.
        """
        plane_no = int(name[-1])
        return self._segmentations[plane_no].get_images(name=name.split("_")[0])

    def get_images_dict(self) -> dict:
        return_dict = dict()
        for i in range(self._no_planes):
            for image_name, image in self._segmentations[i].get_images_dict().items():
                return_dict.update({f"{image_name}_Plane{i}": image})
        return return_dict

    def get_traces_dict(self) -> dict:
        return_dict = dict()
        for i in range(self._no_planes):
            for trace_name, trace in self._segmentations[i].get_traces_dict().items():
                return_dict.update({f"{trace_name}_Plane{i}": trace})
        return return_dict

    def get_frame_shape(self) -> tuple[int, int]:
        return self._segmentations[0].get_frame_shape()

    def get_image_size(self) -> tuple[int, int]:
        warnings.warn(
            "get_image_size is deprecated and will be removed on or after January 2026. "
            "Use get_frame_shape instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_frame_shape()

    @concatenate_output
    def get_traces(self, roi_ids=None, start_frame=None, end_frame=None, name="Fluorescence"):
        return lambda x: np.concatenate(x, axis=0)

    @concatenate_output
    def get_roi_pixel_masks(self, roi_ids=None):
        return lambda x: [j for i in x for j in i]

    @concatenate_output
    def get_roi_image_masks(self, roi_ids=None):
        return lambda x: np.concatenate(x, axis=2)

    @concatenate_output
    def get_roi_locations(self, roi_ids=None):
        return lambda x: np.concatenate(x, axis=1)

    def get_num_frames(self):
        return np.sum([self._segmentations[i].get_num_frames() for i in range(self._no_planes)])

    def get_accepted_list(self) -> list[int]:
        accepted_list_all = []
        for i in range(self._no_planes):
            ids_loop = self._segmentations[i].get_accepted_list()
            accepted_list_all.extend([j for j in self._all_roi_ids if self._roi_map[j]["roi_id"] in ids_loop])
        return accepted_list_all

    def get_rejected_list(self) -> list[int]:
        rejected_list_all = []
        for i in range(self._no_planes):
            ids_loop = self._segmentations[i].get_rejected_list()
            rejected_list_all.extend([j for j in self._all_roi_ids if self._roi_map[j]["roi_id"] in ids_loop])
        return rejected_list_all
