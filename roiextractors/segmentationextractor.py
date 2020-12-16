from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from spikeextractors.baseextractor import BaseExtractor

from .extraction_tools import ArrayType
from .extraction_tools import _pixel_mask_extractor


class SegmentationExtractor(ABC, BaseExtractor):
    """
    An abstract class that contains all the meta-data and output data from
    the ROI segmentation operation when applied to the pre-processed data.
    It also contains methods to read from and write to various data formats
    ouput from the processing pipelines like SIMA, CaImAn, Suite2p, CNNM-E.
    All the methods with @abstract decorator have to be defined by the
    format specific classes that inherit from this.
    """

    def __init__(self):
        BaseExtractor.__init__(self)
        self._sampling_frequency = None
        self._channel_names = ['OpticalChannel']
        self._num_planes = 1
        self._roi_response_raw = None
        self._roi_response_dff = None
        self._roi_response_neuropil = None
        self._roi_response_deconvolved = None
        self._image_correlation = None
        self._image_mean = None

    @abstractmethod
    def get_accepted_list(self) -> list:
        """
        The ids of the ROIs which are accepted after manual verification of
        ROIs.

        Returns
        -------
        accepted_list: list
            List of accepted ROIs
        """
        pass

    @abstractmethod
    def get_rejected_list(self) -> list:
        """
        The ids of the ROIs which are rejected after manual verification of
        ROIs.

        Returns
        -------
        accepted_list: list
            List of rejected ROIs
        """
        pass

    @property
    def roi_locations(self):
        roi_location = np.ndarray([2, self.get_num_rois()], dtype='int')
        for i in range(self.get_num_rois()):
            temp = np.where(self.image_masks[:, :, i] == np.amax(self.image_masks[:, :, i]))
            roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
        return roi_location

    def get_num_frames(self) -> int:
        """This function returns the number of frames in the recording.

        Returns
        -------
        num_of_frames: int
            Number of frames in the recording (duration of recording).
        """
        for trace in self.get_traces_dict().values():
            if len(trace.shape) > 0:
                return trace.shape[1]

    def get_roi_locations(self, roi_ids=None) -> np.array:
        """
        Returns the locations of the Regions of Interest

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

        Returns
        ------
        roi_locs: numpy.ndarray
            2-D array: 2 X no_ROIs. The pixel ids (x,y) where the centroid of the ROI is.
        """
        if roi_ids is None:
            return self.roi_locations
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
            return self.roi_locations[:, roi_idx_]

    @abstractmethod
    def get_roi_ids(self) -> list:
        """Returns the list of channel ids. If not specified, the range from 0 to num_channels - 1 is returned.

        Returns
        -------
        channel_ids: list
            Channel list.
        """
        pass

    def get_roi_image_masks(self, roi_ids=None) -> np.array:
        """Returns the image masks extracted from segmentation algorithm.

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

        Returns
        -------
        image_masks: numpy.ndarray
            3-D array(val 0 or 1): image_height X image_width X length(roi_ids)
        """
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return np.array(self.image_masks)[:, :, roi_idx_]

    def get_roi_pixel_masks(self, roi_ids=None) -> np.array:
        """
        Returns the weights applied to each of the pixels of the mask.

        Parameters
        ----------
        roi_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

        Returns
        -------
        pixel_masks: [list, NoneType]
            list of length number of rois, each element is a 2-D array os shape (no-pixels, 2)
        """
        if roi_ids is None:
            return None
        return _pixel_mask_extractor(self.get_roi_image_masks(roi_ids=roi_ids), range(len(roi_ids)))

    @abstractmethod
    def get_image_size(self) -> ArrayType:
        """
        Frame size of movie ( x and y size of image).

        Returns
        -------
        no_rois: array_like
            2-D array: image y x image x
        """
        pass

    def get_traces(self, roi_ids=None, start_frame=None, end_frame=None, name='raw'):
        """
        Return RoiResponseSeries
        Returns
        -------
        traces: array_like
            2-D array (ROI x timepoints)
        """
        if name not in self.get_traces_dict():
            raise ValueError(f'traces for {name} not found, enter one of {list(self.get_traces_dict().keys())}')
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        traces = self.get_traces_dict().get(name)
        if len(traces.shape)!=0:
            return np.array([traces[int(i), start_frame:end_frame] for i in roi_idx_])

    def get_traces_dict(self):
        """
        Returns traces as a dictionary with key as the name of the ROiResponseSeries
        Returns
        -------
        _roi_response_dict: dict
            dictionary with key, values representing different types of RoiResponseSeries
            Flourescence, Neuropil, Deconvolved, Background etc
        """
        return deepcopy(dict(raw=np.array(self._roi_response_raw),
                             dff=np.array(self._roi_response_dff),
                             neuropil=np.array(self._roi_response_neuropil),
                             deconvolved=np.array(self._roi_response_deconvolved)))

    def get_images_dict(self):
        """
        Returns traces as a dictionary with key as the name of the ROiResponseSeries
        Returns
        -------
        _roi_response_dict: dict
            dictionary with key, values representing different types of Images used in segmentation:
            Mean, Correlation image
        """
        return deepcopy(dict(mean=self._image_mean,
                             correlation=self._image_correlation))

    def get_image(self, name='correlation'):
        """
        Return specific images: mean or correlation
        Parameters
        ----------
        name:str
            name of the type of image to retrieve
        Returns
        -------
        images: np.ndarray
        """
        if name not in self.get_images_dict():
            raise ValueError(f'could not find {name} image, enter one of {list(self.get_images_dict().keys())}')
        return self.get_images_dict().get(name)

    def get_sampling_frequency(self):
        """This function returns the sampling frequency in units of Hz.

        Returns
        -------
        samp_freq: float
            Sampling frequency of the recordings in Hz.
        """
        return np.float(self._sampling_frequency)

    def get_num_rois(self):
        """Returns total number of Regions of Interest in the acquired images.

        Returns
        -------
        no_rois: int
            integer number of ROIs extracted.
        """
        for trace in self.get_traces_dict().values():
            if len(trace.shape) > 0:
                return trace.shape[0]

    def get_channel_names(self):
        """
        Names of channels in the pipeline
        Returns
        -------
        _channel_names: list
            names of channels (str)
        """
        return self._channel_names

    def get_num_channels(self):
        """
        Number of channels in the pipeline
        Returns
        -------
        num_of_channels: int
        """
        return len(self._channel_names)

    def get_num_planes(self):
        """
        Returns the default number of planes of imaging for the segmentation extractor.
        Detaults to 1 for all but the MultiSegmentationExtractor
        Returns
        -------
        self._num_planes: int
        """
        return self._num_planes

    @staticmethod
    def write_segmentation(segmentation_extractor, save_path, overwrite=False):
        """
        Static method to write recording back to the native format.

        Parameters
        ----------
        segmentation_extractor: [SegmentationExtractor, MultiSegmentationExtractor]
            The EXTRACT segmentation object from which an EXTRACT native format
            file has to be generated.
        save_path: str
            path to save the native format.
        overwrite: bool
            If True, the file is overwritten if existing (default False)
        """
        raise NotImplementedError
