import numpy as np
from pathlib import Path
from ...imagingextractor import ImagingExtractor
from ...segmentationextractor import SegmentationExtractor
from ...extraction_tools import check_get_frames_args, get_video_shape, _pixel_mask_extractor


# TODO this class should also be able to instantiate an in-memory object (useful for testing)
class NumpyImagingExtractor(ImagingExtractor):
    def __init__(self, file_path, sampling_frequency=None,
                 channel_names=None):

        ImagingExtractor.__init__(self)
        self.file_path = Path(file_path)
        self._sampling_frequency = sampling_frequency
        assert self.file_path.suffix == '.npy'
        self._video = np.load(self.file_path, mmap_mode='r')
        self._channel_names = channel_names

        self._num_channels, self._num_frames, self._size_x, self._size_y = get_video_shape(self._video)

        if len(self._video.shape) == 3:
            # check if this converts to np.ndarray
            self._video = self._video[np.newaxis, :]

        if self._channel_names is not None:
            assert len(self._channel_names) == self._num_channels, "'channel_names' length is different than number " \
                                                                   "of channels"
        else:
            self._channel_names = [f'channel_{ch}' for ch in range(self._num_channels)]

    @check_get_frames_args
    def get_frames(self, frame_idxs, channel=0):
        return self._video[channel, frame_idxs]

    def get_image_size(self):
        return [self._size_x, self._size_y]

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_channel_names(self):
        """List of  channels in the recoding.

        Returns
        -------
        channel_names: list
            List of strings of channel names
        """
        return self._channel_names

    def get_num_channels(self):
        """Total number of active channels in the recording

        Returns
        -------
        no_of_channels: int
            integer count of number of channels
        """
        return self._num_channels

    @staticmethod
    def write_imaging(imaging, save_path):
        save_path = Path(save_path)
        assert save_path.suffix == '.npy', "'save_path' should havve a .npy extension"

        np.save(save_path, imaging.get_video())


class NumpySegmentationExtractor(SegmentationExtractor):
    """
    NumpySegmentationExtractor objects are built to contain all data coming from
    a file format for which there is currently no support. To construct this,
    all data must be entered manually as arguments.
    """

    def __init__(self, image_masks, signal,
                 rawfileloc=None, accepted_lst=None,
                 mean_image=None, correlation_image=None,
                 roi_idx=None, roi_locs=None, samp_freq=None,
                 rejected_list=None, channel_names=None,
                 movie_dims=None):
        """
        Parameters:
        ----------
        file_path: str
            The location of the folder containing the custom file format.
        image_masks: np.ndarray
            Binary image for each of the regions of interest
        signal: np.ndarray
            Fluorescence response of each of the ROI in time
        mean_image: np.ndarray
            Mean image
        correlation_image: np.ndarray
            correlation image
        roi_idx: int list
            Unique ids of the ROIs if any
        roi_locs: np.ndarray
            x and y location representative of ROI mask
        samp_freq: float
            Frame rate of the movie
        rejected_list: list
            list of ROI ids that are rejected manually or via automated rejection
        channel_names: list
            list of strings representing channel names
        movie_dims: list
            height x width of the movie
        """
        SegmentationExtractor.__init__(self)
        self.image_masks = image_masks
        self._roi_response_raw = signal
        self._movie_dims = movie_dims if movie_dims is not None else image_masks.shape
        self._image_mean = mean_image
        self._image_correlation = correlation_image
        self._raw_movie_file_location = rawfileloc
        self._roi_ids = roi_idx
        self._roi_locs = roi_locs
        self._sampling_frequency = samp_freq
        self._channel_names = channel_names
        self._rejected_list = rejected_list
        self._accepted_list = accepted_lst

    @property
    def image_dims(self):
        return list(self.image_masks.shape[0:2])

    def get_accepted_list(self):
        if self._accepted_list is None:
            return list(range(self.get_num_rois()))
        else:
            return self._accepted_list

    def get_rejected_list(self):
        if self._rejected_list is None:
            return [a for a in range(self.get_num_rois()) if a not in set(self.get_accepted_list())]
        else:
            return self._rejected_list

    @property
    def roi_locations(self):
        if self._roi_locs is None:
            num_ROIs = self.get_num_rois()
            raw_images = self.image_masks
            roi_location = np.ndarray([2, num_ROIs], dtype='int')
            for i in range(num_ROIs):
                temp = np.where(raw_images[:, :, i] == np.amax(raw_images[:, :, i]))
                roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
            return roi_location
        else:
            return self._roi_locs

    @staticmethod
    def write_segmentation(segmentation_object, save_path):
        raise NotImplementedError

    # defining the abstract class enformed methods:
    def get_roi_ids(self):
        if self._roi_ids is None:
            return list(range(self.get_num_rois()))
        else:
            return self._roi_ids

    def get_image_size(self):
        return self._movie_dims
