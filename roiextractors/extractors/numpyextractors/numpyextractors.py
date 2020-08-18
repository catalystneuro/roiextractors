import numpy as np
from pathlib import Path
from ...segmentationextractor import SegmentationExtractor
from ...imagingextractor import ImagingExtractor
from ...extraction_tools import ArrayType, PathType, check_get_frames_args, check_get_videos_args, get_video_shape



# TODO this class should also be able to instantiate an in-memory object (useful for testing)
class NumpyImagingExtractor(ImagingExtractor):
    def __init__(self, file_path, sampling_frequency=None,
                 channel_names=None):

        ImagingExtractor.__init__(self)
        self.filepath = Path(file_path)
        self._sampling_frequency = sampling_frequency
        assert self.filepath.suffix == '.npy'
        self._video = np.load(self.filepath, mmap_mode='r')
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

    @check_get_videos_args
    def get_video(self, start_frame=None, end_frame=None, channel=0):
        video = self._video[channel, start_frame:end_frame]
        return video

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

    def __init__(self, filepath=None, image_masks=None,
                 pixel_masks=None, signal=None,
                 rawfileloc=None, accepted_lst=None,
                 summary_image=None, roi_idx=None,
                 roi_locs=None, samp_freq=None,
                 rejected_list=None, channel_names=None,
                 movie_dims=None):
        """
        Parameters:
        ----------
        filepath: str
            The location of the folder containing the custom file format.
        image_masks: np.ndarray (dimensions: image width x height x # of ROIs)
            Binary image for each of the regions of interest
        pixel_masks: list
            list of np.ndarray (dimensions: no_pixels X 2) pixel mask for every ROI
        signal: np.ndarray (dimensions: # of ROIs x # timesteps)
            Fluorescence response of each of the ROI in time
        summary_image: np.ndarray (dimensions: d1 x d2)
            Mean or the correlation image
        roi_idx: int list (length is # of ROIs)
            Unique ids of the ROIs if any
        roi_locs: np.ndarray (dimensions: # of ROIs x 2)
            x and y location representative of ROI mask
        samp_freq: float
            Frame rate of the movie
        rejected_list: list
            list of ROI ids that are rejected manually or via automated rejection
        channel_names: list
            list of strings representing channel names
        movie_dims: list(2-D)
            height x width of the movie
        """
        SegmentationExtractor.__init__(self)
        self.filepath = filepath
        if image_masks is None:
            self.image_masks = np.empty([0, 0, 0])
        if pixel_masks is None:
            self.pixel_masks = pixel_masks
        if signal is None:#initialize with a empty value
            self._roi_response = np.empty([0, 0])
        else:
            self._roi_response = signal
        self._roi_response_dict = {'Fluorescence': self._roi_response}
        self._movie_dims = movie_dims
        self._summary_image = summary_image
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
            return list(range(self.no_rois))
        else:
            return self._accepted_list

    def get_rejected_list(self):
        if self._rejected_list is None:
            return [a for a in range(self.no_rois) if a not in set(self.get_accepted_list())]
        else:
            return self._rejected_list

    @property
    def roi_locations(self):
        if self._roi_locs is None:
            no_ROIs = self.no_rois
            raw_images = self.image_masks
            roi_location = np.ndarray([2, no_ROIs], dtype='int')
            for i in range(no_ROIs):
                temp = np.where(raw_images[:, :, i] == np.amax(raw_images[:, :, i]))
                roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
            return roi_location
        else:
            return self._roi_locs

    @staticmethod
    def write_segmentation(segmentation_object, savepath):
        raise NotImplementedError

    # defining the abstract class enformed methods:
    def get_roi_ids(self):
        if self._roi_ids is None:
            return list(range(self.no_rois))
        else:
            return self._roi_ids

    def get_num_rois(self):
        return self.image_masks.shape[2]

    def get_roi_locations(self, roi_ids=None):
        if roi_ids is None:
            return self.roi_locations
        else:
            roi_idx = [np.where(np.array(i) == self.roi_ids)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
            return self.roi_locations[:, roi_idx_]

    def get_num_frames(self):
        return self._roi_response.shape[1]

    def get_traces(self, roi_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames() + 1
        if roi_ids is None:
            roi_idx_ = list(range(self.get_num_rois()))
        else:
            roi_idx = [np.where(np.array(i) == self.roi_ids)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return self._roi_response[roi_idx_, start_frame:end_frame]

    def get_roi_image_masks(self, roi_ids=None):
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.roi_ids)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return self.image_masks[:, :, roi_idx_]

    def get_roi_pixel_masks(self, roi_ids=None):
        if roi_ids is None:
            roi_idx_ = self.roi_ids
        else:
            roi_idx = [np.where(i == self.roi_ids)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        temp = np.empty((1, 4))
        for i, roiid in enumerate(roi_idx_):
            temp = \
                np.append(temp, self.pixel_masks[self.pixel_masks[:, 3] == roiid, :], axis=0)
        return temp[1::, :]

    def get_images(self):
        return {'Images': {'meanImg': self._summary_image}}

    def get_image_size(self):
        return self._movie_dims


