import numpy as np
from pathlib import Path
from ...imagingextractor import ImagingExtractor
from ...segmentationextractor import SegmentationExtractor
from ...extraction_tools import check_get_frames_args, get_video_shape, _pixel_mask_extractor


class NumpyImagingExtractor(ImagingExtractor):
    extractor_name = 'NumpyImagingExtractor'
    is_writable = True

    def __init__(self, timeseries, sampling_frequency, channel_names=None):
        ImagingExtractor.__init__(self)

        if isinstance(timeseries, (str, Path)):
            timeseries = Path(timeseries)
            if timeseries.is_file():
                assert timeseries.suffix == '.npy', "'timeseries' file is not a numpy file (.npy)"
                self.is_dumpable = True
                self._video = np.load(timeseries, mmap_mode='r')
                self._kwargs = {'timeseries': str(Path(timeseries).absolute()),
                                'sampling_frequency': sampling_frequency}
            else:
                raise ValueError("'timeeseries' is does not exist")
        elif isinstance(timeseries, np.ndarray):
            self.is_dumpable = False
            self._video = timeseries
            self._kwargs = {'timeseries': timeseries,
                            'sampling_frequency': sampling_frequency}
        else:
            raise TypeError("'timeseries' can be a str or a numpy array")

        self._sampling_frequency = float(sampling_frequency)

        self._sampling_frequency = sampling_frequency
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

    def __init__(self, image_masks, signal, accepted_lst=None,
                 summary_image=None, roi_ids=None,
                 roi_locations=None, sampling_frequency=None,
                 rejected_list=None, channel_names=None,
                 movie_dims=None):
        """
        Parameters:
        ----------
        image_masks: np.ndarray
            Binary image for each of the regions of interest (num_rois x size_x x size_y)
        signal: np.ndarray
            Fluorescence response of each of the ROI in time
        summary_image: np.ndarray
            Mean or the correlation image
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
        if isinstance(image_masks, (str, Path)):
            image_masks = Path(image_masks)
            signal = Path(signal)
            if image_masks.is_file():
                assert image_masks.suffix == '.npy', "'image_masks' file is not a numpy file (.npy)"
                assert signal.suffix == '.npy', "'signal' file is not a numpy file (.npy)"

                self.is_dumpable = True
                self.image_masks = np.load(image_masks, mmap_mode='r')
                self._roi_response = np.load(signal, mmap_mode='r')
                self._kwargs = {'image_masks': str(Path(image_masks).absolute()),
                                'signal': str(Path(signal).absolute())}
            else:
                raise ValueError("'timeeseries' is does not exist")
        elif isinstance(image_masks, np.ndarray):
            assert isinstance(signal, np.ndarray)
            self.is_dumpable = False
            self.image_masks = image_masks
            self._roi_response = signal
            self._kwargs = {'image_masks': image_masks,
                            'signal': signal}
        else:
            raise TypeError("'image_masks' can be a str or a numpy array")
        self._roi_response_dict = {'Fluorescence': self._roi_response}
        self._movie_dims = movie_dims if movie_dims is not None else image_masks.shape
        self._summary_image = summary_image
        if roi_ids is None:
            self._roi_ids = list(np.arange(len(image_masks)))
        else:
            self._roi_ids = roi_ids
        self._roi_locs = roi_locations
        self._sampling_frequency = sampling_frequency
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
        return self._roi_ids

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
        pass

    def get_images(self):
        return {'Images': {'meanImg': self._summary_image}}

    def get_image_size(self):
        return self._movie_dims

