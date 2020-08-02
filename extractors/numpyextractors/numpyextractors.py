import numpy as np
from pathlib import Path
from ..segmentationextractor import SegmentationExtractor
from ..imagingextractor import ImagingExtractor
from ..extraction_tools import get_video_shape


class NumpyImagingExtractor(ImagingExtractor):
    def __init__(self, filepath, sampling_frequency=None,
                 channel_names=None):

        ImagingExtractor.__init__(self)
        self.filepath = Path(filepath)
        self._sampling_frequency = sampling_frequency
        assert self.filepath.suffix == '.npy'
        self._video = np.load(self.filepath)
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

    def get_frame(self, frame_idx, channel=0):
        assert frame_idx < self.get_num_frames()
        return self._video[channel, frame_idx]

    def get_frames(self, frame_idxs, channel=0):
        assert np.all(frame_idxs < self.get_num_frames())
        return self._video[channel, frame_idxs]

    # TODO make decorator to check and correct inputs
    def get_video(self, start_frame=None, end_frame=None, channel=0):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        end_frame = min(end_frame, self.get_num_frames())

        video = self._video[channel, start_frame: end_frame]

        return video

    def get_image_size(self):
        return [self._size_x, self._size_y]

    def get_num_frames(self):
        return self._num_frames

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_dtype(self):
        return self._video.dtype

    def get_channel_names(self):
        '''List of  channels in the recoding.

        Returns
        -------
        channel_names: list
            List of strings of channel names
        '''
        return self._channel_names

    def get_num_channels(self):
        '''Total number of active channels in the recording

        Returns
        -------
        no_of_channels: int
            integer count of number of channels
        '''
        return self._num_channels

    @staticmethod
    def write_imaging(imaging, savepath):
        pass


class NumpySegmentationExtractor(SegmentationExtractor):
    '''
    NumpySegmentationExtractor objects are built to contain all data coming from
    a file format for which there is currently no support. To construct this,
    all data must be entered manually as arguments.
    '''

    def __init__(self, filepath=None, masks=None, signal=None,
                 background_signal=None, background_masks=None,
                 rawfileloc=None, accepted_lst=None,
                 summary_image=None, roi_idx=None,
                 roi_locs=None, samp_freq=None, nback=1,
                 total_time=0, rejected_list=None, channel_names=None,
                 no_of_channels=None, movie_dims=None):
        '''
        Parameters:
        ----------
        filepath: str
            The location of the folder containing the custom file format.
        masks: np.ndarray (dimensions: image width x height x # of ROIs)
            Binary image for each of the regions of interest
        signal: np.ndarray (dimensions: # of ROIs x # timesteps)
            Fluorescence response of each of the ROI in time
        background_signal: np.ndarray (dimensions: # of BackgroundRegions x # timesteps)
            Fluorescence response of each of the background ROIs in time
        background_masks: np.ndarray (dimensions: image width x height x # of ROIs)
            Binary image for the background ROIs
        summary_image: np.ndarray (dimensions: d1 x d2)
            Mean or the correlation image
        roi_idx: int list (length is # of ROIs)
            Unique ids of the ROIs if any
        roi_locs: np.ndarray (dimensions: # of ROIs x 2)
            x and y location of centroid of ROI mask
        samp_freq: float
            Frame rate of the movie
        nback: int
            Number of background components extracted
        total_time: float
            total time of the experiment data
        rejected_list: list
            list of ROI ids that are rejected manually or via automated rejection
        channel_names: list
            list of strings representing channel names
        no_of_channels: int
            number of channels
        movie_dims: list(2-D)
            height x width of the movie
        '''
        self.filepath = filepath
        self._dataset_file = None
        if masks is None:
            self.image_masks = np.empty([0, 0])
            self.raw_images = np.empty([0, 0, 0])
        elif len(masks.shape) > 2:
            self.image_masks = masks.reshape([np.prod(masks.shape[0:2]), masks.shape[2]], order='F')
            self.raw_images = masks
        else:
            self.image_masks = masks
            if not movie_dims:
                raise Exception('enter movie dimensions as height x width list')
            self.raw_images = masks.reshape(movie_dims.append(masks.shape[2]), order='F')
        if signal is None:
            self.roi_response = np.empty([0, 0])
        else:
            self.roi_response = signal
        self.cn = summary_image
        self.total_time = total_time
        self.filetype = self._file_type_extractor_read()
        self.raw_movie_file_location = rawfileloc
        self.image_masks_bk = background_masks

        if background_signal is None:
            self.roi_response_bk = np.nan * np.ones([nback, self.roi_response.shape[1]])
        else:
            self.roi_response_bk = background_signal

        self._roi_ids = roi_idx
        self._roi_locs = roi_locs
        self._samp_freq = samp_freq
        self.channel_names = channel_names
        self.no_of_channels = no_of_channels
        self._rejected_list = rejected_list
        self._accepted_list = accepted_lst
        self.idx_components = self.accepted_list
        self.idx_components_bad = self.rejected_list

    def _file_type_extractor_read(self):
        if self.filepath is not None:
            return self.filepath.split('.')[1]
        else:
            return None

    @property
    def image_dims(self):
        return list(self.raw_images.shape[0:2])

    @property
    def no_rois(self):
        return self.raw_images.shape[2]

    @property
    def roi_idx(self):
        if self._roi_ids is None:
            return list(range(self.no_rois))
        else:
            return self._roi_ids

    @property
    def accepted_list(self):
        if self._accepted_list is None:
            return list(range(self.no_rois))
        else:
            return self._accepted_list

    @property
    def rejected_list(self):
        if self._rejected_list is None:
            return [a for a in range(self.no_rois) if a not in set(self.accepted_list)]
        else:
            return self._rejected_list

    @property
    def roi_locs(self):
        if self._roi_locs is None:
            no_ROIs = self.no_rois
            raw_images = self.raw_images
            roi_location = np.ndarray([2, no_ROIs], dtype='int')
            for i in range(no_ROIs):
                temp = np.where(raw_images[:, :, i] == np.amax(raw_images[:, :, i]))
                roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
            return roi_location
        else:
            return self._roi_locs

    @property
    def num_of_frames(self):
        return self.roi_response.shape[1]

    @property
    def samp_freq(self):
        if self._samp_freq is None:
            time = self.total_time
            nframes = self.num_of_frames
            try:
                return nframes / time
            except ZeroDivisionError:
                return 0
        else:
            return self._samp_freq

    @staticmethod
    def write_recording(segmentation_object, savepath):
        raise NotImplementedError

    # defining the abstract class enformed methods:
    def get_roi_ids(self):
        return list(range(self.no_rois))

    def get_num_rois(self):
        return self.no_rois

    def get_roi_locations(self, ROI_ids=None):
        if ROI_ids is None:
            return self.roi_locs
        else:
            ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
            ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
            return self.roi_locs[:, ROI_idx_]

    def get_num_frames(self):
        return self.num_of_frames

    def get_sampling_frequency(self):
        return self.samp_freq

    def get_traces(self, ROI_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames() + 1
        if ROI_ids is None:
            ROI_idx_ = list(range(self.get_num_rois()))
        else:
            ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
            ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
        return self.roi_response[ROI_idx_, start_frame:end_frame]

    def get_image_masks(self, ROI_ids=None):
        if ROI_ids is None:
            ROI_idx_ = range(self.get_num_rois())
        else:
            ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
            ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
        return self.raw_images[:, :, ROI_idx_]

    def get_pixel_masks(self, ROI_ids=None):
        if ROI_ids is None:
            ROI_idx_ = self.roi_idx
        else:
            ROI_idx = [np.where(i == self.roi_idx)[0] for i in ROI_ids]
            ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
        temp = np.empty((1, 4))
        for i, roiid in enumerate(ROI_idx_):
            temp = \
                np.append(temp, self.pixel_masks[self.pixel_masks[:, 3] == roiid, :], axis=0)
        return temp[1::, :]

    def get_movie_framesize(self):
        return self.image_dims

    def get_movie_location(self):
        return self.raw_movie_file_location

    def get_channel_names(self):
        return self.channel_names

    def get_num_channels(self):
        return self.no_of_channels
