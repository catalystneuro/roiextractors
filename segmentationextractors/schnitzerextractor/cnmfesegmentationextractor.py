import numpy as np
import h5py
from ..segmentationextractor import SegmentationExtractor
from lazy_ops import DatasetView


class CnmfeSegmentationExtractor(SegmentationExtractor):
    '''
    This class inherits from the SegmentationExtractor class, having all
    its funtionality specifically applied to the dataset output from
    the \'CNMF-E\' ROI segmentation method.
    '''

    def __init__(self, filepath):
        '''
        Parameters
        ----------
        filepath: str
            The location of the folder containing dataset.mat file.
        '''
        self.filepath = filepath
        self._dataset_file, self._group0 = self._file_extractor_read()
        self.extimage_dims, self.raw_images = self._image_mask_extractor_read()
        self.image_masks = self.raw_images
        self.roi_response = self._trace_extractor_read()
        self._roi_ids = None
        self.pixel_masks = self._pixel_mask_extractor_read()
        self.cn = self._summary_image_read()
        self.total_time = self._tot_exptime_extractor_read()
        self.filetype = self._file_type_extractor_read()
        self.raw_movie_file_location = self._raw_datafile_read()
        # Not found data:
        self.channel_names = ['OpticalChannel']
        self.no_of_channels = 1
        self._no_background_comps = 1
        self._roi_locs = None
        self._samp_freq = None
        self._num_of_frames = None
        self.snr_comp = np.nan * np.ones(self.roi_response.shape)
        self.r_values = np.nan * np.ones(self.roi_response.shape)
        self.cnn_preds = np.nan * np.ones(self.roi_response.shape)
        self._rejected_list = []
        self._accepted_list = None
        self.idx_components = self.accepted_list
        self.idx_components_bad = self.rejected_list
        self.image_masks_bk = np.nan * \
            np.ones(list(self.raw_images.shape[0:2]) + [self._no_background_comps])
        self.roi_response_bk = np.nan * \
            np.ones([self._no_background_comps, self.roi_response.shape[1]])
        # file close:
        # self._file_close()

    def _file_close(self):
        self._dataset_file.close()

    def _file_extractor_read(self):
        f = h5py.File(self.filepath, 'r')
        _group0_temp = list(f.keys())
        _group0 = [a for a in _group0_temp if '#' not in a]
        return f, _group0

    def _image_mask_extractor_read(self):
        _raw_images_trans = DatasetView(self._dataset_file[self._group0[0]]['extractedImages']).T
        return _raw_images_trans.shape[0:2], _raw_images_trans

    def _pixel_mask_extractor_read(self):
        return super()._pixel_mask_extractor(self.raw_images, self.roi_idx)

    def _trace_extractor_read(self):
        extracted_signals = DatasetView(self._dataset_file[self._group0[0]]['extractedSignals'])
        return extracted_signals.T

    def _tot_exptime_extractor_read(self):
        return self._dataset_file[self._group0[0]]['time']['totalTime'][0][0]

    def _file_type_extractor_read(self):
        return self.filepath.split('.')[1]

    def _summary_image_read(self):
        summary_images_ = self._dataset_file[self._group0[0]]['Cn']
        return np.array(summary_images_).T

    def _raw_datafile_read(self):
        charlist = [chr(i) for i in self._dataset_file[self._group0[0]]['movieList'][:]]
        return ''.join(charlist)

    # defining abstract enforced properties:
    @property
    def image_dims(self):
        return list(self.extimage_dims)

    @property
    def no_rois(self):
        return self.roi_response.shape[0]

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
        return [a for a in range(self.no_rois) if a not in set(self.accepted_list)]

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
        if self._num_of_frames is None:
            extracted_signals = self.roi_response
            return extracted_signals.shape[1]
        else:
            return self._num_of_frames

    @property
    def samp_freq(self):
        if self._samp_freq is None:
            time = self.total_time
            nframes = self.num_of_frames
            return nframes / time
        else:
            return self._samp_freq

    @staticmethod
    def write_recording(segmentation_object, savepath):
        raise NotImplementedError

    # defining the abstract class enformed methods:
    def get_roi_ids(self):
        return self.roi_idx

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
            ROI_idx_ = range(self.get_num_rois())
        else:
            ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
            ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
        return np.array([self.roi_response[int(i), start_frame:end_frame] for i in ROI_idx_])

    def get_image_masks(self, ROI_ids=None):
        if ROI_ids is None:
            ROI_idx_ = range(self.get_num_rois())
        else:
            ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
            ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
        return np.array([self.raw_images[:, :, int(i)].T for i in ROI_idx_]).T

    def get_pixel_masks(self, ROI_ids=None):
        if ROI_ids is None:
            ROI_idx_ = self.roi_idx
        else:
            ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
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
