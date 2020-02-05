import numpy as np
import h5py
from ..segmentationextractor import SegmentationExtractor


class ExtractSegmentationExtractor(SegmentationExtractor):
    '''
    This class inherits from the SegmentationExtractor class, having all
    its funtionality specifically applied to the dataset output from
    the \'EXTRACT\' ROI segmentation method.
    '''

    def __init__(self, filepath):
        self.filepath = filepath
        self._dataset_file, self._group0 = self._file_extractor_read()
        self.image_masks, self.extimage_dims, self.raw_images =\
            self._image_mask_extractor_read()
        self.roi_response = self._trace_extractor_read()
        self._roi_ids = None
        self.pixel_masks = self._pixel_mask_extractor_read()
        self.cn = self._summary_image_read()
        self.total_time = self._tot_exptime_extractor_read()
        self.filetype = self._file_type_extractor_read()
        self.raw_movie_file_location = self._raw_datafile_read()
        # Not found data:
        self.channel_names = None
        self.no_of_channels = 1
        self._no_background_comps = 1
        self.snr_comp = np.nan * np.ones(self.roi_response.shape)
        self.r_values = np.nan * np.ones(self.roi_response.shape)
        self.cnn_preds = np.nan * np.ones(self.roi_response.shape)
        self._rejected_list = []
        self._accepted_list = None
        self.idx_components = self.accepted_list
        self.idx_components_bad = self.rejected_list
        self.image_masks_bk = np.nan * np.ones([self.image_masks.shape[0], self._no_background_comps])
        self.roi_response_bk = np.nan * np.ones([self._no_background_comps, self.roi_response.shape[1]])
        # file close:
        self._file_close()

    def _file_close(self):
        self._dataset_file.close()

    def _file_extractor_read(self):
        f = h5py.File(self.filepath, 'r')
        _group0_temp = list(f.keys())
        _group0 = [a for a in _group0_temp if '#' not in a]
        return f, _group0

    def _image_mask_extractor_read(self):
        _raw_images_trans = np.array(self._dataset_file[self._group0[0]]['filters']).T
        return _raw_images_trans.reshape(
                        [np.prod(_raw_images_trans.shape[0:2]),
                            _raw_images_trans.shape[2]],
                         order='F'),\
               _raw_images_trans.shape[0:2],\
               _raw_images_trans

    def _pixel_mask_extractor_read(self):
        return super()._pixel_mask_extractor(self.raw_images, self.roi_idx)

    def _trace_extractor_read(self):
        extracted_signals = self._dataset_file[self._group0[0]]['traces']
        return np.array(extracted_signals).T

    def _tot_exptime_extractor_read(self):
        return self._dataset_file[self._group0[0]]['time']['totalTime'][0][0]

    def _file_type_extractor_read(self):
        return self.filepath.split('.')[1]

    def _summary_image_read(self):
        summary_images_ = self._dataset_file[self._group0[0]]['info']['summary_image']
        return np.array(summary_images_).T

    def _raw_datafile_read(self):
        charlist = [chr(i) for i in self._dataset_file[self._group0[0]]['file'][:]]
        return ''.join(charlist)

    @property
    def image_dims(self):
        '''
        Returns
        -------
        image_dims: list
            The width X height of the image.
        '''
        return list(self.extimage_dims)

    @property
    def no_rois(self):
        '''
        The number of Independent sources(neurons) indentified after the
        segmentation operation. The regions of interest for which fluorescence
        traces will be extracted downstream.

        Returns
        -------
        no_rois: int
            The number of rois
        '''
        return self.roi_response.shape[0]

    @property
    def roi_idx(self):
        '''
        Integer label given to each region of interest (neuron).

        Returns
        -------
        roi_idx: list
            list of integers of the ROIs. Listed in the order in which the ROIs
            occur in the image_masks (2nd dimention)
        '''
        if self._roi_ids is None:
            return list(range(self.no_rois))
        else:
            return self._roi_ids

    @property
    def accepted_list(self):
        '''
        The ids of the ROIs which are accepted after manual verification of
        ROIs.

        Returns
        -------
        accepted_list: list
            List of accepted ROIs
        '''
        return list(range(self.no_rois))

    @property
    def rejected_list(self):
        '''
        The ids of the ROIs which are rejected after manual verification of
        ROIs.

        Returns
        -------
        accepted_list: list
            List of rejected ROIs
        '''
        return [a for a in range(self.no_rois) if a not in set(self.accepted_list)]

    @property
    def roi_locs(self):
        '''
        The x and y pixel location of the ROIs. The location where the pixel
        value is maximum in the image mask.

        Returns
        -------
        roi_locs: np.array
            Array with the first column representing the x (width) and second representing
            the y (height) coordinates of the ROI.
        '''
        no_ROIs = self.no_rois
        raw_images = self.raw_images
        roi_location = np.ndarray([2, no_ROIs], dtype='int')
        for i in range(no_ROIs):
            temp = np.where(raw_images[:, :, i] == np.amax(raw_images[:, :, i]))
            roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
        return roi_location

    @property
    def num_of_frames(self):
        '''
        Total number of images in the image sequence across time.

        Returns
        -------
        num_of_frames: int
            Same as the -1 dimention of the dF/F trace(roi_response).
        '''
        extracted_signals = self.roi_response
        return extracted_signals.shape[1]

    @property
    def samp_freq(self):
        '''
        Returns
        -------
        samp_freq: int
            Sampling frequency of the dF/F trace.
        '''
        time = self.total_time
        nframes = self.num_of_frames
        return nframes / time

    @staticmethod
    def write_recording(segmentation_object, savepath):
        '''
        Static method to write recording back to the native format.

        Parameters
        ----------
        segmentation_object: SegmentationExtracteor object
            The EXTRACT segmentation object from which an EXTRACT native format
            file has to be generated.
        savepath: str
            path to save the native format.
        '''
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
            ROI_idx = [np.where(i==self.roi_idx)[0] for i in ROI_ids]
            return self.roi_locs[:, ROI_idx]

    def get_num_frames(self):
        return self.num_of_frames

    def get_sampling_frequency(self):
        return self.samp_freq

    def get_traces(self, ROI_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if ROI_ids is None:
            ROI_idx = range(self.get_num_rois())
        else:
            ROI_idx = [np.where(i==self.roi_idx)[0] for i in ROI_ids]
        return self.roi_response[ROI_idx, start_frame:end_frame]

    def get_image_masks(self, ROI_ids=None):
        if ROI_ids is None:
            ROI_idx = self.roi_idx
        else:
            ROI_idx = [np.where(i==self.roi_idx)[0] for i in ROI_ids]
        return self.image_masks.reshape(self.image_dims + [self.no_rois], order='F')[:, :, ROI_idx]

    def get_pixel_masks(self, ROI_ids=None):
        if ROI_ids is None:
            ROI_idx = range(self.get_num_rois())
        else:
            ROI_idx = [np.where(i==self.roi_idx)[0] for i in ROI_ids]
        temp = np.empty((1, 4))
        for i, roiid in enumerate(ROI_idx):
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
