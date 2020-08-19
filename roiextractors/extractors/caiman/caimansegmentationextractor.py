import numpy as np
import h5py
from lazy_ops import DatasetView
from ...segmentationextractor import SegmentationExtractor
from ...extraction_tools import _pixel_mask_extractor

class CaimanSegmentationExtractor(SegmentationExtractor):
    """
    This class inherits from the SegmentationExtractor class, having all
    its funtionality specifically applied to the dataset output from
    the \'CNMF-E\' ROI segmentation method.
    """
    extractor_name = 'CaimanSegmentation'
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, filepath):
        """
        Parameters
        ----------
        filepath: str
            The location of the folder containing caiman *.hdmf output file.
        """
        SegmentationExtractor.__init__(self)
        self.filepath = filepath
        self._dataset_file = self._file_extractor_read()
        self._roi_response = self._trace_extractor_read('F_dff')
        self._roi_response_fluorescence = self._roi_response,
        self._roi_response_neuropil = self._trace_extractor_read('C'),
        self._roi_response_deconvolved = self._trace_extractor_read('S')
        self._images_mean = self._summary_image_read()
        self._raw_movie_file_location = self._dataset_file['params']['data']['fnames'][0].decode('utf-8')
        self._sampling_frequency = self._dataset_file['params']['data']['fr'].value
        self.image_masks = None

    def __del__(self):
        self._dataset_file.close()

    def _file_extractor_read(self):
        f = h5py.File(self.filepath, 'r')
        return f

    def _image_mask_sparse_read(self):
        roi_ids = self._dataset_file['estimates']['A']['indices']
        masks = self._dataset_file['estimates']['A']['data']
        ids = self._dataset_file['estimates']['A']['indptr']
        return masks, roi_ids, ids

    def _trace_extractor_read(self, field):
        extracted_signals = self._dataset_file['estimates'][field] # lazy read dataset)
        return extracted_signals

    def _summary_image_read(self):
        if self._dataset_file['estimates'].get('Cn'):
            summary_images_ = self._dataset_file['estimates']['Cn']
            return np.array(summary_images_).T
        else:
            return None

    def get_accepted_list(self):
        accepted = self._dataset_file['estimates']['idx_components']
        if len(accepted.shape)==0:
            accepted = list(range(self.no_rois))
        return accepted

    def get_rejected_list(self):
        return [a for a in range(self.no_rois) if a not in set(self.get_accepted_list())]

    @property
    def roi_locations(self):
        _masks, _mask_roi_ids, _mask_ids = self._image_mask_sparse_read()
        roi_location = np.ndarray([2, self.no_rois], dtype='int')
        for i in range(self.no_rois):
            max_mask_roi_id = _mask_roi_ids[_mask_ids[i]+np.argmax(
                _masks[_mask_ids[i]:_mask_ids[i+1]]
            )]
            roi_location[:, i] = [((max_mask_roi_id+1)%(self.image_size[0]+1))-1,#assuming order='F'
                                  ((max_mask_roi_id+1)//(self.image_size[0]+1))]
            if roi_location[0,i]<0:
                roi_location[0,i]=0
        return roi_location

    @staticmethod
    def write_segmentation(segmentation_object, savepath):
        raise NotImplementedError

    # defining the abstract class enformed methods:
    def get_roi_ids(self):
        return list(range(self.no_rois))

    def get_num_rois(self):
        return self._roi_response.shape[0]

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

    def get_traces(self, roi_ids=None, start_frame=None, end_frame=None, name='Fluorescence'):
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.roi_ids)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        traces = [getattr(self, i) for i in self.__dict__.keys() if name.lower() in i]
        if traces:
            return np.array([traces[0][int(i), start_frame:end_frame] for i in roi_idx_])
        else:
            return None

    def get_roi_image_masks(self, roi_ids=None):
        _masks, _mask_roi_ids, _mask_ids = self._image_mask_sparse_read()
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.roi_ids)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        image_mask = np.zeros([np.prod(self.image_size),len(roi_idx_)])
        for j,i in enumerate(roi_idx_):
            roi_ids_loop = _mask_roi_ids[_mask_ids[i]:_mask_ids[i+1]]
            image_mask_loop = _masks[_mask_ids[i]:_mask_ids[i+1]]
            image_mask[[roi_ids_loop],j] = image_mask_loop
        return image_mask.reshape(list(self.image_size)+[len(roi_idx_)],order='F')

    def get_roi_pixel_masks(self, roi_ids=None):

        if roi_ids is None:
            roi_idx_ = self.roi_ids
        else:
            roi_idx = [np.where(np.array(i) == self.roi_ids)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        self.pixel_masks = _pixel_mask_extractor(self.get_roi_image_masks(roi_idx_), range(len(roi_idx_)))
        return self.pixel_masks

    def get_images(self, name='mean'):
        images = [getattr(self, i) for i in self.__dict__.keys() if name.lower() in i]
        if images:
            return images[0]

    def get_image_size(self):
        return self._dataset_file['params']['data']['dims'].value
