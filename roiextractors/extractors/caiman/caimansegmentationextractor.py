import numpy as np
import h5py
from lazy_ops import DatasetView
from ...segmentationextractor import SegmentationExtractor
from ...extraction_tools import _pixel_mask_extractor
import os
from scipy.sparse import csc_matrix

class CaimanSegmentationExtractor(SegmentationExtractor):
    """
    This class inherits from the SegmentationExtractor class, having all
    its funtionality specifically applied to the dataset output from
    the \'CNMF-E\' ROI segmentation method.
    """
    extractor_name = 'CaimanSegmentation'
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path):
        """
        Parameters
        ----------
        file_path: str
            The location of the folder containing caiman *.hdmf output file.
        """
        SegmentationExtractor.__init__(self)
        self.file_path = file_path
        self._dataset_file = self._file_extractor_read()
        self._roi_response_dff = self._trace_extractor_read('F_dff')
        self._roi_response_neuropil = self._trace_extractor_read('C')
        self._roi_response_deconvolved = self._trace_extractor_read('S')
        self._image_correlation = self._summary_image_read()
        self._sampling_frequency = self._dataset_file['params']['data']['fr'][()]
        self.image_masks = self._image_mask_sparse_read()

    def __del__(self):
        self._dataset_file.close()

    def _file_extractor_read(self):
        f = h5py.File(self.file_path, 'r')
        return f

    def _image_mask_sparse_read(self):
        roi_ids = self._dataset_file['estimates']['A']['indices']
        masks = self._dataset_file['estimates']['A']['data']
        ids = self._dataset_file['estimates']['A']['indptr']
        image_mask_in = csc_matrix((masks, roi_ids, ids), shape=(np.prod(self.get_image_size()), self.no_rois)).toarray()
        image_masks = np.reshape(image_mask_in, (*self.get_image_size(), -1), order='F')
        return image_masks

    def _trace_extractor_read(self, field):
        if self._dataset_file['estimates'].get(field):
            return self._dataset_file['estimates'][field] # lazy read dataset)

    def _summary_image_read(self):
        if self._dataset_file['estimates'].get('Cn'):
            return np.array(self._dataset_file['estimates']['Cn']).T

    def get_accepted_list(self):
        accepted = self._dataset_file['estimates']['idx_components']
        if len(accepted.shape)==0:
            accepted = list(range(self.get_num_rois()))
        return accepted

    def get_rejected_list(self):
        rejected = self._dataset_file['estimates']['idx_components_bad']
        if len(rejected.shape) > 0:
            return rejected

    @property
    def roi_locations(self):
        num_ROIs = self.get_num_rois()
        roi_locations = np.ndarray([2, num_ROIs], dtype='int')
        for i in range(num_ROIs):
            temp = np.where(self.image_masks[:, :, i] == np.amax(self.image_masks[:, :, i]))
            roi_locations[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
        return roi_locations

    @staticmethod
    def write_segmentation(segmentation_object, save_path, plane_num=0):
        if save_path.split('.')[-1]!='hdf5':
            raise ValueError('filetype to save must be *.hdf5')
        filename = os.path.basename(save_path)
        save_path_folder = os.path.join(os.path.dirname(save_path), f'Plane_{plane_num}')
        save_path = os.path.join(save_path_folder, filename)
        if not os.path.exists(save_path_folder):
            os.makedirs(save_path_folder)
        else:
            if os.path.exists(save_path):
                os.remove(save_path)
        with h5py.File(save_path,'a') as f:
            #create base groups:
            estimates = f.create_group('estimates')
            params = f.create_group('params')
            #adding to estimates:
            if segmentation_object._roi_response_neuropil is not None:
                estimates.create_dataset('C',data=segmentation_object._roi_response_neuropil)
            estimates.create_dataset('F_dff', data=segmentation_object._roi_response_fluorescence)
            if segmentation_object._roi_response_deconvolved is not None:
                estimates.create_dataset('S', data=segmentation_object._roi_response_deconvolved)
            if segmentation_object._image_correlation is not None:
                estimates.create_dataset('Cn', data=segmentation_object._images_correlation)
            estimates.create_dataset('idx_components', data=np.array(segmentation_object.get_accepted_list()))
            estimates.create_dataset('idx_components_bad', data=np.array(segmentation_object.get_rejected_list()))

            #adding image_masks:
            image_mask_data = np.reshape(segmentation_object.get_roi_image_masks(),[-1,segmentation_object.get_num_rois()],order='F')
            image_mask_csc = csc_matrix(image_mask_data)
            estimates.create_dataset('A/data',data=image_mask_csc.data)
            estimates.create_dataset('A/indptr', data=image_mask_csc.indptr)
            estimates.create_dataset('A/indices', data=image_mask_csc.indices)
            estimates.create_dataset('A/shape', data=image_mask_csc.shape)

            #adding params:
            params.create_dataset('data/fr',data=segmentation_object._sampling_frequency)
            params.create_dataset('data/dims', data=segmentation_object.get_image_size())
            f.create_dataset('dims',data=segmentation_object.get_image_size())

    # defining the abstract class enformed methods:
    def get_roi_ids(self):
        return list(range(self.get_num_rois()))

    def get_image_size(self):
        return self._dataset_file['params']['data']['dims'][()]
