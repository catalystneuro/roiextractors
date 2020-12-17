from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csc_matrix

from ...extraction_tools import PathType
from ...multisegmentationextractor import MultiSegmentationExtractor
from ...segmentationextractor import SegmentationExtractor


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

    def __init__(self, file_path: PathType):
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
        return h5py.File(self.file_path, 'r')

    def _image_mask_sparse_read(self):
        roi_ids = self._dataset_file['estimates']['A']['indices']
        masks = self._dataset_file['estimates']['A']['data']
        ids = self._dataset_file['estimates']['A']['indptr']
        image_mask_in = csc_matrix((masks, roi_ids, ids),
                                   shape=(np.prod(self.get_image_size()), self.get_num_rois())).toarray()
        image_masks = np.reshape(image_mask_in, (*self.get_image_size(), -1), order='F')
        return image_masks

    def _trace_extractor_read(self, field):
        if self._dataset_file['estimates'].get(field):
            return self._dataset_file['estimates'][field]  # lazy read dataset)

    def _summary_image_read(self):
        if self._dataset_file['estimates'].get('Cn'):
            return np.array(self._dataset_file['estimates']['Cn']).T

    def get_accepted_list(self):
        accepted = self._dataset_file['estimates']['idx_components']
        if len(accepted.shape) == 0:
            accepted = list(range(self.get_num_rois()))
        return accepted

    def get_rejected_list(self):
        rejected = self._dataset_file['estimates']['idx_components_bad']
        if len(rejected.shape) > 0:
            return rejected

    @staticmethod
    def write_segmentation(segmentation_object, save_path, overwrite=True):
        save_path = Path(save_path)
        assert save_path.suffix in ['.hdf5', '.h5'], "'save_path' must be a *.hdf5 or *.h5 file"
        if save_path.is_file():
            if not overwrite:
                raise FileExistsError("The specified path exists! Use overwrite=True to overwrite it.")
            else:
                save_path.unlink()

        folder_path = save_path.parent
        file_name = save_path.name
        if isinstance(segmentation_object, MultiSegmentationExtractor):
            segext_objs = segmentation_object.segmentations
            for plane_num, segext_obj in enumerate(segext_objs):
                save_path_plane = folder_path / f'Plane_{plane_num}' / file_name
                CaimanSegmentationExtractor.write_segmentation(segext_obj, save_path_plane)
        if not folder_path.is_dir():
            folder_path.mkdir(parents=True)

        with h5py.File(save_path, 'a') as f:
            # create base groups:
            estimates = f.create_group('estimates')
            params = f.create_group('params')
            # adding to estimates:
            if segmentation_object.get_traces(name='neuropil') is not None:
                estimates.create_dataset('C', data=segmentation_object.get_traces(name='neuropil'))
            if segmentation_object.get_traces(name='dff') is not None:
                estimates.create_dataset('F_dff', data=segmentation_object.get_traces(name='dff'))
            if segmentation_object.get_traces(name='deconvolved') is not None:
                estimates.create_dataset('S', data=segmentation_object.get_traces(name='deconvolved'))
            if segmentation_object.get_image('correlation') is not None:
                estimates.create_dataset('Cn', data=segmentation_object.get_image('correlation'))
            estimates.create_dataset('idx_components', data=np.array([] if segmentation_object.get_accepted_list() is None
                                                                         else segmentation_object.get_accepted_list()))
            estimates.create_dataset('idx_components_bad', data=np.array([] if segmentation_object.get_rejected_list() is None
                                                                         else segmentation_object.get_rejected_list()))

            # adding image_masks:
            image_mask_data = np.reshape(segmentation_object.get_roi_image_masks(),
                                         [-1, segmentation_object.get_num_rois()], order='F')
            image_mask_csc = csc_matrix(image_mask_data)
            estimates.create_dataset('A/data', data=image_mask_csc.data)
            estimates.create_dataset('A/indptr', data=image_mask_csc.indptr)
            estimates.create_dataset('A/indices', data=image_mask_csc.indices)
            estimates.create_dataset('A/shape', data=image_mask_csc.shape)

            # adding params:
            params.create_dataset('data/fr', data=segmentation_object.get_sampling_frequency())
            params.create_dataset('data/dims', data=segmentation_object.get_image_size())
            f.create_dataset('dims', data=segmentation_object.get_image_size())

    # defining the abstract class enformed methods:
    def get_roi_ids(self):
        return list(range(self.get_num_rois()))

    def get_image_size(self):
        return self._dataset_file['params']['data']['dims'][()]
