import numpy as np
import h5py
from ...segmentationextractor import SegmentationExtractor
from lazy_ops import DatasetView
from ...extraction_tools import _pixel_mask_extractor
import os


class ExtractSegmentationExtractor(SegmentationExtractor):
    """
    This class inherits from the SegmentationExtractor class, having all
    its funtionality specifically applied to the dataset output from
    the \'EXTRACT\' ROI segmentation method.
    """
    extractor_name = 'ExtractSegmentation'
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path):
        """
        Parameters
        ----------
        file_path: str
            The location of the folder containing dataset.mat file.
        """
        SegmentationExtractor.__init__(self)
        self.file_path = file_path
        self._dataset_file, self._group0 = self._file_extractor_read()
        self.image_masks = self._image_mask_extractor_read()
        self._roi_response_raw = self._trace_extractor_read()
        self._raw_movie_file_location = self._raw_datafile_read()
        self._sampling_frequency = self._roi_response_raw.shape[1]/self._tot_exptime_extractor_read()
        self._image_correlation = self._summary_image_read()

    def __del__(self):
        self._dataset_file.close()

    def _file_extractor_read(self):
        f = h5py.File(self.file_path, 'r')
        _group0_temp = list(f.keys())
        _group0 = [a for a in _group0_temp if '#' not in a]
        return f, _group0

    def _image_mask_extractor_read(self):
        return DatasetView(self._dataset_file[self._group0[0]]['filters']).T

    def _trace_extractor_read(self):
        extracted_signals = DatasetView(self._dataset_file[self._group0[0]]['traces'])
        return extracted_signals.T

    def _tot_exptime_extractor_read(self):
        return self._dataset_file[self._group0[0]]['time']['totalTime'][0][0]

    def _summary_image_read(self):
        summary_image = self._dataset_file[self._group0[0]]['info']['summary_image']
        return np.array(summary_image).T

    def _raw_datafile_read(self):
        charlist = [chr(i) for i in self._dataset_file[self._group0[0]]['file'][:]]
        return ''.join(charlist)

    def get_accepted_list(self):
        return list(range(self.get_num_rois()))

    def get_rejected_list(self):
        return [a for a in range(self.get_num_rois()) if a not in set(self.get_accepted_list())]

    @property
    def roi_locations(self):
        num_ROIs = self.get_num_rois()
        raw_images = self.image_masks
        roi_location = np.ndarray([2, num_ROIs], dtype='int')
        for i in range(num_ROIs):
            temp = np.where(raw_images[:, :, i] == np.amax(raw_images[:, :, i]))
            roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
        return roi_location

    @staticmethod
    def write_segmentation(segmentation_object, save_path, plane_num=0):
        if save_path.split('.')[-1] != 'mat':
            raise ValueError('filetype to save must be *.mat')
        filename = os.path.basename(save_path)
        save_path_folder = os.path.join(os.path.dirname(save_path), f'Plane_{plane_num}')
        save_path = os.path.join(save_path_folder, filename)
        if not os.path.exists(save_path_folder):
            os.makedirs(save_path_folder)
        else:
            if os.path.exists(save_path):
                os.remove(save_path)
        with h5py.File(save_path, 'a') as f:
            # create base groups:
            _ = f.create_group('#refs#')
            main = f.create_group('extractAnalysisOutput')
            #create datasets:
            main.create_dataset('filters',data=segmentation_object.get_roi_image_masks().T)
            main.create_dataset('traces', data=segmentation_object.get_traces())
            info = main.create_group('info')
            if segmentation_object.get_images() is not None:
                info.create_dataset('summary_image', data=segmentation_object.get_images())
            main.create_dataset('file', data=[ord(i) for i in segmentation_object.get_movie_location()])
            time = main.create_group('time')
            if segmentation_object.get_sampling_frequency() is not None:
                time.create_dataset('totalTime', (1,1), data=segmentation_object.get_roi_image_masks().shape[1]/
                                                        segmentation_object.get_sampling_frequency())

    # defining the abstract class enformed methods:
    def get_roi_ids(self):
        return list(range(self.get_num_rois()))
    
    def get_image_size(self):
        return self.image_masks.shape[0:2]
