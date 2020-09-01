import numpy as np
import h5py
from ...segmentationextractor import SegmentationExtractor
from lazy_ops import DatasetView
from ...extraction_tools import _pixel_mask_extractor

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

    def __init__(self, filepath):
        """
        Parameters
        ----------
        filepath: str
            The location of the folder containing dataset.mat file.
        """
        SegmentationExtractor.__init__(self)
        self.filepath = filepath
        self._dataset_file, self._group0 = self._file_extractor_read()
        self.image_masks = self._image_mask_extractor_read()
        self._roi_response = self._trace_extractor_read()
        self._roi_response_dict = {'Fluorescence': self._roi_response}
        self._total_time = self._tot_exptime_extractor_read()
        self._raw_movie_file_location = self._raw_datafile_read()
        self._sampling_frequency = self._roi_response.shape[1]/self._total_time
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
        return DatasetView(self._dataset_file[self._group0[0]]['filters']).T

    def _trace_extractor_read(self):
        extracted_signals = DatasetView(self._dataset_file[self._group0[0]]['traces'])
        return extracted_signals.T

    def _tot_exptime_extractor_read(self):
        return self._dataset_file[self._group0[0]]['time']['totalTime'][0][0]

    def _summary_image_read(self):
        summary_images_ = self._dataset_file[self._group0[0]]['info']['summary_image']
        return np.array(summary_images_).T

    def _raw_datafile_read(self):
        charlist = [chr(i) for i in self._dataset_file[self._group0[0]]['file'][:]]
        return ''.join(charlist)

    def get_accepted_list(self):
        return list(range(self.no_rois))

    def get_rejected_list(self):
        return [a for a in range(self.no_rois) if a not in set(self.get_accepted_list())]

    @property
    def roi_locations(self):
        no_ROIs = self.no_rois
        raw_images = self.image_masks
        roi_location = np.ndarray([2, no_ROIs], dtype='int')
        for i in range(no_ROIs):
            temp = np.where(raw_images[:, :, i] == np.amax(raw_images[:, :, i]))
            roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
        return roi_location

    @staticmethod
    def write_segmentation(segmentation_object, savepath):
        raise NotImplementedError

    # defining the abstract class enformed methods:
    def get_roi_ids(self):
        return list(range(self.no_rois))

    def get_images(self):
        return {'Images': {'meanImg': self._summary_image_read()}}
    
    def get_image_size(self):
        return self.image_masks.shape[0:2]
