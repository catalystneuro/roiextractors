import numpy as np
import h5py
from ...segmentationextractor import SegmentationExtractor
from lazy_ops import DatasetView
from ...extraction_tools import _pixel_mask_extractor
import os
from scipy.sparse import csc_matrix

class CnmfeSegmentationExtractor(SegmentationExtractor):
    """
    This class inherits from the SegmentationExtractor class, having all
    its funtionality specifically applied to the dataset output from
    the \'CNMF-E\' ROI segmentation method.
    """
    extractor_name = 'CnmfeSegmentation'
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
        return DatasetView(self._dataset_file[self._group0[0]]['extractedImages']).T

    def _trace_extractor_read(self):
        extracted_signals = DatasetView(self._dataset_file[self._group0[0]]['extractedSignals'])
        return extracted_signals.T

    def _tot_exptime_extractor_read(self):
        return self._dataset_file[self._group0[0]]['time']['totalTime'][0][0]

    def _summary_image_read(self):
        summary_image = self._dataset_file[self._group0[0]]['Cn']
        return np.array(summary_image).T

    def _raw_datafile_read(self):
        charlist = [chr(i) for i in self._dataset_file[self._group0[0]]['movieList'][:]]
        return ''.join(charlist)

    def get_accepted_list(self):
        return list(range(self.get_num_rois()))

    def get_rejected_list(self):
        return [a for a in range(self.get_num_rois()) if a not in set(self.get_accepted_list())]

    @property
    def roi_locations(self):
        roi_location = np.ndarray([2, self.get_num_rois()], dtype='int')
        for i in range(self.get_num_rois()):
            temp = np.where(self.image_masks[:, :, i] == np.amax(self.image_masks[:, :, i]))
            roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
        return roi_location

    @staticmethod
    def write_segmentation(segmentation_object, savepath, plane_no=0):
        filename = os.path.basename(savepath)
        savepath_folder = os.path.join(os.path.dirname(savepath),f'Plane_{plane_no}')
        savepath = os.path.join(savepath_folder,filename)
        if not os.path.exists(savepath_folder):
            os.makedirs(savepath_folder)
        else:
            if os.path.exists(savepath):
                os.remove(savepath)
        if savepath.split('.')[-1] != 'mat':
            raise ValueError('filetype to save must be *.mat')
        with h5py.File(savepath, 'a') as f:
            # create base groups:
            _ = f.create_group('#refs#')
            main = f.create_group('cnmfeAnalysisOutput')
            # create datasets:
            main.create_dataset('extractedImages', data=segmentation_object.get_roi_image_masks().T)
            main.create_dataset('extractedSignals', data=segmentation_object.get_traces().T)
            if segmentation_object.get_traces(name='deconvolved') is not None:
                image_mask_csc = csc_matrix(segmentation_object.get_traces(name='deconvolved'))
                main.create_dataset('extractedPeaks/data', data=image_mask_csc.data)
                main.create_dataset('extractedPeaks/ir', data=image_mask_csc.indices)
                main.create_dataset('extractedPeaks/jc', data=image_mask_csc.indptr)
            if segmentation_object.get_images() is not None:
                main.create_dataset('Cn', data=segmentation_object.get_images())
            main.create_dataset('movieList', data=[ord(i) for i in segmentation_object.get_movie_location()])
            inputoptions = main.create_group('inputOptions')
            if segmentation_object.get_sampling_frequency() is not None:
                inputoptions.create_dataset('Fs', data=segmentation_object.get_sampling_frequency())


    # defining the abstract class enformed methods:
    def get_roi_ids(self):
        return list(range(self.get_num_rois()))

    def get_image_size(self):
        return self.image_masks.shape[0:2]
