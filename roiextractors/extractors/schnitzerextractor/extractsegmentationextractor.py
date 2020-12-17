from pathlib import Path

import h5py
import numpy as np
from lazy_ops import DatasetView

from ...extraction_tools import PathType
from ...multisegmentationextractor import MultiSegmentationExtractor
from ...segmentationextractor import SegmentationExtractor


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

    def __init__(self, file_path: PathType):
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
        self._sampling_frequency = self._roi_response_raw.shape[1] / self._tot_exptime_extractor_read()
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
        if self._dataset_file[self._group0[0]].get('file'):
            charlist = [chr(i) for i in np.squeeze(self._dataset_file[self._group0[0]]['file'][:])]
            return ''.join(charlist)

    def get_accepted_list(self):
        return list(range(self.get_num_rois()))

    def get_rejected_list(self):
        ac_set = set(self.get_accepted_list())
        return [a for a in range(self.get_num_rois()) if a not in ac_set]

    @staticmethod
    def write_segmentation(segmentation_object: SegmentationExtractor, save_path, overwrite=True):
        save_path = Path(save_path)
        assert save_path.suffix == '.mat', "'save_path' must be a *.mat file"
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
                ExtractSegmentationExtractor.write_segmentation(segext_obj, save_path_plane)
        if not folder_path.is_dir():
            folder_path.mkdir(parents=True)

        with h5py.File(save_path, 'a') as f:
            # create base groups:
            _ = f.create_group('#refs#')
            main = f.create_group('extractAnalysisOutput')
            # create datasets:
            main.create_dataset('filters', data=segmentation_object.get_roi_image_masks().T)
            main.create_dataset('traces', data=segmentation_object.get_traces())
            if getattr(segmentation_object,'_raw_movie_file_location', None):
                main.create_dataset('file', data=[ord(alph) for alph in str(segmentation_object._raw_movie_file_location)])
            info = main.create_group('info')
            if segmentation_object.get_image() is not None:
                info.create_dataset('summary_image', data=segmentation_object.get_image())
            time = main.create_group('time')
            if segmentation_object.get_sampling_frequency() is not None:
                time.create_dataset('totalTime', (1, 1), data=segmentation_object.get_roi_image_masks().shape[1] /
                                                              segmentation_object.get_sampling_frequency())

    # defining the abstract class informed methods:
    def get_roi_ids(self):
        return list(range(self.get_num_rois()))

    def get_image_size(self):
        return self.image_masks.shape[0:2]
