import numpy as np
import h5py
from ciextractor import CIExtractor
from nwbwriter import write_recording
import re
import sima


class TraceExtractor(CIExtractor):
    ''' TraceExtractor class:
        input all releveant data and metadata related to the main analysis file parsed by h5py

        Arguments:

        masks:
            description: binary image for each of the regions of interest
            type: np.ndarray (dimensions: # of pixels(d1 X d2 length) x # of ROIs), 2-D

        signal:
            description: fluorescence response of each of the ROI in time
            type: np.ndarray (dimensions: # of ROIs x # timesteps), 2-D

        background_signal:
            description: fluorescence response of each of the background ROIs in time
            type: np.ndarray (dimensions: # of BackgroundRegions x # timesteps), 2-D

        background_masks:
            description: binary image for the background ROIs
            type: np.ndarray (dimensions: # of pixels(d1 X d2) x # of ROIs), 2-D

        summary_image:
            description: mean or the correlation image
            type: np.ndarray (dimensions: d1 x d2)

        roi_idx:
            description: ids of the ROIs
            type: np.ndarray (dimensions: 1 x # of ROIs)

        roi_locs:
            description: x and y location of centroid of ROI mask
            type: np.ndarray (dimensions: # of ROIs x 2)

        samp_freq:
            description: frame rate
            type: np.ndarray (dimensions: 1D)

    '''
    def __init__(self, filepath, algotype, masks=None, signal=None,
                 background_signal=None, background_masks=None,
                 rawfileloc=None, accepted_lst=None,
                 summary_image=None, roi_idx=None,
                 roi_locs=None, samp_freq=None, nback=1,
                 sima_segmentation_label='auto_ROIs'):

        self.filepath = filepath
        self.algotype = algotype
        self.dataset_file, self.group0 = self.file_extractor_io()
        self.sima_segmentation_label = sima_segmentation_label
        if masks is None:
            self.mask_extracter_io()
        elif len(masks.shape) > 2:
            self.image_masks = masks.reshape([masks.shape[2], np.prod(masks.shape[0:1])]).T
        else:
            self.image_masks = masks

        if signal is None:
            self.trace_extracter_io()
        else:
            self.roi_response = signal

        if summary_image is None:
            self.summary_image_io()
        else:
            self.cn = summary_image

        self.tot_exptime_txtractor_io()
        self.file_type_extractor_io()

        if rawfileloc is None:
            self.raw_datafile_io()
        else:
            self.raw_data_file = rawfileloc

        if background_masks is None:
            self.image_masks_bk = np.nan * np.ones([self.image_masks.shape[0], nback])
        else:
            self.image_masks_bk = background_masks

        if background_signal is None:
            self.roi_response_bk = np.nan * np.ones([nback, self.roi_response.shape[1]])
        else:
            self.roi_response_bk = background_signal

        self._roi_ids = roi_idx
        self._roi_locs = roi_locs
        self._no_rois = None
        self._samp_freq = samp_freq
        self._num_of_frames = None
        # remains to mine from mat file or nan it
        self.snr_comp = np.nan * np.ones(self.roi_response.shape)
        # remains to mine from mat file or nan it
        self.r_values = np.nan * np.ones(self.roi_response.shape)
        # remains to mine from mat file or nan it
        self.cnn_preds = np.nan * np.ones(self.roi_response.shape)
        self._rejected_list = []  # remains to mine from mat file or nan it
        self._accepted_list = accepted_lst  # remains to mine from mat file or nan it
        self.idx_components = self.accepted_list  # remains to mine from mat file or nan it
        self.idx_components_bad = self.rejected_list
        self._dims = None
        self._raw_data_file = rawfileloc
        self.file_close()

    def file_close(self):
        self.dataset_file.close()

    def file_extractor_io(self):
        if self.algotype in ['cnmfe', 'extract']:
            f = h5py.File(self.filepath, 'r')
            group0_temp = list(f.keys())
            group0 = [a for a in group0_temp if '#' not in a]
            return f, group0
        elif self.algotype == 'sima':
            _img_dataset = sima.ImagingDataset.load(self.filepath)
            return _img_dataset, None
        else:
            raise Exception('unknown analysis type. Enter ''cnmfe'' /'
                            ' ''extract'' / ''sima''')

    def mask_extracter_io(self):
        if self.algotype == 'cnmfe':
            raw_images = self.dataset_file[self.group0[0]]['extractedImages']
        elif self.algotype == 'extract':
            raw_images = self.dataset_file[self.group0[0]]['filters']
        elif self.algotype == 'sima':
            _sima_rois = self.dataset_file.ROIs
            if len(_sima_rois) > 1:
                if self.sima_segmentation_label in list(_sima_rois.keys()):
                    _sima_rois_data = _sima_rois[self.sima_segmentation_label]
                else:
                    raise Exception('Enter a valid name of ROIs from: {}'.format(
                        ','.join(list(_sima_rois.keys()))))
            elif len(_sima_rois) == 1:
                _sima_rois_data = _sima_rois.values()[0]
                self.sima_segmentation_label = _sima_rois.keys()[0]
            else:
                raise Exception('no ROIs found in the sima file')

            image_masks_ = [roi_dat.mask for roi_dat in _sima_rois_data]
            raw_images = np.array(image_masks_).transpose()
        else:
            raise Exception('unknown analysis type, enter ''cnmfe'' / ''extract'' / ''sima''')

        temp = np.array(raw_images).transpose()
        self.images = temp
        self.image_masks = temp.reshape([np.prod(temp.shape[0:2]), temp.shape[2]], order='F')
        self.extdims = temp.shape[0:2]
        return self.image_masks

    def trace_extracter_io(self):
        if self.algotype == 'cnmfe':
            extracted_signals = self.dataset_file[self.group0[0]]['extracted_signals']
        elif self.algotype == 'extract':
            extracted_signals = self.dataset_file[self.group0[0]]['traces']
        elif self.algotype == 'sima':
            extracted_signals = self.dataset_file.extract(
                rois=self.sima_segmentation_label, save_summary=False)['raw'][0].T
        else:
            raise Exception('unknown analysis type, enter ''cnmfe'' / ''extract'' / ''sima''')

        self.roi_response = np.array(extracted_signals).T
        return self.roi_response

    def tot_exptime_txtractor_io(self):
        if self.algotype in ['cnmfe', 'extract']:
            self.total_time = self.dataset_file[self.group0[0]]['time']['total_time'][0][0]
        elif self.algotype == 'sima':
            self.total_time = None
        return self.total_time

    def file_type_extractor_io(self):
        self.filetype = self.filepath.split('.')[1]
        return self.filetype

    def summary_image_io(self):
        if self.algotype == 'cnmfe':
            summary_images_ = self.dataset_file[self.group0[0]]['cn']
        elif self.algotype == 'extract':
            summary_images_ = self.dataset_file[self.group0[0]]['info']['summary_image']
        elif self.algotype == 'sima':
            summary_images_ = np.squeeze(self.dataset_file.time_averages[0]).T
        else:
            raise Exception('unknown analysis type, enter ''cnmfe'' / ''extract'' / ''sima''')
        self.cn = np.array(summary_images_).T
        return self.cn

    def raw_datafile_io(self):
        if self.algotype == 'cnmfe':
            self.raw_data_file = self.dataset_file[self.group0[0]]['movieList']
        elif self.algotype == 'extract':
            self.raw_data_file = self.dataset_file[self.group0[0]]['file']
        elif self.algotype == 'sima':
            try:
                self.raw_data_file = self.dataset_file.sequences[0]._path
            except AttributeError:
                self.raw_data_file = self.dataset_file.sequences[0]._sequences[0]._path
        else:
            raise Exception('unknown analysis type, enter ''cnmfe'' / ''extract'' / ''sima''')
            self.raw_data_file = None
        return self.raw_data_file

    @property
    def dims(self):
        if self._dims is None:
            self.imagesize = self.extdims
            return list(self.imagesize)
        else:
            self.imagesize = self._dims
            return self.imagesize

    @property
    def no_rois(self):
        if self._no_rois is None:
            raw_images = self.image_masks
            return raw_images.shape[1]
        else:
            return self._no_rois

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
            raw_images = self.image_masks
            roi_location = np.ndarray([2, no_ROIs], dtype='int')
            for i in range(no_ROIs):
                temp = np.where(raw_images[:, :, i] == np.amax(raw_images[:, :, i]))
                roi_location[:, i] = np.array([temp[0][0], temp[1][0]]).T
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
    def nwbwrite(nwbfilename, sourcefilepath, analysis_type, propertydict):
        write_recording(TraceExtractor(sourcefilepath, analysis_type),
                        propertydict, nwbfilename)
        print(f'successfully saved nwb as {nwbfilename}')

    # defining the abstract class enformed methods:
    def get_roi_ids(self):
        return list(range(self.no_rois))

    def get_num_rois(self):
        return self.no_rois

    def get_roi_locations(self, ROI_ids=None):
        if ROI_ids is None:
            return self.roi_locs
        else:
            return self.roi_locs[:, ROI_ids]

    def get_num_frames(self):
        return self.num_of_frames

    def get_sampling_frequency(self):
        return self.samp_freq

    def get_traces(self, ROI_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if ROI_ids is None:  # !!need to make ROI as a 2-d array, specifying the location in image plane
            ROI_ids = range(self.get_roi_ids())
        return self.roi_response[ROI_ids, start_frame:end_frame]

    def get_masks(self, ROI_ids=None):
        if ROI_ids is None:  # !!need to make ROI as a 2-d array, specifying the location in image plane
            ROI_ids = range(self.get_roi_ids())
        return self.image_masks.reshape([*self.dims, *self.no_rois])[:, :, ROI_ids]

    def get_movie_framesize(self):
        return self.dims

    def get_raw_file(self):
        return self.raw_data_file
