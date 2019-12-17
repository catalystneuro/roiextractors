
''' data container class:
    input all releveant data and metadata related to the main analysis file parsed by h5py
    masks:
        type: np.ndarray (dimensions: # of pixels(d1 X d2) x # of ROIs)

    signal:
        type: np.ndarray (dimensions: # of ROIs x # components), 3D

    background_signal:
        type: np.ndarray (dimensions: # of BackgroundRegions x # components), 3D

    background_masks:
        type: np.ndarray (dimensions: # of pixels(d1 X d2) x # of ROIs)

    trace_residuals:
        type: np.ndarray (dimensions: # of ROIs x # of timesteps)

    deconvolved_activity:
        type: np.ndarray (dimensions: # of ROIs x # of timesteps)

    Correlation:
        type: np.ndarray (dimensions: # of pixels(d1 X d2) x # of ROIs)

    ROIids:
        type: np.ndarray (dimensions: 1 x # of ROIs)

    no_rois:
        type: np.ndarray (dimensions: 1D)

    roi_locs:
        type: np.ndarray (dimensions: # of ROIs x 2)

    samp_freq:
        type: np.ndarray (dimensions: 1D)

    num_of_frames:
        type: np.ndarray (dimensions: 1D)

    MovieFramesize:
        type: np.ndarray (dimensions: 2D)

'''
import numpy as np
import h5py
from ciextractor import CIExtractor
from nwbwriter import write_recording

class TraceExtractor(CIExtractor):

    def __init__(self,filepath,algotype,masks=[],signal=[],background_signal=[],background_masks=[],trace_residuals=[],rawfileloc=None,
                deconvolved_activity=[],accepted_lst=None,correlation_image=[],roi_idx=[],roi_locs=None,samp_freq=None,nback=1):
        self.filepath=filepath
        self.algotype=algotype
        self.rf,self.group0=self.file_extractor_io()
        if len(masks) is 0:
            self.mask_extracter_io()
        elif len(masks.shape)>2:
            self.masks=temp.reshape([masks.shape[2],np.prod(masks.shape[0:1])]).T
        else:
            self.masks=masks

        if len(signal) is 0:
            self.trace_extracter_io()
        else:
            self.roi_response=signal

        if len(correlation_image) is 0:
            self.summary_image_io()
        else:
            self.cn=correlation_image

        self.tot_exptime_txtractor_io()
        self.file_type_extractor_io()

        if rawfileloc is None:
            self.raw_datafile_io()
        else:
             self.raw_data_file=rawfileloc

        if len(background_masks) is 0:
            self.masks_bk=np.nan*np.ones([self.masks.shape[0],nback])
        else:
            self.masks_bk=background_masks

        if len(background_signal) is 0:
            self.roi_response_bk=np.nan*np.ones([nback,self.roi_response.shape[1]])
        else:
            self.roi_response_bk=background_signal

        if len(trace_residuals) is 0:
            self.roi_response_residual=np.nan*np.ones(self.roi_response.shape)
        else:
            self.roi_response_residual=trace_residuals

         #remains to mine from mat file or nan it
         # need to resolve the diff between correlation image and summary image
        self._roi_ids=roi_idx
        self._roi_locs=roi_locs
        self._no_rois=None
        self._samp_freq=samp_freq
        self._num_of_frames=None
        self.snr_comp=np.nan*np.ones(self.roi_response.shape)    #remains to mine from mat file or nan it
        self.r_values=np.nan*np.ones(self.roi_response.shape)     #remains to mine from mat file or nan it
        self.cnn_preds=np.nan*np.ones(self.roi_response.shape)    #remains to mine from mat file or nan it
        self._rejected_list=[] #remains to mine from mat file or nan it
        self._accepted_list=accepted_lst #remains to mine from mat file or nan it
        self.idx_components=self.accepted_list #remains to mine from mat file or nan it
        self.idx_components_bad=self.rejected_list
        self._dims=None
        self._raw_data_file=rawfileloc
        self.file_close()

    def file_close(self):
        self.rf.close()

    def file_extractor_io(self):
        f=h5py.File(self.filepath,'r')
        group0_temp=list(f.keys())
        group0=[a for a in group0_temp if '#' not in a]
        return f,group0

    def mask_extracter_io(self):
        if self.algotype=='cnmfe':
            raw_images=self.rf[self.group0[0]]['extractedImages']
        elif self.algotype=='extract':
            raw_images=self.rf[self.group0[0]]['filters']
        else:
            raise Exception('unknown analysis type, enter ''cnmfe'' or ''extract')
        temp=np.array(raw_images).transpose()
        self.images=temp
        self.masks=temp.reshape([np.prod(temp.shape[0:2]),temp.shape[2]])
        self.extdims=temp.shape[0:2]
        return self.masks

    def trace_extracter_io(self):
        if self.algotype=='cnmfe':
            extracted_signals=self.rf[self.group0[0]]['extracted_signals']
        elif self.algotype=='extract':
            extracted_signals=self.rf[self.group0[0]]['traces']
        else:
            raise Exception('unknown analysis type, enter ''cnmfe'' or ''extract')
        self.roi_response=np.array(extracted_signals).T
        return self.roi_response

    def tot_exptime_txtractor_io(self):
        self.total_time=self.rf[self.group0[0]]['time']['total_time'][0][0]
        return self.total_time

    def file_type_extractor_io(self):
        self.filetype=self.filepath.split('.')[1]
        return self.filetype

    def summary_image_io(self):
        if self.algotype=='cnmfe':
            summary_images_=self.rf[self.group0[0]]['cn']
        elif self.algotype=='extract':
            summary_images_=self.rf[self.group0[0]]['info']['summary_image']
        else:
            raise Exception('unknown analysis type, enter ''cnmfe'' or ''extract')
        self.cn=np.array(summary_images_).T
        return self.cn

    def raw_datafile_io(self):
        if self.algotype=='cnmfe':
            self.raw_data_file=self.rf[self.group0[0]]['movieList']
        elif self.algotype=='extract':
            self.raw_data_file=self.rf[self.group0[0]]['file']
        else:
            raise Exception('unknown analysis type, enter ''cnmfe'' or ''extract')
            self.raw_data_file=None
        return self.raw_data_file

    @property
    def dims(self):
        if self._dims is None:
            self.imagesize=self.extdims
            return list(self.imagesize)
        else:
            self.imagesize=self._dims
            return self.imagesize

    @property
    def no_rois(self):
        if self._no_rois is None:
            raw_images=self.masks
            return raw_images.shape[1]
        else:
            return self._no_rois

    @property
    def roi_idx(self):
        if len(self._roi_ids) is 0:
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
            no_ROIs=self.no_rois
            raw_images=self.masks
            roi_location=np.ndarray([2,no_ROIs],dtype='int')
            for i in range(no_ROIs):
                temp=np.where(raw_images[:,:,i]==np.amax(raw_images[:,:,i]))
                roi_location[:,i]=np.array([temp[0][0],temp[1][0]]).T
            return roi_location
        else:
            return self._roi_locs

    @property
    def num_of_frames(self):
        if self._num_of_frames is None:
            extracted_signals=self.roi_response
            return extracted_signals.shape[1]
        else:
            return self._num_of_frames

    @property
    def samp_freq(self):
        if self._samp_freq is None:
            time=self.total_time
            nframes=self.num_of_frames
            return nframes/time
        else:
            return self._samp_freq

    @staticmethod
    def nwbwrite(nwbfilename,sourcefilepath,analysis_type,propertydict):
        write_recording(TraceExtractor(sourcefilepath,analysis_type),propertydict,nwbfilename)
        print(f'successfully saved nwb as {nwbfilename}')

    #defining the abstract class enformed methods:
    def get_roi_ids(self):
        return list(range(self.no_rois))

    def get_num_rois(self):
        return self.no_rois

    def get_roi_locations(self,ROI_ids=None):
        if ROI_ids is None:
            return self.roi_locs
        else:
            return self.roi_locs[:,ROI_ids]

    def get_num_frames(self):
        return self.num_of_frames

    def get_sampling_frequency(self):
        return self.samp_freq

    def get_traces(self, ROI_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if ROI_ids is None:#!!need to make ROI as a 2-d array, specifying the location in image plane
            ROI_ids = range(self.get_roi_ids())
        return self.roi_response[ROI_ids,start_frame:end_frame]

    def get_masks(self, ROI_ids=None):
        if ROI_ids is None:#!!need to make ROI as a 2-d array, specifying the location in image plane
            ROI_ids = range(self.get_roi_ids())
        return self.masks.reshape([*self.dims,*self.no_rois])[:,:,ROI_ids]

    def get_movie_framesize(self):
        return self.dims

    def get_raw_file(self):
        return self.raw_data_file
