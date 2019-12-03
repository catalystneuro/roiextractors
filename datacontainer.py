
''' data container class:
    input all releveant data and metadata related to the main analysis file parsed by h5py
    Masks:
        type: np.ndarray (dimensions: # of pixels(d1 X d2) x # of ROIs)

    Signal:
        type: np.ndarray (dimensions: # of ROIs x # components), 3D

    BackgroundSignal:
        type: np.ndarray (dimensions: # of BackgroundRegions x # components), 3D

    BackgroundMasks:
        type: np.ndarray (dimensions: # of pixels(d1 X d2) x # of ROIs)

    TraceResiduals:
        type: np.ndarray (dimensions: # of ROIs x # of timesteps)

    DeconvolvedActivity:
        type: np.ndarray (dimensions: # of ROIs x # of timesteps)

    Correlation:
        type: np.ndarray (dimensions: # of pixels(d1 X d2) x # of ROIs)

    ROIids:
        type: np.ndarray (dimensions: 1 x # of ROIs)

    noROIs:
        type: np.ndarray (dimensions: 1D)

    ROIlocs:
        type: np.ndarray (dimensions: # of ROIs x 2)

    SampFreq:
        type: np.ndarray (dimensions: 1D)

    NumOfFrames:
        type: np.ndarray (dimensions: 1D)

    MovieFramesize:
        type: np.ndarray (dimensions: 2D)

'''
import numpy as np
import h5py
class CIDataContainer:

    def __init__(self,filepath, algotype, masks=None, signal=None, background_signal=None, background_masks=None, 
                 trace_residuals=None, rawfileloc=None, deconvolved_activity=None,accepted_lst=None,CorrelationImage=[],ROIidx=[],ROIlocs=None,SampFreq=None,nback=1):
        self.filepath=filepath
        self.algotype=algotype
        self.rf,self.group0=self.FileExtractor_IO()
        if len(Masks) is 0:
            self.MaskExtracter_IO()
        elif len(Masks.shape)>2:
            self.A=temp.reshape([Masks.shape[2],np.prod(Masks.shape[0:1])]).T
        else:
            self.A=Masks

        if len(Signal) is 0:
            self.TraceExtracter_IO()
        else:
            self.C=Signal

        if len(CorrelationImage) is 0:
            self.SummaryImage_IO()
        else:
            self.Cn=CorrelationImage

        self.TotExpTimeExtractor_IO()
        self.FiletypeExtractor_IO()

        if rawfileloc is None:
            self.RawDataFile_IO()
        else:
             self.raw_data_file=rawfileloc

        if len(BackgroundMasks) is 0:
            self.b=np.nan*np.ones([self.A.shape[0],nback])
        else:
            self.b=BackgroundMasks

        if len(BackgroundSignal) is 0:
            self.f=np.nan*np.ones([nback,self.C.shape[1]])
        else:
            self.f=BackgroundSignal

        if len(TraceResiduals) is 0:
            self.YrA=np.nan*np.ones(self.C.shape)
        else:
            self.YrA=TraceResiduals

         #remains to mine from mat file or nan it
         # need to resolve the diff between correlation image and summary image
        self._ROIids=ROIidx
        self._ROIlocs=ROIlocs
        self._noROIs=None
        self._SampFreq=SampFreq
        self._NumOfFrames=None
        self.SNR_comp=np.nan*np.ones(self.C.shape)    #remains to mine from mat file or nan it
        self.r_values=np.nan*np.ones(self.C.shape)     #remains to mine from mat file or nan it
        self.cnn_preds=np.nan*np.ones(self.C.shape)    #remains to mine from mat file or nan it
        self._rejected_list=[] #remains to mine from mat file or nan it
        self._accepted_list=accepted_lst #remains to mine from mat file or nan it
        self.idx_components=self.accepted_list #remains to mine from mat file or nan it
        self.idx_components_bad=self.rejected_list
        self._dims=None
        self._raw_data_file=rawfileloc
        self.FileClose()

    def FileClose(self):
        self.rf.close()

    def FileExtractor_IO(self):
        f=h5py.File(self.filepath,'r')
        group0_temp=list(f.keys())
        group0=[a for a in group0_temp if '#' not in a]
        return f,group0

    def MaskExtracter_IO(self):
        if self.algotype=='cnmfe':
            raw_images=self.rf[self.group0[0]]['extractedImages']
        elif self.algotype=='extract':
            raw_images=self.rf[self.group0[0]]['filters']
        else:
            raise Exception('unknown analysis type, enter ''cnmfe'' or ''extract')
        temp=np.array(raw_images).transpose()
        self.images=temp
        self.A=temp.reshape([np.prod(temp.shape[0:2]),temp.shape[2]])
        self.extdims=temp.shape[0:2]
        return self.A

    def TraceExtracter_IO(self):
        if self.algotype=='cnmfe':
            extractedSignals=self.rf[self.group0[0]]['extractedSignals']
        elif self.algotype=='extract':
            extractedSignals=self.rf[self.group0[0]]['traces']
        else:
            raise Exception('unknown analysis type, enter ''cnmfe'' or ''extract')
        self.C=np.array(extractedSignals).T
        return self.C

    def TotExpTimeExtractor_IO(self):
        self.TotalTime=self.rf[self.group0[0]]['time']['totalTime'][0][0]
        return self.TotalTime

    def FiletypeExtractor_IO(self):
        self.filetype=self.filepath.split('.')[1]
        return self.filetype

    def SummaryImage_IO(self):
        if self.algotype=='cnmfe':
            summary_images_=self.rf[self.group0[0]]['Cn']
        elif self.algotype=='extract':
            summary_images_=self.rf[self.group0[0]]['info']['summary_image']
        else:
            raise Exception('unknown analysis type, enter ''cnmfe'' or ''extract')
        self.Cn=np.array(summary_images_).T
        return self.Cn

    def RawDataFile_IO(self):
        if self.algotype=='cnmfe':
            self.raw_data_file=self.rf[self.group0[0]]['movieList']
        elif self.algotype=='extract':
            self.raw_data_file=self.rf[self.group0[0]]['file']
        else:
            raise Exception('unknown analysis type, enter ''cnmfe'' or ''extract')
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
    def noROIs(self):
        if self._noROIs is None:
            raw_images=self.A
            return raw_images.shape[1]
        else:
            return self._noROIs

    @property
    def ROIidx(self):
        if len(self._ROIids) is 0:
            return list(range(self.noROIs))
        else:
            return self._ROIids

    @property
    def accepted_list(self):
        if self._accepted_list is None:
            return list(range(self.noROIs))
        else:
            return self._accepted_list

    @property
    def rejected_list(self):
        return [a for a in range(self.noROIs) if a not in set(self.accepted_list)]

    @property
    def ROIlocs(self):
        if self._ROIlocs is None:
            no_ROIs=self.noROIs
            raw_images=self.A
            ROI_loca=np.ndarray([2,no_ROIs],dtype='int')
            for i in range(no_ROIs):
                temp=np.where(raw_images[:,:,i]==np.amax(raw_images[:,:,i]))
                ROI_loca[:,i]=np.array([temp[0][0],temp[1][0]]).T
            return ROI_loca
        else:
            return self._ROIlocs

    @property
    def NumOfFrames(self):
        if self._NumOfFrames is None:
            extractedSignals=self.C
            return extractedSignals.shape[1]
        else:
            return self._NumOfFrames

    @property
    def SampFreq(self):
        if self._SampFreq is None:
            time=self.TotalTime
            nframes=self.NumOfFrames
            return nframes/time
        else:
            return self._SampFreq
