from ciextractor import CIExtractor
from datacontainer import CIDataContainer
from nwbwriter import save_NWB
import numpy as np

try:
    import h5py
    HAVE_SNTE = True
except ImportError:
    HAVE_SNTE = False

print('running outside class schnitzer also')
class SchnitzerTraceExtractor(CIExtractor,CIDataContainer):
    name='SchnitzerTraceExtractor'
    installed=HAVE_SNTE

    installation_mesg = "To use the ScnitzerTraceExtractor install h5py: \n\n pip install h5py\n\n"  # error message when not installed

    def __init__(self,filepath,analysis_type,devicetype):
        #storing some meta data associated with general Calcium Imaging
        self._recordingfile=filepath
        self._analysis_type=analysis_type
        # self._fov=fov# field of view of the microscope
        self._recordingDevice=devicetype
        self.data=CIDataContainer(filepath,analysis_type)


    # def __del__(self):
    #     self.rf.close()

    def get_ROI_ids(self):
        self.ROIids=list(range(self.data.noROIs))
        return self.ROIids

    def get_num_ROIs(self):
        self._nROIs=self.data.noROIs
        return self._nROIs

    def get_ROI_locations(self,ROI_ids=None):
        self._ROIlocs=self.data.ROIlocs
        if ROI_ids is None:
            return self._ROIlocs
        else:
            return self._ROIlocs[:,ROI_ids]

    def get_num_frames(self):
        self._num_frames=self.data.NumOfFrames
        return self._num_frames

    def get_sampling_frequency(self):
        self._samplingRate=self.data.SampFreq
        return self._samplingRate

    def get_traces(self, ROI_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if ROI_ids is None:#!!need to make ROI as a 2-d array, specifying the location in image plane
            ROI_ids = range(self.get_ROI_ids())
        self._extractedSignals=self.data.C
        return self._extractedSignals[ROI_ids,start_frame:end_frame]

    def get_masks(self, ROI_ids=None):
        if ROI_ids is None:#!!need to make ROI as a 2-d array, specifying the location in image plane
            ROI_ids = range(self.get_ROI_ids())
        self._extractedmasks=self.data.A.reshape([*self.data.dims,*self.data.noROIs])
        return self.maskExtracter_IO[:,:,ROI_ids]

    def get_movie_framesize(self):
        self.movie_framesize=self.data.dims
        return self.movie_framesize

    @staticmethod
    def nwbwrite(nwbfilename,sourcefilepath,analysis_type):
        save_NWB(CIDataContainer(sourcefilepath,analysis_type),nwbfilename)
        print(f'successfully saved nwb as {nwbfilename}')

# def maskExtracter_IO(filepath,algotype):
#     with h5py.File(filepath,'r') as f:
#         group0_temp=list(f.keys())
#         group0=[a for a in group0_temp if '#' not in a]
#         if algotype=='cnmfe':
#             raw_images=f[group0[0]]['extractedImages']
#         elif algotype=='extract':
#             raw_images=f[group0[0]]['filters']
#         else:
#             raise Exception('unknown analysis type, enter ''cnmfe'' or ''extract')
#         return np.array(raw_images)
#
# def imageSizeExtractor_IO(filepath,algotype):
#     raw_images=maskExtracter_IO(filepath,algotype)
#     return np.array(raw_images[:,:,1].shape)
#
#
# def traceExtracter_IO(filepath,algotype):
#     with h5py.File(filepath,'r') as f:
#         group0_temp=list(f.keys())
#         group0=[a for a in group0_temp if '#' not in a]
#         if algotype=='cnmfe':
#             extractedSignals=f[group0[0]]['extractedSignals']
#         elif algotype=='extract':
#             extractedSignals=f[group0[0]]['traces']
#         else:
#             raise Exception('unknown analysis type, enter ''cnmfe'' or ''extract')
#         return extractedSignals
#
# def RoiCountExtracter_IO(filepath,algotype):
#     raw_images=maskExtracter_IO(filepath,algotype)
#     return raw_images.shape[2]
#
# def RoiLocExtracter_IO(filepath,algotype):
#     no_ROIs=RoiCountExtracter_IO(filepath,algotype)
#     raw_images=maskExtracter_IO(filepath,algotype)
#     ROI_loca=np.ndarray([2,no_ROIs],dtype='int')
#     for i in range(no_ROIs):
#         temp=np.where(raw_images[:,:,i]==np.amax(raw_images[:,:,i]))
#         ROI_loca[:,i]=np.array([temp[0][0],temp[1][0]]).T
#     return ROI_loca
#
# def noFrameExtractor_IO(filepath,algotype):
#     extractedSignals=traceExtracter_IO(filepath,algotype)
#     return extractedSignals.shape[1]
#
# def TotExpTimeExtractor_IO(filepath):
#     with h5py.File(filepath,'r') as f:
#         group0_temp=list(f.keys())
#         group0=[a for a in group0_temp if '#' not in a]
#         return f[group0[0]]['time']['totalTime'][0]
#
# def FreqExtractor_IO(filepath,algotype):
#     time=TotExpTimeExtractor_IO(filepath)
#     nframes=noFrameExtractor_IO(filepath,algotype)
#     return nframes/time
#
# def FiletypeExtractor_IO(filepath):
#     return filepath.split('.')[1]
