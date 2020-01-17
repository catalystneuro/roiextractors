from abc import ABC, abstractmethod
import numpy as np
print('running outside class ciextractor also')


class SegmentationExtractor(ABC):
    def __init__(self):
        self._epochs = {}
        self._channel_properties = {}
        self.id = random.randint(a=0, b=9223372036854775807)

    @abstractmethod
    def get_traces(self, ROI_ids=None, start_frame=None, end_frame=None):
        ''' Extracts specific regions of traces based on the start and end frame values
        ----------
        start_frame: int
            The starting frame of the trace to be returned (inclusive).
        end_frame: int
            The ending frame of the trace to be returned (exclusive).
        channel_ids: array_like
            A list or 1D array of Regions of Interest (ints) from which each trace will be
            extracted.
        Returns
        ----------
        traces: numpy.ndarray
            A 2D array that contains all of the traces from each channel.
            Dimensions are: (number of ROIs x num_frames)
        '''
        pass

    @abstractmethod
    def get_num_frames(self):
        '''This function returns the number of frames in the recording.
        Returns
        -------
        num_frames: int
            Number of frames in the recording (duration of recording).
        '''
        pass

    @abstractmethod
    def get_sampling_frequency(self):
        '''This function returns the sampling frequency in units of Hz.
        Returns
        -------
        fs: float
            Sampling frequency of the recordings in Hz.
        '''
        pass

    @abstractmethod
    def get_roi_locations(self):
        '''
        Returns the locations of the Regions of Interest
        Returns
        ------
        roi_locs: numpy.ndarray
            2-D array: 2 X no_ROIs. The pixel ids (x,y) where the centroid of the ROI is.
        '''
        pass

    @abstractmethod
    def get_roi_ids(self):
        '''Returns the list of channel ids. If not specified, the range from 0 to num_channels - 1 is returned.
        Returns
        -------
        channel_ids: list
            Channel list
        '''
        pass
