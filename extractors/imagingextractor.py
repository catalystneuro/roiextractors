from abc import ABC, abstractmethod
import numpy as np
from spikeextractors.baseextractor import BaseExtractor
from .extraction_tools import ArrayType, PathType, NumpyArray, DtypeType


class ImagingExtractor(ABC, BaseExtractor):
    '''An abstract class that contains all the meta-data and input data from
       the imaging data.
    '''
    def __init__(self):
        BaseExtractor.__init__(self)
        self._memmapped = False

    @abstractmethod
    def get_frame(self, frame_idx) -> NumpyArray:
        pass

    @abstractmethod
    def get_frames(self, frame_idxs) -> NumpyArray:
        pass

    @abstractmethod
    def get_video(self, start_frame=None, end_frame=None) -> NumpyArray:
        pass

    @abstractmethod
    def get_image_size(self) -> ArrayType:
        pass

    @abstractmethod
    def get_num_frames(self) -> int:
        pass

    @abstractmethod
    def get_sampling_frequency(self) -> float:
        pass

    @abstractmethod
    def get_dtype(self) -> DtypeType:
        pass

    @abstractmethod
    def get_channel_names(self) -> list:
        '''List of  channels in the recoding.

        Returns
        -------
        channel_names: list
            List of strings of channel names
        '''
        pass

    @abstractmethod
    def get_num_channels(self) -> int:
        '''Total number of active channels in the recording

        Returns
        -------
        no_of_channels: int
            integer count of number of channels
        '''
        pass

    def save_memmap(self, save_path):
        raise NotImplementedError

    def load_memmap(self, load_path):
        raise NotImplementedError

    @staticmethod
    def write_imaging(imaging, savepath):
        '''
        Static method to write imaging.

        Parameters
        ----------
        imaging: ImagingExtractor object
            The EXTRACT segmentation object from which an EXTRACT native format
            file has to be generated.
        savepath: str
            path to save the native format.
        '''
        raise NotImplementedError


