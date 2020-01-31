from abc import ABC, abstractmethod
import numpy as np


class SegmentationExtractor(ABC):
    '''An abstract class that contains all the meta-data and output data from
        the ROI segmentation operation when applied to the pre-processed data.
        It also contains methods to read from and write to various data formats
        ouput from the processing pipelines like SIMA, CaImAn, Suite2p, CNNM-E.
        All the methods with @abstract decorator have to be defined by the
        format specific classes that inherit from this.
    '''

    def __init__(self):
        self._epochs = {}
        self._channel_properties = {}
        self.id = np.random.randint(a=0, b=9223372036854775807)

    @abstractmethod
    def get_traces(self, ROI_ids=None, start_frame=None, end_frame=None):
        ''' Extracts specific regions of traces based on the start and end frame values

        Parameters
        ----------
        start_frame: int
            The starting frame of the trace to be returned (inclusive).
        end_frame: int
            The ending frame of the trace to be returned (exclusive).
        ROI_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

        Returns
        ----------
        roi_response: numpy.ndarray
            A 2D array that contains all of the traces from each channel.
            Dimensions are: (number of ROIs x num_frames)
        '''
        pass

    @abstractmethod
    def get_num_frames(self):
        '''This function returns the number of frames in the recording.

        Returns
        -------
        num_of_frames: int
            Number of frames in the recording (duration of recording).
        '''
        pass

    @abstractmethod
    def get_sampling_frequency(self):
        '''This function returns the sampling frequency in units of Hz.

        Returns
        -------
        samp_freq: float
            Sampling frequency of the recordings in Hz.
        '''
        pass

    @abstractmethod
    def get_roi_locations(self, ROI_ids=None):
        '''
        Returns the locations of the Regions of Interest

        Parameters
        ----------
        ROI_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

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
            Channel list.
        '''
        pass

    @abstractmethod
    def get_num_rois(self):
        '''Returns total number of Regions of Interest in the acquired images.

        Returns
        -------
        no_rois: int
            integer number of ROIs extracted.
        '''
        pass

    @abstractmethod
    def get_image_masks(self, ROI_ids=None):
        '''Returns the image masks extracted from segmentation algorithm.

        Parameters
        ----------
        ROI_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

        Returns
        -------
        image_masks: numpy.ndarray
            3-D array(val 0 or 1): image_height X image_width X length(ROI_ids)
        '''
        pass

    @abstractmethod
    def get_pixel_masks(self, ROI_ids=None):
        '''Returns the weights applied to each of the pixels of the mask.

        Parameters
        ----------
        ROI_ids: array_like
            A list or 1D array of ids of the ROIs. Length is the number of ROIs
            requested.

        Returns
        -------
        pixel_masks: numpy.ndarray
            2-D array: (total pixels X length(ROI_ids)) X 4
            Column 1 and 2 are x and y values of the pixels.
            Column 3 is the weight for that pixel.
            Column 4 is the id of the ROI that that pixel belongs to
        '''
        pass

    @abstractmethod
    def get_movie_framesize(self):
        '''Frame size of movie ( x and y size of image).

        Returns
        -------
        no_rois: array_like
            2-D array: image y x image x
        '''
        pass

    @abstractmethod
    def get_raw_file(self):
        '''Raw file location on storage.

        Returns
        -------
        raw_data_file_location: str
            location as a string
        '''
        pass

    @staticmethod
    def _pixel_mask_extractor(_raw_images_trans, _roi_idx):
        '''An alternative data format for storage of image masks.

        Returns
        -------
        pixel_mask: numpy array
            Total pixels X 4 size. Col 1 and 2 are x and y location of the mask
            pixel, Col 3 is the weight of that pixel, Col 4 is the ROI index.
        '''
        temp = np.empty((1, 4))
        for i, roiid in enumerate(_roi_idx):
            _locs = np.where(_raw_images_trans[:, :, i] > 0)
            _pix_values = _raw_images_trans[_raw_images_trans[:, :, i] > 0, i]
            temp = np.append(temp, np.concatenate(
                                (_locs[0].reshape([1, np.size(_locs[0])]),
                                _locs[1].reshape([1, np.size(_locs[1])]),
                                _pix_values.reshape([1, np.size(_locs[1])]),
                                roiid * np.ones([1, np.size(_locs[1])]))).T, axis=0)
        return temp[1::, :]

        @abstractmethod
        def get_channel_names(self):
            '''List of  channels in the recoding.

            Returns
            -------
            channel_names: list
                List of strings of channel names
            '''
            pass

        @abstractmethod
        def get_no_of_channels(self):
            '''Total number of active channels in the recording

            Returns
            -------
            no_of_channels: int
                integer count of number of channels
            '''
            pass
