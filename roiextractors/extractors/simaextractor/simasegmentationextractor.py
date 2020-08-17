import numpy as np
from ...segmentationextractor import SegmentationExtractor
import dill
import re
import os
import pickle
from shutil import copyfile

try:
    import sima
    HAVE_SIMA = True
except:
    HAVE_SIMA = False


class SimaSegmentationExtractor(SegmentationExtractor):
    """
    This class inherits from the SegmentationExtractor class, having all
    its functionality specifically applied to the dataset output from
    the \'SIMA\' ROI segmentation method.
    """
    extractor_name = 'SimaSegmentation'
    installed = HAVE_SIMA  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = "To use the SimaSegmentationExtractor install sima: \n\n pip install sima\n\n"  # error message when not installed

    def __init__(self, filepath, sima_segmentation_label='auto_ROIs'):
        """
        Parameters
        ----------
        filepath: str
            The location of the folder containing dataset.sima file and the raw
            image file(s) (tiff, h5, .zip)
        sima_segmentation_label: str
            name of the ROIs in the dataset from which to extract all ROI info
        """
        assert HAVE_SIMA, self.installation_mesg
        SegmentationExtractor.__init__(self)
        self.filepath = filepath
        self._convert_sima(filepath)
        self._dataset_file = self._file_extractor_read()
        self.channel_names = [str(i) for i in self._dataset_file.channel_names]
        self.no_of_channels = len(self.channel_names)
        self.sima_segmentation_label = sima_segmentation_label
        self.image_masks, self.extimage_dims, self.raw_images =\
            self._image_mask_extractor_read()
        self.pixel_masks = self._pixel_mask_extractor_read()
        self.roi_response = self._trace_extractor_read()
        self.cn = self._summary_image_read()
        self.total_time = self._tot_exptime_extractor_read()
        self.filetype = self._file_type_extractor_read()
        self.raw_movie_file_location = self._raw_datafile_read()
        # Not found data:
        self._no_background_comps = 1
        self.snr_comp = np.nan * np.ones(self.roi_response.shape)
        self.r_values = np.nan * np.ones(self.roi_response.shape)
        self.cnn_preds = np.nan * np.ones(self.roi_response.shape)
        self._rejected_list = []
        self._accepted_list = None
        self.idx_components = self.accepted_list
        self.idx_components_bad = self.rejected_list
        self.image_masks_bk = np.nan * \
            np.ones(list(self.raw_images.shape[0:2]) + [self._no_background_comps])
        self.roi_response_bk = np.nan * \
            np.ones([self._no_background_comps, self.roi_response.shape[1]])

    @staticmethod
    def _convert_sima(old_pkl_loc):
        """
        This function is used to convert python 2 pickles to python 3 pickles.
        Forward compatibility of \'*.sima\' files containing .pkl dataset, rois,
        sequences, signals, time_averages.

        Replaces the pickle file with a python 3 version with the same name. Saves
        the old Py2 pickle as \'oldpicklename_p2.pkl\''

        Parameters
        ----------
        old_pkl_loc: str
            Path of the pickle file to be converted
        """
        # Make a name for the new pickle
        old_pkl_loc = old_pkl_loc + '/'
        for dirpath, dirnames, filenames in os.walk(old_pkl_loc):
            _exit = [True for file in filenames if '_p2.pkl' in file]
            if True in _exit:
                print('pickle already in Py3 format')
                continue
            for file in filenames:
                if '.pkl' in file:
                    old_pkl = os.path.join(dirpath, file)
                    print(old_pkl)
                    # Make a name for the new pickle
                    new_pkl_name = os.path.splitext(os.path.basename(old_pkl))[0] + "_p2.pkl"
                    base_directory = os.path.split(old_pkl)[0]
                    new_pkl = base_directory + '/' + new_pkl_name
                    # Convert Python 2 "ObjectType" to Python 3 object
                    dill._dill._reverse_typemap["ObjectType"] = object

                    # Open the pickle using latin1 encoding
                    with open(old_pkl, "rb") as f:
                        loaded = pickle.load(f, encoding="latin1")
                    copyfile(old_pkl, new_pkl)
                    os.remove(f.name)
                    # Re-save as Python 3 pickle
                    with open(old_pkl, "wb") as outfile:
                        pickle.dump(loaded, outfile)

    def _file_extractor_read(self):
        _img_dataset = sima.ImagingDataset.load(self.filepath)
        _img_dataset._savedir = self.filepath
        return _img_dataset

    def _image_mask_extractor_read(self):
        _sima_rois = self._dataset_file.ROIs
        if len(_sima_rois) > 1:
            if self.sima_segmentation_label in list(_sima_rois.keys()):
                _sima_rois_data = _sima_rois[self.sima_segmentation_label]
            else:
                raise Exception('Enter a valid name of ROIs from: {}'.format(
                    ','.join(list(_sima_rois.keys()))))
        elif len(_sima_rois) == 1:
            _sima_rois_data = list(_sima_rois.values())[0]
            self.sima_segmentation_label = list(_sima_rois.keys())[0]
        else:
            raise Exception('no ROIs found in the sima file')

        image_masks_ = [np.squeeze(np.array(roi_dat)).T for roi_dat in _sima_rois_data]
        _raw_images_trans = np.array(image_masks_).T
        return _raw_images_trans.reshape(
            [np.prod(_raw_images_trans.shape[0:2]),
             _raw_images_trans.shape[2]],
            order='F'),\
            _raw_images_trans.shape[0:2],\
            _raw_images_trans

    def _pixel_mask_extractor_read(self):
        return super()._pixel_mask_extractor(self.raw_images, self.roi_idx)

    def _trace_extractor_read(self):
        for channel_now in self.channel_names:
            for labels in self._dataset_file.signals(channel=channel_now):
                if labels:
                    _active_channel = channel_now
                    break
            print('extracting signal from channel {} from {} no of channels'.
                  format(_active_channel, self.no_of_channels))
        # label for the extraction method in SIMA:
        for labels in self._dataset_file.signals(channel=_active_channel):
            _count = 0
            if not re.findall(r'[\d]{4}-[\d]{2}-[\d]{2}-', labels):
                _count = _count + 1
                _label = labels
                break
        if _count > 1:
            print('multiple labels found for extract method using {}'.format(_label))
        elif _count == 0:
            print('no label found for extract method using {}'.format(labels))
            _label = labels
        extracted_signals = np.array(self._dataset_file.signals(
            channel=_active_channel)[_label]['raw'][0])
        return extracted_signals

    def _tot_exptime_extractor_read(self):
        return None

    def _file_type_extractor_read(self):
        return self.filepath.split('.')[1]

    def _summary_image_read(self):
        summary_images_ = np.squeeze(self._dataset_file.time_averages[0]).T
        return np.array(summary_images_).T

    def _raw_datafile_read(self):
        return None
        # try:
        #     return self._dataset_file.sequences[0]._path
        # except AttributeError:
        #     return self._dataset_file.sequences[0]._sequences[0]._path

    @property
    def image_dims(self):
        return list(self.extimage_dims)

    @property
    def no_rois(self):
        return self.raw_images.shape[2]

    @property
    def roi_idx(self):
        id_vals = []
        for ind, val in enumerate(list(self._dataset_file.ROIs.values())[0]):
            if val.id:
                id_vals.append(int(val.id))
            else:
                id_vals.append(int(ind * -1))
        return id_vals

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
        no_ROIs = self.no_rois
        raw_images = self.raw_images
        roi_location = np.ndarray([2, no_ROIs], dtype='int')
        for i in range(no_ROIs):
            temp = np.where(raw_images[:, :, i] == np.amax(raw_images[:, :, i]))
            roi_location[:, i] = np.array([np.median(temp[0]), np.median(temp[1])]).T
        return roi_location

    @property
    def num_of_frames(self):
        extracted_signals = self.roi_response
        return extracted_signals.shape[1]

    @property
    def samp_freq(self):
        time = self.total_time
        nframes = self.num_of_frames
        if time:
            return nframes / time
        else:
            return 0.

    @staticmethod
    def write_segmentation(segmentation_object, savepath):
        raise NotImplementedError

    # defining the abstract class enformed methods:
    def get_roi_ids(self):
        return self.roi_idx

    def get_num_rois(self):
        return self.no_rois

    def get_roi_locations(self, roi_ids=None):
        if roi_ids is None:
            return self.roi_locs
        else:
            roi_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
            return self.roi_locs[:, roi_idx_]

    def get_num_frames(self):
        return self.num_of_frames

    def get_sampling_frequency(self):
        return self.samp_freq

    def get_traces(self, roi_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames() + 1
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return self.roi_response[roi_idx_, start_frame:end_frame]

    def get_image_masks(self, roi_ids=None):
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return self.raw_images[:, :, roi_idx_]

    def get_pixel_masks(self, roi_ids=None):
        if roi_ids is None:
            roi_idx_ = self.roi_idx
        else:
            roi_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        temp = np.empty((1, 4))
        for i, roiid in enumerate(roi_idx_):
            temp = \
                np.append(temp, self.pixel_masks[self.pixel_masks[:, 3] == roiid, :], axis=0)
        return temp[1::, :]

    def get_images(self):
        return None

    def get_movie_framesize(self):
        return self.image_dims

    def get_movie_location(self):
        return self.raw_movie_file_location

    def get_channel_names(self):
        return self.channel_names

    def get_num_channels(self):
        return self.no_of_channels
