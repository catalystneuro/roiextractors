import numpy as np
from roiextractors import SegmentationExtractor
import os


class Suite2pSegmentationExtractor(SegmentationExtractor):
    def __init__(self, fileloc):
        """
        Creating SegmentationExtractor object out of suite 2p data type.
        Parameters
        ----------
        op: dict
            options that need the suite 2p file takes as arguments.
        db: dict
            db overwrites any ops (allows for experiment specific settings)
        """
        SegmentationExtractor.__init__(self)
        self.filepath = fileloc
        self.no_planes_extract = 1
        self.stat = self._load_npy('stat.npy')
        self.F = self._load_npy('F.npy', mmap_mode='r')
        self.Fneu = self._load_npy('Fneu.npy', mmap_mode='r')
        self.spks = self._load_npy('spks.npy', mmap_mode='r')
        self.iscell = self._load_npy('iscell.npy', mmap_mode='r')
        self.ops = [i.item() for i in self._load_npy('ops.npy')]
        self.op_inp = self.ops[0]
        self.no_channels = self.op_inp['nchannels']
        self.no_planes = self.op_inp['nplanes']
        self.rois_per_plane = [i.shape[0] for i in self.iscell]
        self.raw_images = None
        self.roi_response = None

    def _load_npy(self, filename, mmap_mode=None):
        ret_val = [[None]] * self.no_planes_extract
        for i in range(self.no_planes_extract):
            fpath = os.path.join(self.filepath, 'Plane{}'.format(i), filename)
            ret_val[i] = np.load(fpath,
                                 mmap_mode=mmap_mode,
                                 allow_pickle=not mmap_mode and True)
        return ret_val

    @property
    def image_dims(self):
        return [self.ops[0]['Lx'], self.ops[0]['Ly']]

    @property
    def no_rois(self):
        return sum([len(i) for i in self.stat])

    @property
    def roi_idx(self):
        return [i for i in range(self.no_rois)]

    @property
    def accepted_list(self):
        plane_wise = [np.where(i[:, 0] == 1)[0] for i in self.iscell]
        return [plane_wise[0].tolist(), (len(self.stat[0]) + plane_wise[0]).tolist()][0:self.no_planes_extract]

    @property
    def rejected_list(self):
        plane_wise = [np.where(i[:, 0] == 0)[0] for i in self.iscell]
        return [plane_wise[0].tolist(), (len(self.stat[0]) + plane_wise[0]).tolist()][0:self.no_planes_extract]

    @property
    def roi_locs(self):
        plane_wise = [[j['med'] for j in i] for i in self.stat]
        ret_val = []
        [ret_val.extend(i) for i in plane_wise]
        return ret_val

    @property
    def num_of_frames(self):
        return sum([i['nframes'] for i in self.ops])

    @property
    def samp_freq(self):
        return self.ops[0]['fs'] * self.no_planes

    @staticmethod
    def write_segmentation(nwb_file_path):
        return NotImplementedError

    # defining the abstract class enforced methods:
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

    def get_traces(self, roi_ids=None, start_frame=None, end_frame=None, name=None):
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
        if name == 'Fluorescence':
            return np.concatenate(self.F[0:self.no_planes_extract])[[roi_idx_], start_frame:end_frame].squeeze()
        if name == 'Neuropil':
            return np.concatenate(self.Fneu[0:self.no_planes_extract])[[roi_idx_], start_frame:end_frame].squeeze()
        if name == 'Deconvolved':
            return np.concatenate(self.spks[0:self.no_planes_extract])[[roi_idx_], start_frame:end_frame].squeeze()
        else:
            return None

    def get_image_masks(self, roi_ids=None):
        return None

    def get_pixel_masks(self, roi_ids=None):
        pixel_mask = [None] * self.no_rois
        c = 0
        for i in range(self.no_planes_extract):
            for j in range(self.rois_per_plane[i]):
                pixel_mask[c] = np.array([self.stat[i][j]['ypix'],
                                          self.stat[i][j]['xpix'],
                                          self.stat[i][j]['lam'],
                                          c * np.ones(self.stat[i][j]['lam'].size)]).T
                c += 1
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return np.concatenate([pixel_mask[i] for i in roi_idx_])

    def get_images(self):
        bg_strs = ['meanImg', 'Vcorr', 'max_proj', 'meanImg_chan2']
        out_dict = {'Background0': {}}
        for bstr in bg_strs:
            if bstr in self.op_inp:
                if bstr == 'Vcorr' or bstr == 'max_proj':
                    img = np.zeros((self.op_inp['Ly'], self.op_inp['Lx']), np.float32)
                    img[self.op_inp['yrange'][0]:self.op_inp['yrange'][-1],
                    self.op_inp['xrange'][0]:self.op_inp['xrange'][-1]] = self.op_inp[bstr]
                else:
                    img = self.op_inp[bstr]
                out_dict['Background0'].update({bstr: img})
        return out_dict

    def get_movie_framesize(self):
        return self.image_dims

    def get_movie_location(self):
        return os.path.abspath(os.path.join(self.filepath, os.path.pardir))

    def get_channel_names(self):
        return [f'OpticalChannel{i}' for i in range(self.no_channels)]

    def get_num_channels(self):
        return self.no_channels
