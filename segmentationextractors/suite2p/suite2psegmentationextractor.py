import numpy as np
from segmentationextractors import SegmentationExtractor
from suite2p import run_s2p
from suite2p.io import nwb as s2p_nwb


class Suite2pSegmentationExtractor(SegmentationExtractor):

    def __init__(self, fileloc, op=None, db=None, s2pobj=None):
        """
        Creating SegmentationExtractor object out of suite 2p data type.
        Parameters
        ----------
        op: dict
            options that need the suite 2p file takes as arguments.
        db: dict
            db overwrites any ops (allows for experiment specific settings)
        """
        if op is None:
            op = run_s2p.default_ops()
        if db is None:
            db = dict(data_path=[fileloc])
        else:
            db.update(data_path=[fileloc])
        if s2pobj is None:
            self.s2p_obj = run_s2p.run_s2p(ops=op, db=db)
        else:
            self.s2p_obj = s2pobj

        if len(op['data_path'] == 0):
            self.savepath = fileloc + '\suite2p'
        else:
            self.savepath = op['data_path'][0] + '\suite2p'
        self.fileloc = fileloc
        self.op_inp = op
        self.no_channels = op['nchannels']
        self.no_planes = op['nplanes']
        self.stat = self._load_npy('stat.npy')
        self.F = self._load_npy('F.npy',mmap_mode='r')
        self.Fneu = self._load_npy('Fneu.npy',mmap_mode='r')
        self.spks = self._load_npy('spks.npy',mmap_mode='r')
        self.iscell = self._load_npy('iscell.npy',mmap_mode='r')
        self.ops = [i.item() for i in self._load_npy('ops.npy')]
        self.rois_per_plane = [i.shape[0] for i in self.iscell]

    def _load_npy(self, filename, mmap_mode=None):
        ret_val = [[None]]*self.no_planes
        for i in range(self.no_planes):
            ret_val[i] = np.load(self.savepath + f'\\Plane{i}\\' + filename,
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
        plane_wise = [np.where(i[:, 0] == 1) for i in self.iscell]
        return [plane_wise[0],len(self.stat[0])+plane_wise[0]]

    @property
    def rejected_list(self):
        plane_wise = [np.where(i[:, 0] == 0) for i in self.iscell]
        return [plane_wise[0], len(self.stat[0]) + plane_wise[0]]

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
        return self.ops_inp['fs']*self.no_planes

    @staticmethod
    def write_recording(nwb_file_path):
        return s2p_nwb.read(nwb_file_path)

    # defining the abstract class enforced methods:
    def get_roi_ids(self):
        return self.roi_idx

    def get_num_rois(self):
        return self.no_rois

    def get_roi_locations(self, ROI_ids=None):
        if ROI_ids is None:
            return self.roi_locs
        else:
            ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
            ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
            return self.roi_locs[:, ROI_idx_]

    def get_num_frames(self):
        return self.num_of_frames

    def get_sampling_frequency(self):
        return self.samp_freq

    def get_traces(self, ROI_ids=None, start_frame=None, end_frame=None, name=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames() + 1
        if ROI_ids is None:
            ROI_idx_ = range(self.get_num_rois())
        else:
            ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
            ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
        if name=='fluorescence':
            return np.concatenate(self.F)[[ROI_idx_],start_frame:end_frame]
        if name=='neuropil':
            return np.concatenate(self.Fneu)[[ROI_idx_],start_frame:end_frame]
        if name=='deconvolved':
            return np.concatenate(self.spks)[[ROI_idx_],start_frame:end_frame]
        else:
            return None

    def get_image_masks(self, ROI_ids=None):
        return None

    def get_pixel_masks(self, ROI_ids=None):
        pixel_mask = [None]*self.no_rois
        c = 0
        for i in self.no_planes:
            for j in self.rois_per_plane[i]:
                c +=1
                pixel_mask[c] = np.array([self.stat[i][j]['ypix'],
                                       self.stat[i][j]['xpix'],
                                       self.stat[i][j]['lam']])
        if ROI_ids is None:
            ROI_idx_ = range(self.get_num_rois())
        else:
            ROI_idx = [np.where(np.array(i) == self.roi_idx)[0] for i in ROI_ids]
            ele = [i for i, j in enumerate(ROI_idx) if j.size == 0]
            ROI_idx_ = [j[0] for i, j in enumerate(ROI_idx) if i not in ele]
        return [pixel_mask[i] for i in ROI_idx_]

    def get_movie_framesize(self):
        return self.image_dims

    def get_movie_location(self):
        return self.fileloc

    def get_channel_names(self):
        return [f'OpticalChannel{i}' for i in range(self.no_channels)]

    def get_num_channels(self):
        return self.no_channels
