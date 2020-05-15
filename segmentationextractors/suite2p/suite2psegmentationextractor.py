import numpy as np
from segmentationextractors import SegmentationExtractor
from suite2p import run_s2p
from skimage.external.tifffile import imread


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
        self.op = op
        self.no_channels = op['nchannels']
        self.no_planes = op['nplanes']
        self.stat = self._load_npy('stat.npy')
        self.F = self._load_npy('F.npy')
        self.Fneu = self._load_npy('Fneu.npy')
        self.spks = self._load_npy('spks.npy')
        self.iscell = self._load_npy('iscell.npy')
        self.ops = self._load_npy('ops.npy')

    def _load_npy(self, filename):
        ret_val = [[None]]*self.no_planes
        for i in range(self.no_planes):
            ret_val[i] = np.load(self.savepath + f'\\Plane{i}\\' + filename, mmap_mode='r')
        return ret_val

    @property
    def image_dims(self):
        return [self.ops['Lx'], self.ops['Ly']]

    @property
    def no_rois(self):
        return [len(i) for i in self.stat]

    @property
    def roi_idx(self):
        return [i for i in range(self.no_rois)]

    @property
    def accepted_list(self):
        return [np.where(i[:, 0] == 1) for i in self.iscell]

    @property
    def rejected_list(self):
        return [np.where(i[:, 0] == 0) for i in self.iscell]

    @property
    def roi_locs(self):
        return [[j['med'] for j in i] for i in self.stat]

    @property
    def num_of_frames(self):
        return self.ops['nframes']

    @property
    def samp_freq(self):
        return self.ops['fs']

    @staticmethod
    def write_recording(segmentation_object, savepath):
        raise NotImplementedError

    # defining the abstract class enformed methods:
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

    def get_traces(self, ROI_ids=None, start_frame=None, end_frame=None):
        pass  # TODO

    def get_image_masks(self, ROI_ids=None):
        pass  # TODO

    def get_pixel_masks(self, ROI_ids=None):
        pass  # TODO

    def get_movie_framesize(self):
        return self.image_dims

    def get_movie_location(self):
        return self.fileloc

    def get_channel_names(self):
        return list(range(self.no_channels))

    def get_num_channels(self):
        return self.no_channels
