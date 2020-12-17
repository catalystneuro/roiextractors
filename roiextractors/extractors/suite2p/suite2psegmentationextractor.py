import os
import shutil
from pathlib import Path

import numpy as np

from ...extraction_tools import PathType, IntType
from ...extraction_tools import _image_mask_extractor
from ...multisegmentationextractor import MultiSegmentationExtractor
from ...segmentationextractor import SegmentationExtractor


class Suite2pSegmentationExtractor(SegmentationExtractor):
    extractor_name = 'Suite2pSegmentationExtractor'
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, file_path: PathType, combined: bool = False, plane_no: IntType = 0):
        """
        Creating SegmentationExtractor object out of suite 2p data type.
        Parameters
        ----------
        file_path: str
            ~/suite2p folder location on disk
        combined: bool
            if the plane is a combined plane as in the Suite2p pipeline
        plane_no: int
            the plane for which to extract segmentation for.
        """
        SegmentationExtractor.__init__(self)
        self.combined = combined
        self.plane_no = plane_no
        self.file_path = file_path
        self.stat = self._load_npy('stat.npy')
        self._roi_response_raw = self._load_npy('F.npy', mmap_mode='r')
        self._roi_response_neuropil = self._load_npy('Fneu.npy', mmap_mode='r')
        self._roi_response_deconvolved = self._load_npy('spks.npy', mmap_mode='r')
        self.iscell = self._load_npy('iscell.npy', mmap_mode='r')
        self.ops = self._load_npy('ops.npy').item()
        self._channel_names = [f'OpticalChannel{i}' for i in range(self.ops['nchannels'])]
        self._sampling_frequency = self.ops['fs'] * [2 if self.combined else 1][0]
        self._raw_movie_file_location = self.ops['filelist'][0]
        self.image_masks = self.get_roi_image_masks()
        self._image_correlation = self._summary_image_read('Vcorr')
        self._image_mean = self._summary_image_read('meanImg')

    def _load_npy(self, filename, mmap_mode=None):
        fpath = os.path.join(self.file_path, f'plane{self.plane_no}', filename)
        return np.load(fpath, mmap_mode=mmap_mode, allow_pickle=mmap_mode is None)

    def get_accepted_list(self):
        return list(np.where(self.iscell[:, 0] == 1)[0])

    def get_rejected_list(self):
        return list(np.where(self.iscell[:, 0] == 0)[0])

    def _summary_image_read(self, bstr='meanImg'):
        img = None
        if bstr in self.ops:
            if bstr == 'Vcorr' or bstr == 'max_proj':
                img = np.zeros((self.ops['Ly'], self.ops['Lx']), np.float32)
                img[self.ops['yrange'][0]:self.ops['yrange'][-1],
                self.ops['xrange'][0]:self.ops['xrange'][-1]] = self.ops[bstr]
            else:
                img = self.ops[bstr]
        return img

    @property
    def roi_locations(self):
        return np.array([j['med'] for j in self.stat]).T.astype(int)

    @staticmethod
    def write_segmentation(segmentation_object: SegmentationExtractor, save_path, overwrite=True):
        save_path = Path(save_path)
        assert not save_path.is_file(), "'save_path' must be a folder"
        if save_path.is_dir():
            if len(list(save_path.glob('*'))) > 0 and not overwrite:
                raise FileExistsError("The specified folder is not empty! Use overwrite=True to overwrite it.")
            else:
                shutil.rmtree(str(save_path))
        if isinstance(segmentation_object, MultiSegmentationExtractor):
            segext_objs = segmentation_object.segmentations
            for plane_num, segext_obj in enumerate(segext_objs):
                save_path_plane = save_path / f'plane{plane_num}'
                Suite2pSegmentationExtractor.write_segmentation(segext_obj, save_path_plane)
        if not save_path.is_dir():
            save_path.mkdir(parents=True)
        if 'plane' not in save_path.stem:
            save_path = save_path/'plane0'

        # saving traces:
        if segmentation_object.get_traces(name='raw') is not None:
            np.save(save_path / 'F.npy', segmentation_object.get_traces(name='raw'))
        if segmentation_object.get_traces(name='neuropil') is not None:
            np.save(save_path / 'Fneu.npy', segmentation_object.get_traces(name='neuropil'))
        if segmentation_object.get_traces(name='deconvolved') is not None:
            np.save(save_path / 'spks.npy', segmentation_object.get_traces(name='deconvolved'))
        # save stat
        stat = np.zeros(segmentation_object.get_num_rois(), 'O')
        roi_locs = segmentation_object.roi_locations.T
        pixel_masks = segmentation_object.get_roi_pixel_masks(roi_ids=range(segmentation_object.get_num_rois()))
        for no, i in enumerate(stat):
            stat[no] = {'med': roi_locs[no, :].tolist(),
                        'ypix': pixel_masks[no][:, 0],
                        'xpix': pixel_masks[no][:, 1],
                        'lam': pixel_masks[no][:, 2]}
        np.save(save_path / 'stat.npy', stat)
        # saving iscell
        iscell = np.ones([segmentation_object.get_num_rois(), 2])
        iscell[segmentation_object.get_rejected_list(), 0] = 0
        np.save(save_path / 'iscell.npy', iscell)
        # saving ops
        ops = dict(nframes=segmentation_object.get_num_frames(),
                   Lx=segmentation_object.get_image_size()[1],
                   Ly=segmentation_object.get_image_size()[0],
                   xrange=[0, segmentation_object.get_image_size()[1]],
                   yrange=[0, segmentation_object.get_image_size()[0]],
                   fs=segmentation_object.get_sampling_frequency(),
                   nchannels=1,
                   meanImg=segmentation_object.get_image('mean'),
                   Vcorr=segmentation_object.get_image('correlation'))
        if getattr(segmentation_object, '_raw_movie_file_location', None):
            ops.update(dict(filelist=[segmentation_object._raw_movie_file_location]))
        else:
            ops.update(dict(filelist=[None]))
        np.save(save_path / 'ops.npy', ops)

    # defining the abstract class enforced methods:
    def get_roi_ids(self):
        return list(range(self.get_num_rois()))

    def get_roi_image_masks(self, roi_ids=None):
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return _image_mask_extractor(self.get_roi_pixel_masks(roi_ids=roi_idx_), list(range(len(roi_idx_))),
                                     self.get_image_size())

    def get_roi_pixel_masks(self, roi_ids=None):
        pixel_mask = []
        for i in range(self.get_num_rois()):
            pixel_mask.append(np.vstack([self.stat[i]['ypix'],
                                         self.stat[i]['xpix'],
                                         self.stat[i]['lam']]).T)
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return [pixel_mask[i] for i in roi_idx_]

    def get_image_size(self):
        return [self.ops['Lx'], self.ops['Ly']]
