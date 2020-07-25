import numpy as np
from typing import Union
from pathlib import Path

ArrayType = Union[list, np.array]
PathType = Union[str, Path]
NumpyArray = Union[np.array, np.memmap]
DtypeType = [str, np.dtype]

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
        _np_raw_images_trans = np.array(_raw_images_trans[:, :, i])
        _locs = np.where(_np_raw_images_trans > 0)
        _pix_values = _np_raw_images_trans[_np_raw_images_trans > 0]
        temp = np.append(temp, np.concatenate(
            (_locs[0].reshape([1, np.size(_locs[0])]),
             _locs[1].reshape([1, np.size(_locs[1])]),
             _pix_values.reshape([1, np.size(_locs[1])]),
             roiid * np.ones([1, np.size(_locs[1])]))).T, axis=0)
    return temp[1::, :]
