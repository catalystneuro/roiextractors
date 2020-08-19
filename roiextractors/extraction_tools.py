import numpy as np
from typing import Union
from pathlib import Path

ArrayType = Union[list, np.array]
PathType = Union[str, Path]
NumpyArray = Union[np.array, np.memmap]
DtypeType = [str, np.dtype]


def _pixel_mask_extractor(image_mask_, _roi_ids):
    '''An alternative data format for storage of image masks.
    Returns
    -------
    pixel_mask: numpy array
        Total pixels X 4 size. Col 1 and 2 are x and y location of the mask
        pixel, Col 3 is the weight of that pixel, Col 4 is the ROI index.
    '''
    pixel_mask_list = []
    for i, roiid in enumerate(_roi_ids):
        image_mask = np.array(image_mask_[:, :, i])
        _locs = np.where(image_mask > 0)
        _pix_values = image_mask[image_mask > 0]
        pixel_mask_list.append(np.vstack((_locs[0], _locs[1],_pix_values)).T)
    return pixel_mask_list

def _image_mask_extractor(pixel_mask, _roi_ids, image_shape):
    """
    Converts a pixel mask to image mask
    
    Parameters
    ----------
    pixel_mask: list
        list of pixel masks (no pixels X 3)
    _roi_ids: list
    image_shape: array_like
    
    Returns
    -------
    image_mask: np.ndarray
    """
    image_mask = np.zeros(list(image_shape)+[len(_roi_ids)])
    for no, rois in enumerate(_roi_ids):
        for x, y, wt in pixel_mask[rois]:
            image_mask[int(x), int(y), no] = wt
    return image_mask

def get_video_shape(video):
    if len(video.shape) == 3:
        # 1 channel
        num_channels = 1
        num_frames, size_x, size_y = video.shape
    else:
        num_channels, num_frames, size_x, size_y = video.shape

    return num_channels, num_frames, size_x, size_y
