import numpy as np
from typing import Union
from pathlib import Path
from functools import wraps
from spikeextractors.extraction_tools import cast_start_end_frame

ArrayType = Union[list, np.array]
PathType = Union[str, Path]
NumpyArray = Union[np.array, np.memmap]
DtypeType = Union[str, np.dtype]
IntType = Union[int, np.integer]
FloatType = Union[float, np.float]


def dict_recursive_update(base, input_):
    for key, val in input_.items():
        if key in base and isinstance(val, dict) and isinstance(base[key], dict):
            dict_recursive_update(base[key], val)
        elif key in base and isinstance(val, list) and isinstance(base[key], list):
            for i, input_list_item in enumerate(val):
                if len(base[key]) < i:
                    if isinstance(base[key][i], dict) and isinstance(input_list_item, dict):
                        dict_recursive_update(base[key][i], input_list_item)
                    else:
                        base[key][i] = input_list_item
                else:
                    base[key].append(input_list_item)
        else:
            base[key] = val

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


def check_get_frames_args(func):
    @wraps(func)
    def corrected_args(imaging, frame_idxs, channel=0):
        channel = int(channel)
        if isinstance(frame_idxs, (int, np.integer)):
            frame_idxs = [frame_idxs]

        if not isinstance(frame_idxs, slice):
            frame_idxs = np.array(frame_idxs)
        assert np.all(frame_idxs < imaging.get_num_frames()), "'frame_idxs' exceed number of frames"
        get_frames_correct_arg = func(imaging, frame_idxs, channel)

        if len(frame_idxs) == 1:
            return get_frames_correct_arg[0]
        else:
            return get_frames_correct_arg
    return corrected_args


def check_get_videos_args(func):
    @wraps(func)
    def corrected_args(imaging, start_frame=None, end_frame=None, channel=0):
        if start_frame is not None:
            if start_frame > imaging.get_num_frames():
                raise Exception(f"'start_frame' exceeds number of frames {imaging.get_num_frames()}!")
            elif start_frame < 0:
                start_frame = imaging.get_num_frames() + start_frame
        else:
            start_frame = 0
        if end_frame is not None:
            if end_frame > imaging.get_num_frames():
                raise Exception(f"'end_frame' exceeds number of frames {imaging.get_num_frames()}!")
            elif end_frame < 0:
                end_frame = imaging.get_num_frames() + end_frame
        else:
            end_frame = imaging.get_num_frames()
        assert end_frame - start_frame > 0, "'start_frame' must be less than 'end_frame'!"

        start_frame, end_frame = cast_start_end_frame(start_frame, end_frame)
        channel = int(channel)
        get_videos_correct_arg = func(imaging, start_frame=start_frame, end_frame=end_frame, channel=channel)

        return get_videos_correct_arg
    return corrected_args

# TODO will be moved eventually, but for now it's very handy :)
def show_video(imaging, ax=None):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    def animate_func(i, imaging, im, ax):
        ax.set_title(f"{i}")
        im.set_array(imaging.get_frames(i))
        return [im]

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    im0 = imaging.get_frames(0)
    im = ax.imshow(im0, interpolation='none', aspect='auto', vmin=0, vmax=1)
    interval = 1 / imaging.get_sampling_frequency() * 1000
    anim = animation.FuncAnimation(fig, animate_func, frames=imaging.get_num_frames(), fargs=(imaging, im, ax),
                                   interval=interval, blit=False)
    return anim
