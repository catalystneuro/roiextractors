import numpy as np
from typing import Union
from pathlib import Path
from functools import wraps
from spikeextractors.extraction_tools import cast_start_end_frame

ArrayType = Union[list, np.array]
PathType = Union[str, Path]
NumpyArray = Union[np.array, np.memmap]
DtypeType = [str, np.dtype]

MAX_FRAMES = 10000


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
    def corrected_args(*args, **kwargs):
        # parse args and kwargs
        assert len(args) >= 2, "'get_frames' requires 'frame_idxs' as first argument"
        if len(args) == 2:
            imaging = args[0]
            frame_idxs = args[1]
            channel = kwargs.get('channel', 0)
        elif len(args) == 3:
            imaging = args[0]
            frame_idxs = args[1]
            channel = args[2]
        else:
            raise Exception("Too many arguments!")

        channel = int(channel)
        if isinstance(frame_idxs, (int, np.integer)):
            frame_idxs = [frame_idxs]
        frame_idxs = np.array(frame_idxs)
        assert np.all(frame_idxs < imaging.get_num_frames()), "'frame_idxs' exceed number of frames"
        kwargs['channel'] = channel
        get_frames_correct_arg = func(args[0], frame_idxs, channel)

        if len(frame_idxs) == 1:
            return get_frames_correct_arg[0]
        else:
            return get_frames_correct_arg
    return corrected_args


def check_get_videos_args(func):
    @wraps(func)
    def corrected_args(*args, **kwargs):
        # parse args and kwargs
        if len(args) == 1:
            imaging = args[0]
            start_frame = kwargs.get('start_frame', None)
            end_frame = kwargs.get('end_frame', None)
            channel = kwargs.get('channel', 0)
        elif len(args) == 2:
            imaging = args[0]
            start_frame = args[1]
            end_frame = kwargs.get('end_frame', None)
            channel = kwargs.get('channel', 0)
        elif len(args) == 3:
            imaging = args[0]
            start_frame = args[1]
            end_frame = args[2]
            channel = kwargs.get('channel', 0)
        elif len(args) == 4:
            recording = args[0]
            start_frame = args[1]
            end_frame = args[2]
            channel = args[3]
        else:
            raise Exception("Too many arguments!")

        if start_frame is not None:
            if start_frame < 0:
                start_frame = imaging.get_num_frames() + start_frame
        else:
            start_frame = 0
        if end_frame is not None:
            if end_frame > imaging.get_num_frames():
                print("'end_frame' set to", imaging.get_num_frames())
                end_frame = imaging.get_num_frames()
            elif end_frame < 0:
                end_frame = imaging.get_num_frames() + end_frame
        else:
            end_frame = imaging.get_num_frames()
        assert end_frame - start_frame > 0, "'start_frame' must be less than 'end_frame'!"

        if end_frame - start_frame > MAX_FRAMES:
            print(f"Warning: retrieving more than {MAX_FRAMES} frames. This might cause memory errors for "
                  f"some extractors")

        start_frame, end_frame = cast_start_end_frame(start_frame, end_frame)
        kwargs['start_frame'] = start_frame
        kwargs['end_frame'] = end_frame
        kwargs['channel'] = channel

        # pass recording as arg and rest as kwargs
        get_videos_correct_arg = func(args[0], **kwargs)

        return get_videos_correct_arg
    return corrected_args
