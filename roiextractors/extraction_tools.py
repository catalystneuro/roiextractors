from functools import wraps
from pathlib import Path
from typing import Union
import collections.abc
from copy import deepcopy

import numpy as np
from spikeextractors.extraction_tools import cast_start_end_frame
from tqdm import tqdm

try:
    import h5py

    HAVE_H5 = True
except ImportError:
    HAVE_H5 = False

try:
    import scipy.io as spio

    HAVE_Scipy = True
except ImportError:
    HAVE_Scipy = False

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
                    if isinstance(base[key][i], dict) and isinstance(
                        input_list_item, dict
                    ):
                        dict_recursive_update(base[key][i], input_list_item)
                    else:
                        base[key][i] = input_list_item
                else:
                    base[key].append(input_list_item)
        else:
            base[key] = val
    return base

def append_replace_dict_in_list(ls, d, compare_key, list_dict_deep_update: bool = True, remove_repeats: bool = True):
    """
    Correctly updates the list ls with the dict d.

    Cases:
    1.  If d is a dict and ls a list of dicts and ints/str, then for a given compare key, if for any element of ls
        (which is a dict) say: ls[3][compare_key] == d[compare_key], then it will dict_deep_update these instead of appending d
        to list ls. Only if compare_key is not present in any of dicts in the list ls, then d is simply appended
        to ls.
    2.  If d is of immutable types like str, int etc, the ls is either appended with d or not.
        This depends on the value of remove_repeats. If remove_repeats is False, then ls is always appended with d.
        If remove_repeats is True, then if value d is present then its not appended else it is.

    Parameters
    ----------
    ls: list
        list of a dicts or int/str or a combination. This is the object to update
    d: list/str/int
        this is the object from which ls is updated.
    compare_key: str
        name of the key for which to check the presence of dicts in ls which need dict_deep_update
    list_dict_deep_update: bool
        whether to update a dict in ls with compare_key present OR simply replace it.
    remove_repeats: bool
        keep repeated values in the updated ls

    Returns
    -------
    ls: list
        updated list
    """
    if not isinstance(ls, list):
        return d
    if isinstance(d, collections.abc.Mapping):
        indxs = np.where(
            [d.get(compare_key, None) == i[compare_key] for i in ls if isinstance(i, collections.abc.Mapping)]
        )[0]
        if len(indxs) > 0:
            for idx in indxs:
                if list_dict_deep_update:
                    ls[idx] = dict_deep_update(ls[idx], d)
                else:
                    ls[idx] = d
        else:
            ls.append(d)
    elif not (d in ls and remove_repeats):
        ls.append(d)
    return ls
    
def dict_deep_update(
    d: collections.abc.Mapping,
    u: collections.abc.Mapping,
    append_list: bool = True,
    remove_repeats: bool = True,
    copy: bool = True,
    compare_key: str = "name",
    list_dict_deep_update: bool = True,
) -> collections.abc.Mapping:
    """
    Perform an update to all nested keys of dictionary d(input) from dictionary u(updating dict).
    Parameters
    ----------
    d: dict
        dictionary to update
    u: dict
        dictionary to update from
    append_list: bool
        if the item to update is a list, whether to append the lists or replace the list in d
        eg. d = dict(key1=[1,2,3]), u = dict(key1=[3,4,5]).
        If True then updated dictionary d=dict(key1=[1,2,3,4,5]) else d=dict(key1=[3,4,5])
    remove_repeats: bool
        for updating list in d[key] with list in u[key]: if true then remove repeats: list(set(ls))
    copy: bool
        whether to deepcopy the input dict d
    compare_key: str
        the key that is used to compare dicts (and perform update op) and update d[key] when it is a list if dicts.
        example:
            >>> d = {'input': [{'name':'timeseries1', 'desc':'desc1 of d', 'starting_time':0.0}, {'name':'timeseries2', 'desc':'desc2'}]}
            >>> u = ['input': {'name':'timeseries1', 'desc':'desc2 of u', 'unit':'n.a.'}]
            >>> # if compre_key='name' output is below
            >>> output = ['input': {'name':'timeseries1', 'desc':'desc2 of u', 'starting_time':0.0, 'unit':'n.a.'}, {'name':'timeseries2', 'desc':'desc2'}]
            >>> # else the output is:
            >>> # dict with the same key will be updated instead of being appended to the list
            >>> output = ['input': {'name':'timeseries1', 'desc':'desc1 of d', 'starting_time': 0.0}, {'name':'timeseries2', 'desc':'desc2'}, {'name':'timeseries1', 'desc':'desc2 of u', 'unit':'n.a.'}]
    list_dict_deep_update: bool
        for back compatibility, if False, this would work as before:
        example: if True then for the compare_key example, the output would be:
            >>> output = ['input': {'name':'timeseries1', 'desc':'desc2 of u', 'starting_time':0.0, 'unit':'n.a.'}, {'name':'timeseries2', 'desc':'desc2'}]
            >>> # if False:
            >>> output = ['input': {'name':'timeseries1', 'desc':'desc2 of u', 'starting_time':0.0}, {'name':'timeseries2', 'desc':'desc2'}]# unit key is absent since its a replacement
    Returns
    -------
    d: dict
        return the updated dictionary
    """
    if not isinstance(d, collections.abc.Mapping):
        warnings.warn("input to update should be a dict, returning output")
        return u
    if copy:
        d = deepcopy(d)
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_deep_update(d.get(k, None), v, append_list=append_list, remove_repeats=remove_repeats)
        elif append_list and isinstance(v, list):
            for vv in v:
                d[k] = append_replace_dict_in_list(d.get(k, []), vv, compare_key, list_dict_deep_update, remove_repeats)
        else:
            d[k] = v
    return d


def _pixel_mask_extractor(image_mask_, _roi_ids):
    """An alternative data format for storage of image masks.
    Returns
    -------
    pixel_mask: numpy array
        Total pixels X 4 size. Col 1 and 2 are x and y location of the mask
        pixel, Col 3 is the weight of that pixel, Col 4 is the ROI index.
    """
    pixel_mask_list = []
    for i, roiid in enumerate(_roi_ids):
        image_mask = np.array(image_mask_[:, :, i])
        _locs = np.where(image_mask > 0)
        _pix_values = image_mask[image_mask > 0]
        pixel_mask_list.append(np.vstack((_locs[0], _locs[1], _pix_values)).T)
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
    image_mask = np.zeros(list(image_shape) + [len(_roi_ids)])
    for no, rois in enumerate(_roi_ids):
        for y, x, wt in pixel_mask[rois]:
            image_mask[int(y), int(x), no] = wt
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
            assert np.all(
                frame_idxs < imaging.get_num_frames()
            ), "'frame_idxs' exceed number of frames"
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
                raise Exception(
                    f"'start_frame' exceeds number of frames {imaging.get_num_frames()}!"
                )
            elif start_frame < 0:
                start_frame = imaging.get_num_frames() + start_frame
        else:
            start_frame = 0
        if end_frame is not None:
            if end_frame > imaging.get_num_frames():
                raise Exception(
                    f"'end_frame' exceeds number of frames {imaging.get_num_frames()}!"
                )
            elif end_frame < 0:
                end_frame = imaging.get_num_frames() + end_frame
        else:
            end_frame = imaging.get_num_frames()
        assert (
            end_frame - start_frame > 0
        ), "'start_frame' must be less than 'end_frame'!"

        start_frame, end_frame = cast_start_end_frame(start_frame, end_frame)
        channel = int(channel)
        get_videos_correct_arg = func(
            imaging, start_frame=start_frame, end_frame=end_frame, channel=channel
        )

        return get_videos_correct_arg

    return corrected_args


def write_to_h5_dataset_format(
    imaging,
    dataset_path,
    save_path=None,
    file_handle=None,
    dtype=None,
    chunk_size=None,
    chunk_mb=1000,
    verbose=False,
):
    """Saves the video of an imaging extractor in an h5 dataset.

    Parameters
    ----------
    imaging: ImagingExtractor
        The imaging extractor object to be saved in the .h5 filr
    dataset_path: str
        Path to dataset in h5 file (e.g. '/dataset')
    save_path: str
        The path to the file.
    file_handle: file handle
        The file handle to dump data. This can be used to append data to an header. In case file_handle is given,
        the file is NOT closed after writing the binary data.
    dtype: dtype
        Type of the saved data. Default float32.
    chunk_size: None or int
        Number of chunks to save the file in. This avoid to much memory consumption for big files.
        If None and 'chunk_mb' is given, the file is saved in chunks of 'chunk_mb' Mb (default 500Mb)
    chunk_mb: None or int
        Chunk size in Mb (default 1000Mb)
    verbose: bool
        If True, output is verbose (when chunks are used)
    """
    assert HAVE_H5, "To write to h5 you need to install h5py: pip install h5py"
    assert (
        save_path is not None or file_handle is not None
    ), "Provide 'save_path' or 'file handle'"

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix == "":
            # when suffix is already raw/bin/dat do not change it.
            save_path = save_path.parent / (save_path.name + ".h5")

    num_channels = imaging.get_num_channels()
    num_frames = imaging.get_num_frames()
    size_x, size_y = imaging.get_image_size()

    if file_handle is not None:
        assert isinstance(file_handle, h5py.File)
    else:
        file_handle = h5py.File(save_path, "w")

    if dtype is None:
        dtype_file = imaging.get_dtype()
    else:
        dtype_file = dtype

    dset = file_handle.create_dataset(
        dataset_path, shape=(num_channels, num_frames, size_x, size_y), dtype=dtype_file
    )

    # set chunk size
    if chunk_size is not None:
        chunk_size = int(chunk_size)
    elif chunk_mb is not None:
        n_bytes = np.dtype(imaging.get_dtype()).itemsize
        max_size = int(chunk_mb * 1e6)  # set Mb per chunk
        chunk_size = max_size // (size_x * size_y * n_bytes)

    # writ one channel at a time
    for ch in range(num_channels):
        if chunk_size is None:
            video = imaging.get_video(channel=ch)
            if dtype is not None:
                video = video.astype(dtype_file)
            dset[ch, ...] = np.squeeze(video)
        else:
            chunk_start = 0
            # chunk size is not None
            n_chunk = num_frames // chunk_size
            if num_frames % chunk_size > 0:
                n_chunk += 1
            if verbose:
                chunks = tqdm(range(n_chunk), ascii=True, desc="Writing to .h5 file")
            else:
                chunks = range(n_chunk)
            for i in chunks:
                video = imaging.get_video(
                    start_frame=i * chunk_size,
                    end_frame=min((i + 1) * chunk_size, num_frames),
                    channel=ch,
                )
                chunk_frames = np.squeeze(video).shape[0]
                if dtype is not None:
                    video = video.astype(dtype_file)
                dset[ch, chunk_start: chunk_start + chunk_frames, ...] = np.squeeze(
                    video
                )
                chunk_start += chunk_frames

    if save_path is not None:
        file_handle.close()
    return save_path


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
    im = ax.imshow(im0, interpolation="none", aspect="auto", vmin=0, vmax=1)
    interval = 1 / imaging.get_sampling_frequency() * 1000
    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=imaging.get_num_frames(),
        fargs=(imaging, im, ax),
        interval=interval,
        blit=False,
    )
    return anim


def check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    assert HAVE_Scipy, "To write to h5 you need to install scipy: pip install scipy"
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = todict(dict[key])
    return dict


def todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = todict(elem)
        else:
            dict[strg] = elem
    return dict
