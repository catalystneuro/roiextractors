from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py


def check_keys(dict_: dict) -> dict:
    """Check keys of dictionary for mat-objects.

    Checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries.

    Parameters
    ----------
    dict_: dict
        Dictionary to check.

    Returns
    -------
    dict: dict
        Dictionary with mat-objects converted to nested dictionaries.

    Raises
    ------
    AssertionError
        If scipy is not installed.
    """
    from scipy.io.matlab.mio5_params import mat_struct

    for key in dict_:
        if isinstance(dict_[key], mat_struct):
            dict_[key] = todict(dict_[key])
    return dict_


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
    """Save the video of an imaging extractor in an h5 dataset.

    Parameters
    ----------
    imaging: ImagingExtractor
        The imaging extractor object to be saved in the .h5 file
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

    Returns
    -------
    save_path: str
        The path to the file.

    Raises
    ------
    AssertionError
        If neither 'save_path' nor 'file_handle' are given.
    """
    assert save_path is not None or file_handle is not None, "Provide 'save_path' or 'file handle'"

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
    dset = file_handle.create_dataset(dataset_path, shape=(num_channels, num_frames, size_x, size_y), dtype=dtype_file)

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
                dset[ch, chunk_start : chunk_start + chunk_frames, ...] = np.squeeze(video)
                chunk_start += chunk_frames
    if save_path is not None:
        file_handle.close()
    return save_path


def todict(matobj):
    """Recursively construct nested dictionaries from matobjects.

    Parameters
    ----------
    matobj: mat_struct
        Matlab object to convert to nested dictionary.

    Returns
    -------
    dict: dict
        Dictionary with mat-objects converted to nested dictionaries.
    """
    from scipy.io.matlab.mio5_params import mat_struct

    dict_ = {}
    from scipy.io.matlab.mio5_params import mat_struct

    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, mat_struct):
            dict_[strg] = todict(elem)
        else:
            dict_[strg] = elem
    return dict_
