from functools import wraps
import numpy as np


def check_get_frames_args(func):
    """Check the arguments of the get_frames function.

    This decorator allows the get_frames function to be queried with either
    an integer, slice or an array and handles a common return. [I think that np.take can be used instead of this]

    Parameters
    ----------
    func: function
        The get_frames function.

    Returns
    -------
    corrected_args: function
        The get_frames function with corrected arguments.

    Raises
    ------
    AssertionError
        If 'frame_idxs' exceed the number of frames.
    """

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


def _cast_start_end_frame(start_frame, end_frame):
    """Cast start and end frame to int or None.

    Parameters
    ----------
    start_frame: int, float, None
        The start frame.
    end_frame: int, float, None
        The end frame.

    Returns
    -------
    start_frame: int, None
        The start frame.
    end_frame: int, None
        The end frame.

    Raises
    ------
    ValueError
        If start_frame is not an int, float or None.
    ValueError
        If end_frame is not an int, float or None.
    """
    if isinstance(start_frame, float):
        start_frame = int(start_frame)
    elif isinstance(start_frame, (int, np.integer, type(None))):
        start_frame = start_frame
    else:
        raise ValueError("start_frame must be an int, float (not infinity), or None")
    if isinstance(end_frame, float) and np.isfinite(end_frame):
        end_frame = int(end_frame)
    elif isinstance(end_frame, (int, np.integer, type(None))):
        end_frame = end_frame
    # else end_frame is infinity (accepted for get_unit_spike_train)
    if start_frame is not None:
        start_frame = int(start_frame)
    if end_frame is not None and np.isfinite(end_frame):
        end_frame = int(end_frame)
    return start_frame, end_frame


def check_get_videos_args(func):
    """Check the arguments of the get_videos function.

    This decorator allows the get_videos function to be queried with either
    an integer or slice and handles a common return.

    Parameters
    ----------
    func: function
        The get_videos function.

    Returns
    -------
    corrected_args: function
        The get_videos function with corrected arguments.

    Raises
    ------
    AssertionError
        If 'start_frame' exceeds the number of frames.
    AssertionError
        If 'end_frame' exceeds the number of frames.
    AssertionError
        If 'start_frame' is greater than 'end_frame'.
    """

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

        start_frame, end_frame = _cast_start_end_frame(start_frame, end_frame)
        channel = int(channel)
        get_videos_correct_arg = func(imaging, start_frame=start_frame, end_frame=end_frame, channel=channel)

        return get_videos_correct_arg

    return corrected_args
