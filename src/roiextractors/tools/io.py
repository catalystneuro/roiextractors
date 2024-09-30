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
