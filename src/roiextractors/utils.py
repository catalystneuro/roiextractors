"""Utility functions for the ROIExtractors package."""

import glob
import os
from typing import Dict, Any

from parse import parse


def match_paths(base: str, pattern: str, sort=True) -> Dict[str, Dict[str, Any]]:
    """
    Match paths in a directory to a pattern.

    Parameters
    ----------
    base: str
        The base directory to search in.
    pattern: str
        The f-string pattern to match the paths to.
    sort: bool, default=True
        Whether to sort the output by the values of the named groups in the pattern.

    Returns
    -------
    dict
    """
    full_pattern = os.path.join(base, pattern)
    paths = glob.glob(os.path.join(base, "*"))
    out = {}
    for path in paths:
        parsed = parse(full_pattern, path)
        if parsed is not None:
            out[path] = parsed.named

    if sort:
        out = dict(sorted(out.items(), key=lambda item: tuple(item[1].values())))

    return out
