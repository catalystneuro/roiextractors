import glob
import os

from parse import parse


def match_paths(base, pattern, sort=True):
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
