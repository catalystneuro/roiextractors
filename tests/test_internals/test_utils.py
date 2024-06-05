import os

from roiextractors.utils import match_paths
from tempfile import TemporaryDirectory


def test_match_paths():
    # create temporary directory
    with TemporaryDirectory() as tmpdir:
        # create temporary files
        files = [
            "split_1.tif",
            "split_2.tif",
            "split_3.tif",
            "split_4.tif",
            "split_5.tif",
            "split_6.tif",
            "split_7.tif",
            "split_8.tif",
            "split_9.tif",
            "split_10.tif",
        ]
        for file in files:
            with open(os.path.join(tmpdir, file), "w") as f:
                f.write("")

        # test match_paths
        out = match_paths(tmpdir, "split_{split:d}.tif")
        assert list(out.keys()) == [os.path.join(tmpdir, x) for x in files]
        assert list([x["split"] for x in out.values()]) == list(range(1, 11))


def test_match_paths_sub_select():
    # create temporary directory
    with TemporaryDirectory() as tmpdir:
        # create temporary files
        files = [
            "chanA_split_1.tif",
            "chanA_split_2.tif",
            "chanA_split_3.tif",
            "chanA_split_4.tif",
            "chanA_split_5.tif",
            "chanB_split_1.tif",
            "chanB_split_2.tif",
            "chanB_split_3.tif",
            "chanB_split_4.tif",
            "chanB_split_5.tif",
        ]
        for file in files:
            with open(os.path.join(tmpdir, file), "w") as f:
                f.write("")

        # test match_paths
        out = match_paths(tmpdir, "chanA_split_{split:d}.tif")
        assert list(out.keys()) == [os.path.join(tmpdir, x) for x in files[:5]]
        assert list([x["split"] for x in out.values()]) == list(range(1, 6))
