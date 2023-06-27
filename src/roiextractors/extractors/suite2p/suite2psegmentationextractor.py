import shutil
from pathlib import Path
from typing import Optional
import warnings

import numpy as np

from ...extraction_tools import PathType, IntType
from ...extraction_tools import _image_mask_extractor
from ...multisegmentationextractor import MultiSegmentationExtractor
from ...segmentationextractor import SegmentationExtractor


class Suite2pSegmentationExtractor(SegmentationExtractor):
    extractor_name = "Suite2pSegmentationExtractor"
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = "file"
    installation_mesg = ""  # error message when not installed

    def __init__(
        self,
        folder_path: Optional[PathType] = None,
        combined: bool = False,
        plane_no: IntType = 0,
        allow_incomplete_import: bool = False,
        warn_missing_files: bool = True,
        file_path: Optional[PathType] = None,
    ):
        """
        Creating SegmentationExtractor object out of suite 2p data type.
        Parameters
        ----------
        folder_path: str or Path
            ~/suite2p folder location on disk
        combined: bool
            if the plane is a combined plane as in the Suite2p pipeline
        plane_no: int
            the plane for which to extract segmentation for.
        allow_incomplete_import: bool
            If True, will not raise an error if the file is incomplete.
        warn_missing_files: bool
            If True, will raise a warning if a file is incomplete and
             allow_incomplete_import is True.
        file_path: str or Path [Deprecated]
            ~/suite2p folder location on disk

        """
        from warnings import warn

        if file_path is not None:
            warning_string = (
                "The keyword argument 'file_path' is being deprecated on or after August, 2022 in favor of 'folder_path'. "
                "'folder_path' takes precence over 'file_path'."
            )
            warn(
                message=warning_string,
                category=DeprecationWarning,
            )
            folder_path = file_path if folder_path is None else folder_path

        SegmentationExtractor.__init__(self)
        self.combined = combined
        self.plane_no = plane_no
        self.folder_path = Path(folder_path)


        def try_load_npy(filename, mmap_mode=None, fn_transform=lambda x: x):
            """
            This function allows for incomplete import of files.
            """
            try:
                return fn_transform(self._load_npy(filename, mmap_mode=mmap_mode))
            except FileNotFoundError:
                if allow_incomplete_import:
                    warnings.warn(f"File {filename} not found.") if warn_missing_files else None
                    return None
                else:
                    raise FileNotFoundError(f"File {filename} not found.")

        self.stat = try_load_npy("stat.npy")
        self._roi_response_raw = try_load_npy("F.npy", mmap_mode="r", fn_transform=lambda x: x.T)
        self._roi_response_neuropil = try_load_npy("Fneu.npy", mmap_mode="r", fn_transform=lambda x: x.T)
        self._roi_response_deconvolved = try_load_npy("spks.npy", mmap_mode="r", fn_transform=lambda x: x.T)
        self.iscell = try_load_npy("iscell.npy", mmap_mode="r")
        self.ops = try_load_npy("ops.npy", fn_transform=lambda x: x.item())

        self._channel_names = [f"OpticalChannel{i}" for i in range(self.ops["nchannels"])]
        self._sampling_frequency = self.ops["fs"] * [2 if self.combined else 1][0]
        self._raw_movie_file_location = self.ops.get("filelist", [None])[0]
        self._image_correlation = self._summary_image_read("Vcorr")
        self._image_mean = self._summary_image_read("meanImg")

    def _attempt_load_npy(self, filename, mmap_mode=None) -> np.ndarray | None:
        """Attempt to load the filename located in the current `plane_no` subfolder; return None if file is missing."""

        file_path = self.folder_path / f"plane{self.plane_no}" / filename
        if not file_path.exists():
            return
        return np.load(file_path, mmap_mode=mmap_mode, allow_pickle=mmap_mode is None)

    def get_accepted_list(self):
        return list(np.where(self.iscell[:, 0] == 1)[0])

    def get_rejected_list(self):
        return list(np.where(self.iscell[:, 0] == 0)[0])

    def _summary_image_read(self, bstr="meanImg"):
        img = None
        if bstr in self.ops:
            if bstr == "Vcorr" or bstr == "max_proj":
                img = np.zeros((self.ops["Ly"], self.ops["Lx"]), np.float32)
                img[
                    (self.ops["Ly"] - self.ops["yrange"][-1]) : (self.ops["Ly"] - self.ops["yrange"][0]),
                    self.ops["xrange"][0] : self.ops["xrange"][-1],
                ] = self.ops[bstr]
            else:
                img = self.ops[bstr]
        return img

    @property
    def roi_locations(self):
        return np.array([j["med"] for j in self.stat]).T.astype(int)

    def get_roi_image_masks(self, roi_ids=None):
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return _image_mask_extractor(
            self.get_roi_pixel_masks(roi_ids=roi_idx_),
            list(range(len(roi_idx_))),
            self.get_image_size(),
        )

    def get_roi_pixel_masks(self, roi_ids=None):
        pixel_mask = []
        for i in range(self.get_num_rois()):
            pixel_mask.append(
                np.vstack(
                    [
                        self.stat[i]["ypix"],
                        self.stat[i]["xpix"],
                        self.stat[i]["lam"],
                    ]
                ).T
            )
        if roi_ids is None:
            roi_idx_ = range(self.get_num_rois())
        else:
            roi_idx = [np.where(np.array(i) == self.get_roi_ids())[0] for i in roi_ids]
            ele = [i for i, j in enumerate(roi_idx) if j.size == 0]
            roi_idx_ = [j[0] for i, j in enumerate(roi_idx) if i not in ele]
        return [pixel_mask[i] for i in roi_idx_]

    def get_image_size(self):
        return [self.ops["Ly"], self.ops["Lx"]]

    @staticmethod
    def write_segmentation(segmentation_object: SegmentationExtractor, save_path: PathType, overwrite=True):
        save_path = Path(save_path)
        assert not save_path.is_file(), "'save_path' must be a folder"

        if save_path.is_dir():
            if len(list(save_path.glob("*"))) > 0 and not overwrite:
                raise FileExistsError("The specified folder is not empty! Use overwrite=True to overwrite it.")
            else:
                shutil.rmtree(str(save_path))

        # Solve with recursion
        if isinstance(segmentation_object, MultiSegmentationExtractor):
            segext_objs = segmentation_object.segmentations
            for plane_num, segext_obj in enumerate(segext_objs):
                save_path_plane = save_path / f"plane{plane_num}"
                Suite2pSegmentationExtractor.write_segmentation(segext_obj, save_path_plane)

        if not save_path.is_dir():
            save_path.mkdir(parents=True)
        if "plane" not in save_path.stem:
            save_path = save_path / "plane0"
            save_path.mkdir()

        # saving traces:
        if segmentation_object.get_traces(name="raw") is not None:
            np.save(save_path / "F.npy", segmentation_object.get_traces(name="raw").T)
        if segmentation_object.get_traces(name="neuropil") is not None:
            np.save(save_path / "Fneu.npy", segmentation_object.get_traces(name="neuropil").T)
        if segmentation_object.get_traces(name="deconvolved") is not None:
            np.save(
                save_path / "spks.npy",
                segmentation_object.get_traces(name="deconvolved").T,
            )
        # save stat
        stat = np.zeros(segmentation_object.get_num_rois(), "O")
        roi_locs = segmentation_object.roi_locations.T
        pixel_masks = segmentation_object.get_roi_pixel_masks(roi_ids=range(segmentation_object.get_num_rois()))
        for no, i in enumerate(stat):
            stat[no] = {
                "med": roi_locs[no, :].tolist(),
                "ypix": pixel_masks[no][:, 0],
                "xpix": pixel_masks[no][:, 1],
                "lam": pixel_masks[no][:, 2],
            }
        np.save(save_path / "stat.npy", stat)
        # saving iscell
        iscell = np.ones([segmentation_object.get_num_rois(), 2])
        iscell[segmentation_object.get_rejected_list(), 0] = 0
        np.save(save_path / "iscell.npy", iscell)
        # saving ops

        ops = dict(
            nframes=segmentation_object.get_num_frames(),
            Lx=segmentation_object.get_image_size()[1],
            Ly=segmentation_object.get_image_size()[0],
            xrange=[0, segmentation_object.get_image_size()[1]],
            yrange=[0, segmentation_object.get_image_size()[0]],
            fs=segmentation_object.get_sampling_frequency(),
            nchannels=segmentation_object.get_num_channels(),
            meanImg=segmentation_object.get_image("mean"),
            Vcorr=segmentation_object.get_image("correlation"),
        )
        if getattr(segmentation_object, "_raw_movie_file_location", None):
            ops.update(dict(filelist=[segmentation_object._raw_movie_file_location]))
        else:
            ops.update(dict(filelist=[None]))
        np.save(save_path / "ops.npy", ops)
