"""A segmentation extractor for Suite2p.

Classes
-------
Suite2pSegmentationExtractor
    A segmentation extractor for Suite2p.
"""

import shutil
from pathlib import Path
from typing import Optional
from warnings import warn
import os
import numpy as np

from ...extraction_tools import PathType
from ...extraction_tools import _image_mask_extractor
from ...multisegmentationextractor import MultiSegmentationExtractor
from ...segmentationextractor import SegmentationExtractor


class Suite2pSegmentationExtractor(SegmentationExtractor):
    """A segmentation extractor for Suite2p."""

    extractor_name = "Suite2pSegmentationExtractor"
    installed = True  # check at class level if installed or not
    is_writable = False
    mode = "folder"
    installation_mesg = ""  # error message when not installed

    @classmethod
    def get_available_channels(cls, folder_path: PathType):
        """Get the available channel names from the folder paths produced by Suite2p.

        Parameters
        ----------
        file_path : PathType
            Path to Suite2p output path.

        Returns
        -------
        channel_names: list
            List of channel names.
        """
        plane_names = cls.get_available_planes(folder_path=folder_path)

        channel_names = ["chan1"]
        second_channel_paths = list((Path(folder_path) / plane_names[0]).glob("F_chan2.npy"))
        if not second_channel_paths:
            return channel_names
        channel_names.append("chan2")

        return channel_names

    @classmethod
    def get_available_planes(cls, folder_path: PathType):
        """Get the available plane names from the folder produced by Suite2p.

        Parameters
        ----------
        file_path : PathType
            Path to Suite2p output path.

        Returns
        -------
        plane_names: list
            List of plane names.
        """
        from natsort import natsorted

        folder_path = Path(folder_path)
        prefix = "plane"
        plane_paths = natsorted(folder_path.glob(pattern=prefix + "*"))
        assert len(plane_paths), f"No planes found in '{folder_path}'."
        plane_names = [plane_path.stem for plane_path in plane_paths]
        return plane_names

    def __init__(
        self,
        folder_path: PathType,
        channel_name: Optional[str] = None,
        plane_name: Optional[str] = None,
        combined: Optional[bool] = None,  # TODO: to be removed
        plane_no: Optional[int] = None,  # TODO: to be removed
    ):
        """Create SegmentationExtractor object out of suite 2p data type.

        Parameters
        ----------
        folder_path: str or Path
            The path to the 'suite2p' folder.
        channel_name: str, optional
            The name of the channel to load, to determine what channels are available use Suite2pSegmentationExtractor.get_available_channels(folder_path).
        plane_name: str, optional
            The name of the plane to load, to determine what planes are available use Suite2pSegmentationExtractor.get_available_planes(folder_path).

        """
        if combined:
            warning_string = "Keyword argument 'combined' is deprecated and will be removed on or after Nov, 2023. "
            warn(
                message=warning_string,
                category=DeprecationWarning,
            )
        if plane_no:
            warning_string = (
                "Keyword argument 'plane_no' is deprecated and will be removed on or after Nov, 2023 in favor of 'plane_name'."
                "Specify which stream you wish to load with the 'plane_name' keyword argument."
            )
            warn(
                message=warning_string,
                category=DeprecationWarning,
            )

        channel_names = self.get_available_channels(folder_path=folder_path)
        if channel_name is None:
            if len(channel_names) > 1:
                # For backward compatibility maybe it is better to warn first
                warn(
                    "More than one channel is detected! Please specify which channel you wish to load with the `channel_name` argument. "
                    "To see what channels are available, call `Suite2pSegmentationExtractor.get_available_channels(folder_path=...)`.",
                    UserWarning,
                )
            channel_name = channel_names[0]

        self.channel_name = channel_name
        if self.channel_name not in channel_names:
            raise ValueError(
                f"The selected channel '{channel_name}' is not a valid channel name. To see what channels are available, "
                f"call `Suite2pSegmentationExtractor.get_available_channels(folder_path=...)`."
            )

        plane_names = self.get_available_planes(folder_path=folder_path)
        if plane_name is None:
            if len(plane_names) > 1:
                # For backward compatibility maybe it is better to warn first
                warn(
                    "More than one plane is detected! Please specify which plane you wish to load with the `plane_name` argument. "
                    "To see what planes are available, call `Suite2pSegmentationExtractor.get_available_planes(folder_path=...)`.",
                    UserWarning,
                )
            plane_name = plane_names[0]

        if plane_name not in plane_names:
            raise ValueError(
                f"The selected plane '{plane_name}' is not a valid plane name. To see what planes are available, "
                f"call `Suite2pSegmentationExtractor.get_available_planes(folder_path=...)`."
            )
        self.plane_name = plane_name

        super().__init__()

        self.folder_path = Path(folder_path)

        options = self._load_npy(file_name="ops.npy")
        self.options = options.item() if options is not None else options
        self._sampling_frequency = self.options["fs"]
        self._num_frames = self.options["nframes"]
        self._image_size = (self.options["Ly"], self.options["Lx"])

        self.stat = self._load_npy(file_name="stat.npy")

        fluorescence_traces_file_name = "F.npy" if channel_name == "chan1" else "F_chan2.npy"
        neuropil_traces_file_name = "Fneu.npy" if channel_name == "chan1" else "Fneu_chan2.npy"
        self._roi_response_raw = self._load_npy(file_name=fluorescence_traces_file_name, mmap_mode="r", transpose=True)
        self._roi_response_neuropil = self._load_npy(file_name=neuropil_traces_file_name, mmap_mode="r", transpose=True)
        self._roi_response_deconvolved = (
            self._load_npy(file_name="spks.npy", mmap_mode="r", transpose=True) if channel_name == "chan1" else None
        )

        # rois segmented from the iamging acquired with second channel (red/anatomical) that match the first channel segmentation
        redcell = self._load_npy(file_name="redcell.npy", mmap_mode="r")
        if channel_name == "chan2" and redcell is not None:
            self.iscell = redcell
        else:
            self.iscell = self._load_npy("iscell.npy", mmap_mode="r")

        # The name of the OpticalChannel object is "OpticalChannel" if there is only one channel, otherwise it is
        # "Chan1" or "Chan2".
        self._channel_names = ["OpticalChannel" if len(channel_names) == 1 else channel_name.capitalize()]

        self._image_correlation = self._correlation_image_read()
        image_mean_name = "meanImg" if channel_name == "chan1" else f"meanImg_chan2"
        self._image_mean = self.options[image_mean_name] if image_mean_name in self.options else None
        roi_indices = list(range(self.get_num_rois()))
        self._image_masks = _image_mask_extractor(
            self.get_roi_pixel_masks(),
            roi_indices,
            self.get_image_size(),
        )

    def _load_npy(self, file_name: str, mmap_mode=None, transpose: bool = False):
        """Load a .npy file with specified filename. Returns None if file is missing.

        Parameters
        ----------
        file_name: str
            The name of the .npy file to load.
        mmap_mode: str
            The mode to use for memory mapping. See numpy.load for details.
        transpose: bool, optional
            Whether to transpose the loaded array.

        Returns
        -------
            The loaded .npy file.
        """
        file_path = self.folder_path / self.plane_name / file_name
        if not file_path.exists():
            return

        data = np.load(file_path, mmap_mode=mmap_mode, allow_pickle=mmap_mode is None)
        if transpose:
            return data.T

        return data

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_accepted_list(self):
        return list(np.where(self.iscell[:, 0] == 1)[0])

    def get_rejected_list(self):
        return list(np.where(self.iscell[:, 0] == 0)[0])

    def _correlation_image_read(self):
        """Read correlation image from ops (settings) dict.

        Returns
        -------
        img : numpy.ndarray | None
            The correlation image.
        """
        if "Vcorr" not in self.options:
            return None

        correlation_image = self.options["Vcorr"]
        if (self.options["yrange"][-1], self.options["xrange"][-1]) == self._image_size:
            return correlation_image

        img = np.zeros(self._image_size, correlation_image.dtype)
        img[
            (self.options["Ly"] - self.options["yrange"][-1]) : (self.options["Ly"] - self.options["yrange"][0]),
            self.options["xrange"][0] : self.options["xrange"][-1],
        ] = correlation_image

        return img

    @property
    def roi_locations(self):
        """Returns the center locations (x, y) of each ROI."""
        return np.array([j["med"] for j in self.stat]).T.astype(int)

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
        return self._image_size

    @staticmethod
    def write_segmentation(segmentation_object: SegmentationExtractor, save_path: PathType, overwrite=True):
        """Write a SegmentationExtractor to a folder specified by save_path.

        Parameters
        ----------
        segmentation_object: SegmentationExtractor
            The SegmentationExtractor object to be written.
        save_path: str or Path
            The folder path where to write the segmentation.
        overwrite: bool
            If True, overwrite the folder if it already exists.

        Raises
        ------
        AssertionError
            If save_path is not a folder.
        FileExistsError
            If the folder already exists and overwrite is False.

        Notes
        -----
        The folder structure is as follows:
        save_path
        └── plane<plane_num>
            ├── F.npy
            ├── Fneu.npy
            ├── spks.npy
            ├── stat.npy
            ├── iscell.npy
            └── ops.npy
        """
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
