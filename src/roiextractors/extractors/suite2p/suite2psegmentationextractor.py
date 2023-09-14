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
    def get_streams(cls, folder_path: PathType):
        folder_path = Path(folder_path)
        stream_paths = [f for f in folder_path.iterdir() if f.is_dir()]
        chan_1_streams = [f"chan1_{stream_path.stem}" for stream_path in stream_paths]
        streams = dict(channel_streams=["chan1"], plane_streams=dict(chan1=chan_1_streams))

        chan_2_streams = []
        for stream_path in stream_paths:
            if list(stream_path.glob("F_chan2.npy")):
                chan_2_streams.append(f"chan2_{stream_path.stem}")

        if chan_2_streams:
            streams["channel_streams"].append("chan2")
            streams["plane_streams"].update(chan2=chan_2_streams)

        return streams

    def __init__(
        self,
        folder_path: PathType,
        stream_name: Optional[str] = None,
        combined: Optional[bool] = None,  # TODO: to be removed
        plane_no: Optional[int] = None,  # TODO: to be removed
    ):
        """Create SegmentationExtractor object out of suite 2p data type.

        Parameters
        ----------
        folder_path: str or Path
            The path to the 'suite2p' folder.
        stream_name: str, optional
            The name of the stream to load, to determine which streams are available use Suite2pSegmentationExtractor.get_streams(folder_path).

        """

        if combined:
            warning_string = "Keyword argument 'combined' is deprecated and will be removed on or after Nov, 2023. "
            warn(
                message=warning_string,
                category=DeprecationWarning,
            )
        if plane_no:
            warning_string = (
                "Keyword argument 'plane_no' is deprecated and will be removed on or after Nov, 2023 in favor of 'stream_name'."
                "Specify which stream you wish to load with the 'stream_name' keyword argument."
            )
            warn(
                message=warning_string,
                category=DeprecationWarning,
            )

        streams = self.get_streams(folder_path=folder_path)
        if stream_name is None:
            if len(streams["channel_streams"]) > 1:
                raise ValueError(
                    "More than one channel is detected! Please specify which stream you wish to load with the `stream_name` argument. "
                    "To see what streams are available, call `Suite2pSegmentationExtractor.get_streams(folder_path=...)`."
                )
            channel_stream_name = streams["channel_streams"][0]
            stream_name = streams["plane_streams"][channel_stream_name][0]

        channel_stream_name = stream_name.split("_")[0]
        if channel_stream_name not in streams["channel_streams"]:
            raise ValueError(
                f"The selected stream '{channel_stream_name}' is not a valid stream name. To see what streams are available, "
                f"call `Suite2pSegmentationExtractor.get_streams(folder_path=...)`."
            )

        plane_stream_names = streams["plane_streams"][channel_stream_name]
        if stream_name is not None and stream_name not in plane_stream_names:
            raise ValueError(
                f"The selected stream '{stream_name}' is not in the available plane_streams '{plane_stream_names}'!"
            )
        self.stream_name = stream_name

        super().__init__()

        self.folder_path = Path(folder_path)

        self._options = self._load_npy(file_name="ops.npy").item()

        fluorescence_traces_file_name = "F.npy" if channel_stream_name == "chan1" else "F_chan2.npy"
        neuropil_traces_file_name = "Fneu.npy" if channel_stream_name == "chan1" else "Fneu_chan2.npy"

        self._sampling_frequency = self._options["fs"]
        self._num_frames = self._options["nframes"]
        self._image_size = (self._options["Ly"], self._options["Lx"])

        self.stat = self._load_npy(file_name="stat.npy")
        self._roi_response_raw = self._load_npy(file_name=fluorescence_traces_file_name, mmap_mode="r").T
        self._roi_response_neuropil = self._load_npy(file_name=neuropil_traces_file_name, mmap_mode="r").T
        self._roi_response_deconvolved = self._load_npy(file_name="spks.npy", mmap_mode="r").T
        self.iscell = self._load_npy("iscell.npy", mmap_mode="r")

        channel_name = (
            "OpticalChannel"
            if len(streams["channel_streams"]) == 1
            else channel_stream_name.capitalize()
        )
        self._channel_names = [channel_name]

        self._image_correlation = self._correlation_image_read()
        image_mean_name = "meanImg" if channel_stream_name == "chan1" else f"meanImg_chan2"
        self._image_mean = self._options[image_mean_name]

    def _load_npy(self, file_name: str, mmap_mode=None):
        """Load a .npy file with specified filename.

        Parameters
        ----------
        filename: str
            The name of the .npy file to load.
        mmap_mode: str
            The mode to use for memory mapping. See numpy.load for details.

        Returns
        -------
            The loaded .npy file.
        """
        plane_stream_name = self.stream_name.split("_")[-1]
        file_path = self.folder_path / plane_stream_name / file_name
        return np.load(file_path, mmap_mode=mmap_mode, allow_pickle=mmap_mode is None)

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
        correlation_image = self._options["Vcorr"]
        if (self._options["yrange"][-1], self._options["xrange"][-1]) == self._image_size:
            return correlation_image

        img = np.zeros(self._image_size, correlation_image.dtype)
        img[
            (self._options["Ly"] - self._options["yrange"][-1]) : (self._options["Ly"] - self._options["yrange"][0]),
            self._options["xrange"][0] : self._options["xrange"][-1],
        ] = correlation_image

        return img

    @property
    def roi_locations(self):
        """Returns the center locations (x, y) of each ROI."""
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
                        self._options["Ly"] - 1 - self.stat[i]["ypix"],
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
                "ypix": segmentation_object.get_image_size()[0] - 1 - pixel_masks[no][:, 0],
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
