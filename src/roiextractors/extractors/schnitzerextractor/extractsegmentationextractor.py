"""Extractor for reading the segmentation data that results from calls to EXTRACT."""
from abc import ABC
from pathlib import Path
from typing import Optional

import numpy as np
from lazy_ops import DatasetView
from packaging import version

try:
    import h5py

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


from ...extraction_tools import PathType, ArrayType
from ...multisegmentationextractor import MultiSegmentationExtractor
from ...segmentationextractor import SegmentationExtractor


class ExtractSegmentationExtractor(ABC):
    """Abstract class that defines which extractor class to use for a given file."""

    extractor_name = "ExtractSegmentation"
    installed = HAVE_H5PY  # check at class level if installed or not
    installation_mesg = "To use ExtractSegmentationExtractor install h5py: \n\n pip install h5py \n\n"  # error message when not installed

    def __new__(cls, file_path: PathType):
        # Check if the file is a .mat file
        cls._assert_file_is_mat(file_path=file_path)
        # Check the version of the .mat file
        if cls._check_extract_file_version(file_path=file_path):
            # For newer versions of the .mat file, use the newer extractor
            return NewExtractSegmentationExtractor(file_path=file_path)

        # For older versions of the .mat file, use the legacy extractor
        return LegacyExtractSegmentationExtractor(file_path=file_path)

    @staticmethod
    def _assert_file_is_mat(file_path: PathType):
        """Check that the file is a .mat file."""
        file_path = Path(file_path)
        assert file_path.suffix == ".mat", "File must be a .mat file"

    @staticmethod
    def _check_extract_file_version(file_path: PathType) -> bool:
        """Check the version of the extract file.
        If the file was created with a newer version of the EXTRACT algorithm, the
        function will return True, otherwise it will return False."""
        with h5py.File(name=file_path, mode="r") as mat_file:
            dataset_version = mat_file["output"]["info"]["version"]
            # dataset_version is an HDF5 dataset of encoded characters
            version_name = "".join(
                chr(unicode_int_array[0]) for unicode_int_array in dataset_version)

            return version.Version(version_name) >= version.Version("1.1.0")


class NewExtractSegmentationExtractor(SegmentationExtractor):
    """
    This class inherits from the SegmentationExtractor class, having all
    its funtionality specifically applied to the dataset output from
    the \'EXTRACT\' ROI segmentation method.
    """

    extractor_name = "NewExtractSegmentation"
    installed = HAVE_H5PY  # check at class level if installed or not
    installation_mesg = (
        "To use NewExtractSegmentation install h5py: \n\n pip install h5py \n\n"
    # error message when not installed
    )
    is_writable = False
    mode = "file"

    def __init__(
            self, file_path: PathType, output_struct_name: Optional[str] = None,
    ):
        """
        Load a SegmentationExtractor from a .mat file containing the output and config structs of the EXTRACT algorithm.

        Parameters
        ----------
        file_path: PathType
            Path to the .mat file containing the structs.
        output_struct_name: str, optional
            The user has control over the names of the variables that return from `extraction(images, config)`.
            The tutorials for EXTRACT follow the naming convention of 'output', which we assume as the default.
        """
        SegmentationExtractor.__init__(self)

        if output_struct_name is None:
            output_struct_name = "output"

        self.file_path = file_path

        self._dataset_file = self._file_extractor_read()
        assert output_struct_name in self._dataset_file, "Output struct not found in file."
        assert "config" in self._dataset_file[output_struct_name], "Config struct not found in file."
        self._output_struct = self._dataset_file[output_struct_name]
        self._info_struct = self._output_struct["info"] if "info" in self._output_struct else None
        config_struct = self._output_struct["config"]
        self._config = self._config_struct_to_dict(config_struct=config_struct)

        traces = self._trace_extractor_read()
        if self._config["preprocess"][0] == 1:
            self._roi_response_dff = traces
        else:
            self._roi_response_raw = traces

        self._image_correlation = self._summary_image_read() if self._info_struct else None

        runtime = self._runtime_extractor_read()
        self._sampling_frequency = traces.shape[1] / runtime

        # (50, 50, 20) [movie_height x movie_width x number_of_cells_found]
        self._image_masks = self._image_mask_extractor_read()

    def __del__(self):
        self._dataset_file.close()

    def _file_extractor_read(self):
        return h5py.File(self.file_path, "r")

    def _config_struct_to_dict(self, config_struct: h5py.Group) -> dict:
        """Flatten the config struct into a dictionary."""
        config_dict = dict()
        for key in config_struct:
            if isinstance(config_struct[key], h5py.Dataset):
                config_dict[key] = np.ravel(config_struct[key][:])
            elif isinstance(config_struct[key], h5py.Group):
                config_dict[key] = self._config_struct_to_dict(config_struct[key])
        return config_dict

    def _image_mask_extractor_read(self) -> np.ndarray:
        """Returns the image masks as a numpy array of shape of height, width, number of ROIs."""
        return DatasetView(self._output_struct["spatial_weights"]).lazy_transpose().dsetread()

    def _trace_extractor_read(self) -> np.ndarray:
        """Returns the traces as a numpy array where the first dimension is the
        number of ROIs and the second dimension is the number of frames."""
        return DatasetView(self._output_struct["temporal_weights"]).dsetread()

    def _summary_image_read(self) -> np.ndarray:
        """Returns the summary image as a numpy array where the first dimension is the
        height of the frame and the second dimension is the width of the frame."""
        return DatasetView(self._info_struct["summary_image"]).dsetread()

    def _runtime_extractor_read(self):
        """Returns the runtime of EXTRACT algorithm in seconds.
        Can be used to calculate the sampling frequency of the traces."""
        return np.ravel(self._info_struct["runtime"][:])[0]

    def get_accepted_list(self) -> list:
        """
        The ids of the ROIs which are accepted after manual verification of
        ROIs.

        Returns
        -------
        accepted_list: list
            List of accepted ROIs
        """
        # for sparse array
        accepted_list = [roi for roi in self.get_roi_ids() if np.any(self._image_masks[roi])]
        # find the indices of the accepted ROIs where the spatial_weights is not all zeros
        # accepted_list = []
        # for roi in self.get_roi_ids():
        #     if any(x != 0 for x in np.ravel(self._image_masks[:, :, roi])):
        #         accepted_list.append(roi)

        return accepted_list

    def get_rejected_list(self) -> list:
        """
        The ids of the ROIs which are rejected after manual verification of
        ROIs.

        Returns
        -------
        rejected_list: list
            List of rejected ROIs
        """
        accepted_list = self.get_accepted_list()
        rejected_list = list(set(self.get_roi_ids()) - set(accepted_list))

        return rejected_list

    def get_roi_ids(self) -> list:
        """Returns the list of ROI ids.

        Returns
        -------
        roi_ids: list
            ROI ids list.
        """
        return list(range(self.get_num_rois()))

    def get_image_size(self) -> ArrayType:
        """
        Frame size of movie (height and width of image).

        Returns
        -------
        image_size: array_like
            2-D array: image height x image width
        """
        return DatasetView(self._info_struct["max_image"]).dsetread()


class LegacyExtractSegmentationExtractor(SegmentationExtractor):
    """
    This class inherits from the SegmentationExtractor class, having all
    its funtionality specifically applied to the dataset output from
    the \'EXTRACT\' ROI segmentation method.
    """

    extractor_name = "LegacyExtractSegmentation"
    installed = HAVE_H5PY  # check at class level if installed or not
    is_writable = False
    mode = "file"
    installation_mesg = "To use extract install h5py: \n\n pip install h5py \n\n"  # error message when not installed

    def __init__(self, file_path: PathType):
        """
        Parameters
        ----------
        file_path: str
            The location of the folder containing dataset.mat file.
        """
        SegmentationExtractor.__init__(self)
        self.file_path = file_path
        self._dataset_file, self._group0 = self._file_extractor_read()
        self._image_masks = self._image_mask_extractor_read()
        self._roi_response_raw = self._trace_extractor_read()
        self._raw_movie_file_location = self._raw_datafile_read()
        self._sampling_frequency = self._roi_response_raw.shape[1] / self._tot_exptime_extractor_read()
        self._image_correlation = self._summary_image_read()

    def __del__(self):
        self._dataset_file.close()

    def _file_extractor_read(self):
        f = h5py.File(self.file_path, "r")
        _group0_temp = list(f.keys())
        _group0 = [a for a in _group0_temp if "#" not in a]
        return f, _group0

    def _image_mask_extractor_read(self):
        return DatasetView(self._dataset_file[self._group0[0]]["filters"]).lazy_transpose([1, 2, 0]).dsetread()

    def _trace_extractor_read(self):
        extracted_signals = DatasetView(self._dataset_file[self._group0[0]]["traces"])
        return extracted_signals.T

    def _tot_exptime_extractor_read(self):
        return self._dataset_file[self._group0[0]]["time"]["totalTime"][0][0]

    def _summary_image_read(self):
        summary_image = self._dataset_file[self._group0[0]]["info"]["summary_image"]
        return np.array(summary_image)

    def _raw_datafile_read(self):
        if self._dataset_file[self._group0[0]].get("file"):
            charlist = [chr(i) for i in np.squeeze(self._dataset_file[self._group0[0]]["file"][:])]
            return "".join(charlist)

    def get_accepted_list(self):
        return list(range(self.get_num_rois()))

    def get_rejected_list(self):
        ac_set = set(self.get_accepted_list())
        return [a for a in range(self.get_num_rois()) if a not in ac_set]

    # defining the abstract class informed methods:
    def get_roi_ids(self):
        return list(range(self.get_num_rois()))

    def get_image_size(self):
        return self._image_masks.shape[0:2]

    @staticmethod
    def write_segmentation(segmentation_object: SegmentationExtractor, save_path, overwrite=True):
        save_path = Path(save_path)
        assert save_path.suffix == ".mat", "'save_path' must be a *.mat file"
        if save_path.is_file():
            if not overwrite:
                raise FileExistsError("The specified path exists! Use overwrite=True to overwrite it.")
            else:
                save_path.unlink()

        folder_path = save_path.parent
        file_name = save_path.name
        if isinstance(segmentation_object, MultiSegmentationExtractor):
            segext_objs = segmentation_object.segmentations
            for plane_num, segext_obj in enumerate(segext_objs):
                save_path_plane = folder_path / f"Plane_{plane_num}" / file_name
                ExtractSegmentationExtractor.write_segmentation(segext_obj, save_path_plane)
        if not folder_path.is_dir():
            folder_path.mkdir(parents=True)

        with h5py.File(save_path, "a") as f:
            # create base groups:
            _ = f.create_group("#refs#")
            main = f.create_group("extractAnalysisOutput")
            # create datasets:
            main.create_dataset("filters", data=segmentation_object.get_roi_image_masks().transpose((2, 0, 1)))
            main.create_dataset("traces", data=segmentation_object.get_traces().T)
            if getattr(segmentation_object, "_raw_movie_file_location", None):
                main.create_dataset(
                    "file",
                    data=[ord(alph) for alph in str(segmentation_object._raw_movie_file_location)],
                )
            info = main.create_group("info")
            if segmentation_object.get_image() is not None:
                info.create_dataset("summary_image", data=segmentation_object.get_image())
            time = main.create_group("time")
            if segmentation_object.get_sampling_frequency() is not None:
                time.create_dataset(
                    "totalTime",
                    (1, 1),
                    data=segmentation_object.get_num_frames() / segmentation_object.get_sampling_frequency(),
                )
