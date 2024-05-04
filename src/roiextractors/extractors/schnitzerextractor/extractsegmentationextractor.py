"""Extractor for reading the segmentation data that results from calls to EXTRACT.

Classes
-------
ExtractSegmentationExtractor
    Abstract class that defines which extractor class to use for a given file.
NewExtractSegmentationExtractor
    Extractor for reading the segmentation data that results from calls to newer versions of EXTRACT.
LegacyExtractSegmentationExtractor
    Extractor for reading the segmentation data that results from calls to older versions of EXTRACT.
"""

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
from ...segmentationextractor import SegmentationExtractor


def _decode_h5py_array(unicode_int_array: np.ndarray) -> str:
    """Auxiliary function to decode a numpy array of unicode ints to a string."""
    return "".join(chr(unicode_int) for unicode_int in unicode_int_array)


class ExtractSegmentationExtractor(ABC):
    """Abstract class that defines which extractor class to use for a given file."""

    extractor_name = "ExtractSegmentation"
    installed = HAVE_H5PY  # check at class level if installed or not
    installation_mesg = "To use ExtractSegmentationExtractor install h5py: \n\n pip install h5py \n\n"  # error message when not installed

    def __new__(
        cls,
        file_path: PathType,
        sampling_frequency: float,
        output_struct_name: Optional[str] = None,
    ):
        """Abstract class that defines which extractor class to use for a given file.

        For newer versions of the EXTRACT algorithm, the extractor class redirects to
        NewExtractSegmentationExtractor. For older versions, the extractor class
        redirects to LegacyExtractSegmentationExtractor.

        Parameters
        ----------
        file_path: str
            The location of the folder containing the .mat file.
        output_struct_name: str, optional
            The name of output struct in the .mat file.
            When unspecified, we check if any of the default values can be found in the file.
            For newer version of extract, the default name is assumed to be "output".
            For older versions the default is "extractAnalysisOutput". If none of them
            can be found, it must be supplied.
        sampling_frequency: float
            The sampling frequency in units of Hz.
        """
        self = super().__new__(cls)
        self.file_path = file_path
        # Check if the file is a .mat file
        cls._assert_file_is_mat(self)

        if output_struct_name is None:
            self.output_struct_name = cls._get_default_output_struct_name_from_file(self)
        else:
            # Check that user-given 'output_struct_name' is in the file
            self.output_struct_name = output_struct_name
            cls._assert_output_struct_name_is_in_file(self)

        # Check the version of the .mat file
        if cls._check_extract_file_version(self):
            # For newer versions of the .mat file, use the newer extractor
            return NewExtractSegmentationExtractor(
                file_path=file_path,
                sampling_frequency=sampling_frequency,
                output_struct_name=self.output_struct_name,
            )

        # For older versions of the .mat file, use the legacy extractor
        return LegacyExtractSegmentationExtractor(
            file_path=file_path,
            output_struct_name=self.output_struct_name,
        )

    def _assert_file_is_mat(self):
        """Check that the file exists and is a .mat file."""
        file_path = Path(self.file_path)
        assert file_path.exists(), f"File {file_path} does not exist."
        assert file_path.suffix == ".mat", f"File {file_path} must be a .mat file."

    def _get_default_output_struct_name_from_file(self):
        """Return the default value for 'output_struct_name' when it is unspecified.

        Returns
        -------
        output_struct_name: str
            The name of output struct in the .mat file.

        Notes
        -----
        For newer version of extract, the default name is assumed to be "output".
        For older versions the default is "extractAnalysisOutput".
        If none of them is found, raise an error that 'output_struct_name' must be supplied.
        """
        newer_default_output_struct_name = "output"
        legacy_default_output_struct_name = "extractAnalysisOutput"
        with h5py.File(name=self.file_path, mode="r") as mat_file:
            if newer_default_output_struct_name in mat_file.keys():
                return newer_default_output_struct_name
            elif legacy_default_output_struct_name in mat_file.keys():
                return legacy_default_output_struct_name
            else:
                raise AssertionError("The 'output_struct_name' must be supplied.")

    def _assert_output_struct_name_is_in_file(self):
        """Check that 'output_struct_name' is in the file, raise an error if not."""
        with h5py.File(name=self.file_path, mode="r") as mat_file:
            assert (
                self.output_struct_name in mat_file
            ), f"Output struct name '{self.output_struct_name}' not found in file."

    def _check_extract_file_version(self) -> bool:
        """Check the version of the extract file.

        Returns
        -------
        True if the file was created with a newer version of the EXTRACT algorithm,
        False otherwise.
        """
        with h5py.File(name=self.file_path, mode="r") as mat_file:
            dataset_version = mat_file[self.output_struct_name]["info"]["version"][:]
            dataset_version = np.ravel(dataset_version)
            # dataset_version is an HDF5 dataset of encoded characters
            version_name = _decode_h5py_array(dataset_version)

            return version.Version(version_name) >= version.Version("1.0.0")


class NewExtractSegmentationExtractor(
    SegmentationExtractor
):  # TODO: refactor to inherit from LegacyExtractSegmentationExtractor
    """Extractor for reading the segmentation data that results from calls to newer versions of EXTRACT.

    This class inherits from the SegmentationExtractor class, having all
    its functionality specifically applied to the dataset output from
    the 'EXTRACT' ROI segmentation method.
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
        self,
        file_path: PathType,
        sampling_frequency: float,
        output_struct_name: str = "output",
    ):
        """Load a SegmentationExtractor from a .mat file containing the output and config structs of the EXTRACT algorithm.

        Parameters
        ----------
        file_path: PathType
            Path to the .mat file containing the structs.
        sampling_frequency: float
            The sampling frequency in units of Hz. Supply if timing is regular.
        output_struct_name: str, optional
            The user has control over the names of the variables that return from `extraction(images, config)`.
            The tutorials for EXTRACT follow the naming convention of 'output', which we assume as the default.

        Notes
        -----
        For regular timing, supply the sampling frequency. For irregular timing, supply the timestamps.
        """
        super().__init__()

        self.output_struct_name = output_struct_name
        self.file_path = file_path

        if sampling_frequency is None:
            raise AssertionError("The sampling_frequency must be provided.")

        self._dataset_file = self._file_extractor_read()
        assert output_struct_name in self._dataset_file, "Output struct not found in file."
        self._output_struct = self._dataset_file[output_struct_name]

        assert "config" in self._output_struct, "Config struct not found in file."
        config_struct = self._output_struct["config"]
        self.config = self._config_struct_to_dict(config_struct=config_struct)

        traces = self._trace_extractor_read()
        if self.config["preprocess"][0] == 1:
            self._roi_response_dff = traces
        else:
            self._roi_response_raw = traces

        self._sampling_frequency = sampling_frequency

        self._image_masks = self._image_mask_extractor_read()

        assert "info" in self._output_struct, "Info struct not found in file."
        self._info_struct = self._output_struct["info"]
        extract_version = np.ravel(self._info_struct["version"][:])
        self.config.update(version=_decode_h5py_array(extract_version))

    def close(self):
        """Close the file when the object is deleted."""
        self._dataset_file.close()

    def _file_extractor_read(self):
        """Read the .mat file and return the file object."""
        return h5py.File(self.file_path, "r")

    def _config_struct_to_dict(self, config_struct: h5py.Group) -> dict:
        """Flatten the config struct into a dictionary."""
        config_dict = dict()
        for property_name in config_struct:
            if isinstance(config_struct[property_name], h5py.Dataset):
                data = np.ravel(config_struct[property_name][:])
                if property_name in ["trace_output_option", "cellfind_filter_type"]:
                    data = _decode_h5py_array(data)
                config_dict[property_name] = data
            elif isinstance(config_struct[property_name], h5py.Group):
                config_dict.update(self._config_struct_to_dict(config_struct=config_struct[property_name]))
        return config_dict

    def _image_mask_extractor_read(self) -> DatasetView:
        """Read the image masks from the .mat file and return the image masks.

        Returns
        -------
        image_masks : DatasetView
            3-D array: height x width x number of ROIs
        """
        return DatasetView(self._output_struct["spatial_weights"]).lazy_transpose()

    def _trace_extractor_read(self) -> DatasetView:
        """Read the traces from the .mat file and return the traces.

        Returns
        -------
        traces : DatasetView
            2-D array: number of frames x number of ROIs
        """
        return DatasetView(self._output_struct["temporal_weights"]).lazy_transpose()

    def get_accepted_list(self) -> list:
        return [roi for roi in self.get_roi_ids() if np.any(self._image_masks[..., roi])]

    def get_rejected_list(self) -> list:
        accepted_list = self.get_accepted_list()
        rejected_list = list(set(self.get_roi_ids()) - set(accepted_list))

        return rejected_list

    def get_roi_ids(self) -> list:
        return list(range(self.get_num_rois()))

    def get_image_size(self) -> ArrayType:
        return self._image_masks.shape[:-1]

    def get_images_dict(self):
        images_dict = super().get_images_dict()
        images_dict.update(
            summary_image=self._info_struct["summary_image"][:].T,
            f_per_pixel=self._info_struct["F_per_pixel"][:].T,
            max_image=self._info_struct["max_image"][:].T,
        )

        return images_dict


class LegacyExtractSegmentationExtractor(SegmentationExtractor):
    """Extractor for reading the segmentation data that results from calls to older versions of EXTRACT.

    This class inherits from the SegmentationExtractor class, having all
    its funtionality specifically applied to the dataset output from
    the 'EXTRACT' ROI segmentation method.
    """

    extractor_name = "LegacyExtractSegmentation"
    installed = HAVE_H5PY  # check at class level if installed or not
    is_writable = False
    mode = "file"
    installation_mesg = "To use extract install h5py: \n\n pip install h5py \n\n"  # error message when not installed

    def __init__(
        self,
        file_path: PathType,
        output_struct_name: str = "extractAnalysisOutput",
    ):
        """Create a LegacyExtractSegmentationExtractor from a .mat file.

        Parameters
        ----------
        file_path: str
            The location of the folder containing dataset.mat file.
        output_struct_name: str, optional
            The user has control over the names of the variables that return from `extraction(images, config)`.
            When unspecified, the default is 'extractAnalysisOutput'.
        """
        super().__init__()
        self.file_path = file_path
        self._dataset_file = self._file_extractor_read()
        self.output_struct_name = output_struct_name
        self._image_masks = self._image_mask_extractor_read()
        self._roi_response_raw = self._trace_extractor_read()
        self._raw_movie_file_location = self._raw_datafile_read()
        self._sampling_frequency = self._roi_response_raw.shape[0] / self._tot_exptime_extractor_read()
        self._image_correlation = self._summary_image_read()

    def __del__(self):
        """Close the file when the object is deleted."""
        self._dataset_file.close()

    def _file_extractor_read(self):
        """Read the .mat file and return the file object."""
        return h5py.File(self.file_path, "r")

    def _image_mask_extractor_read(self):
        """Read the image masks from the .mat file and return the image masks.

        Returns
        -------
        image_masks : DatasetView
            3-D array: height x width x number of ROIs
        """
        return self._dataset_file[self.output_struct_name]["filters"][:].transpose([1, 2, 0])

    def _trace_extractor_read(self):
        """Read the traces from the .mat file and return the traces.

        Returns
        -------
        traces : DatasetView
            2-D array: number of frames x number of ROIs
        """
        return self._dataset_file[self.output_struct_name]["traces"]

    def _tot_exptime_extractor_read(self):
        """Read the total experiment time from the .mat file and return the total experiment time.

        Returns
        -------
        tot_exptime : float
            The total experiment time in units of seconds.
        """
        return self._dataset_file[self.output_struct_name]["time"]["totalTime"][0][0]

    def _summary_image_read(self):
        """Read the summary image from the .mat file and return the summary image.

        Returns
        -------
        summary_image : numpy.ndarray
            The summary image.
        """
        summary_image = self._dataset_file[self.output_struct_name]["info"]["summary_image"]
        return np.array(summary_image)

    def _raw_datafile_read(self):
        """Read the raw data file location from the .mat file and return the raw data file location.

        Returns
        -------
        raw_datafile : str
            The raw data file location.
        """
        if self._dataset_file[self.output_struct_name].get("file"):
            charlist = [chr(i) for i in np.squeeze(self._dataset_file[self.output_struct_name]["file"][:])]
            return "".join(charlist)

    def get_accepted_list(self):
        return list(range(self.get_num_rois()))

    def get_rejected_list(self):
        ac_set = set(self.get_accepted_list())
        return [a for a in range(self.get_num_rois()) if a not in ac_set]

    def get_image_size(self):
        return self._image_masks.shape[0:2]
