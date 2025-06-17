"""A SegmentationExtractor for CaImAn.

Classes
-------
CaimanSegmentationExtractor
    A class for extracting segmentation from multi or single plane CaImAn output.
"""

from pathlib import Path
from warnings import warn
from typing import Optional, List

import h5py
from scipy.sparse import csc_matrix
import numpy as np

from ...extraction_tools import PathType, get_package
from ...multisegmentationextractor import MultiSegmentationExtractor
from ...segmentationextractor import SegmentationExtractor


class CaimanSegmentationMultiExtractor(SegmentationExtractor):
    """A SegmentationExtractor for CaImAn with multiplane handling capabilities."""

    extractor_name = "CaimanSegmentation"

    @classmethod
    def get_available_planes(cls, folder_path: PathType) -> List[str]:
        """Get the available plane names from CaImAn output folder."""
        folder_path = Path(folder_path)

        if folder_path.is_file():
            if folder_path.suffix in [".hdf5", ".h5"]:
                return [folder_path.stem]
            else:
                raise ValueError(f"File {folder_path} is not a valid HDF5 file")

        if not folder_path.is_dir():
            raise ValueError(f"Path {folder_path} does not exist")

        plane_files = cls._find_plane_files_static(folder_path)

        if not plane_files:
            raise ValueError(f"No CaImAn plane files found in {folder_path}")

        plane_names = []
        for plane_file in plane_files:
            if plane_file.parent != folder_path:
                plane_names.append(plane_file.parent.name)
            else:
                plane_names.append(plane_file.stem)

        return plane_names

    @staticmethod
    def _find_plane_files_static(folder_path: Path) -> List[Path]:
        """Find all HDF5 files representing different planes in a CaImAn multiplane output folder."""
        # Try subfolder pattern first
        subfolder_files = CaimanSegmentationMultiExtractor._find_plane_subfolders_static(folder_path)
        if subfolder_files:
            return subfolder_files

        # Try flat files pattern
        flat_files = CaimanSegmentationMultiExtractor._find_flat_plane_files_static(folder_path)

        if not flat_files:
            warn(
                f"No CaImAn plane files found in {folder_path}. "
                "Supported patterns: subfolders named like 'plane0', 'Plane_0', or 'plane_0' containing HDF5 files; "
                "multiple HDF5 files in the main folder; or a single HDF5 file in the main folder.",
                UserWarning,
                stacklevel=2,
            )
        return flat_files

    @staticmethod
    def _find_plane_subfolders_static(folder_path: Path) -> List[Path]:
        """Find HDF5 files in plane subfolders."""
        plane_files = []
        plane_patterns = ["plane{}", "Plane_{}", "plane_{}"]

        plane_index = 0
        while True:
            found_plane = False

            for pattern in plane_patterns:
                plane_folder_name = pattern.format(plane_index)
                plane_folder = folder_path / plane_folder_name

                if plane_folder.is_dir():
                    hdf5_files = list(plane_folder.glob("*.hdf5")) + list(plane_folder.glob("*.h5"))
                    valid_files = [f for f in hdf5_files if CaimanSegmentationMultiExtractor._is_caiman_file_static(f)]

                    if valid_files:
                        plane_files.append(valid_files[0])
                        found_plane = True
                        break

            if not found_plane:
                break
            plane_index += 1

        return sorted(plane_files)

    @staticmethod
    def _find_flat_plane_files_static(folder_path: Path) -> List[Path]:
        """Find multiple HDF5 files in the same folder representing different planes."""
        hdf5_files = list(folder_path.glob("*.hdf5")) + list(folder_path.glob("*.h5"))

        if len(hdf5_files) <= 1:
            return hdf5_files

        valid_files = [f for f in hdf5_files if CaimanSegmentationMultiExtractor._is_caiman_file_static(f)]

        if len(valid_files) > 1:
            return sorted(valid_files)

        return sorted(hdf5_files)

    @staticmethod
    def _is_caiman_file_static(file_path: Path) -> bool:
        """Check if an HDF5 file contains valid CaImAn data structure."""
        try:
            with h5py.File(file_path, "r") as f:
                required_groups = ["estimates", "params"]
                return all(group in f for group in required_groups)
        except Exception:
            warn(f"No valid CaImAn file found at {file_path}.", UserWarning, stacklevel=2)
            return False

    @classmethod
    def create_multiplane_extractor(cls, folder_path: PathType) -> MultiSegmentationExtractor:
        """Create a MultiSegmentationExtractor from folder containing multiple planes."""
        plane_names = cls.get_available_planes(folder_path)

        extractors = []
        for plane_name in plane_names:
            extractor = cls(folder_path, plane_name=plane_name)
            extractors.append(extractor)

        return MultiSegmentationExtractor(extractors, plane_names=plane_names)

    def __init__(self, file_path: PathType, plane_name: Optional[str] = None):
        """Initialize a CaimanSegmentationMultiExtractor instance."""
        SegmentationExtractor.__init__(self)

        self.file_path = Path(file_path)
        self._is_single_file = False

        # Handle single file case
        if self.file_path.is_file() and self.file_path.suffix in [".hdf5", ".h5"]:
            self._is_single_file = True
            if plane_name is not None:
                warn(f"plane_name '{plane_name}' ignored when loading single file")
            self.plane_name = None
            self._load_single_file(self.file_path)
            return

        # Handle folder case
        if not self.file_path.is_dir():
            raise ValueError(f"Invalid path: {file_path}. Must be HDF5 file or directory.")

        try:
            available_planes = self.get_available_planes(self.file_path)
        except ValueError as e:
            raise ValueError(f"No valid CaImAn data found in {file_path}: {e}")

        # Handle plane selection
        if plane_name is None:
            if len(available_planes) > 1:
                warn(
                    f"Multiple planes detected: {available_planes}. "
                    f"Loading first plane '{available_planes[0]}'. "
                    f"To load a specific plane, use plane_name parameter. "
                    f"To load all planes, use CaimanSegmentationExtractor.create_multiplane_extractor()."
                )
            plane_name = available_planes[0]

        if plane_name not in available_planes:
            raise ValueError(
                f"The selected plane '{plane_name}' is not valid. "
                f"Available planes: {available_planes}. "
                f"Use CaimanSegmentationExtractor.get_available_planes() to see all options."
            )

        self.plane_name = plane_name

        plane_file = self._get_plane_file(plane_name)
        self._load_single_file(plane_file)

    def _get_plane_file(self, plane_name: str) -> Path:
        """Get the file path for a specific plane name."""
        plane_files = self._find_plane_files()

        for plane_file in plane_files:
            if plane_file.parent != self.file_path:
                if plane_file.parent.name == plane_name:
                    return plane_file
            else:
                if plane_file.stem == plane_name:
                    return plane_file

        raise ValueError(f"Could not find file for plane '{plane_name}'")

    def _find_plane_files(self) -> List[Path]:
        """Find all HDF5 files representing different planes."""
        return self._find_plane_files_static(self.file_path)

    def _load_single_file(self, hdf5_file: Path):
        """Load data from a single HDF5 file."""
        self._dataset_file = h5py.File(hdf5_file, "r")
        self._raw_movie_file_location = str(hdf5_file)
        self._load_data()

    def _load_data(self):
        """Load essential data from the HDF5 file."""
        # Load required traces - handle cases where raw traces can't be computed
        self._roi_response_raw = self._raw_trace_extractor_read()

        # If raw traces failed, try to get from denoised traces alone
        if self._roi_response_raw is None:
            warn("Could not compute raw traces, trying denoised traces only")
            denoised_traces = self._trace_extractor_read("C")
            if denoised_traces is not None:
                # Convert lazy view to array if needed
                if hasattr(denoised_traces, "lazy_transpose"):
                    self._roi_response_raw = np.array(denoised_traces)
                else:
                    self._roi_response_raw = denoised_traces

        self._roi_response_dff = self._trace_extractor_read("F_dff")
        self._roi_response_denoised = self._trace_extractor_read("C")
        self._roi_response_neuropil = self._trace_extractor_read("f")
        self._roi_response_deconvolved = self._trace_extractor_read("S")

        # Load required images
        self._image_correlation = self._correlation_image_read()
        self._image_mean = self._summary_image_read()

        # Load sampling frequency
        try:
            fr_dataset = self._dataset_file["params"]["data"]["fr"]
            if fr_dataset.shape == ():
                self._sampling_frequency = float(fr_dataset[()])
            else:
                self._sampling_frequency = float(fr_dataset[0])
        except (KeyError, TypeError) as e:
            warn(f"Could not load sampling frequency: {e}. Setting to default 30.0 Hz")
            self._sampling_frequency = 30.0

        # Load image masks
        self._image_masks = self._image_mask_sparse_read()
        self._background_image_masks = self._background_image_mask_read()

    def _trace_extractor_read(self, field):
        """Read the traces specified by the field from the estimates dataset of the h5py file.

        Parameters
        ----------
        field: str
            The field to read from the estimates object.

        Returns
        -------
        lazy_ops.DatasetView
            The traces specified by the field.
        """
        lazy_ops = get_package(package_name="lazy_ops")

        if field in self._dataset_file["estimates"]:
            try:
                dataset = self._dataset_file["estimates"][field]

                # Check if the dataset is valid (not a scalar object)
                if dataset.shape == () or dataset.dtype == object:
                    warn(f"Field '{field}' is a scalar/object dataset, skipping.")
                    return None

                # Check if dataset has at least 2 dimensions for traces
                if len(dataset.shape) < 2:
                    warn(f"Field '{field}' does not have expected 2D shape for traces: {dataset.shape}")
                    return None

                return lazy_ops.DatasetView(dataset).lazy_transpose()
            except Exception as e:
                warn(f"Error reading field '{field}': {e}")
                return None
        else:
            return None

    def _raw_trace_extractor_read(self):
        """Read the denoised trace and the residual trace from the h5py file and sum them to obtain the raw roi response trace.

        Returns
        -------
        roi_response_raw: numpy.ndarray
            The raw roi response trace.
        """
        try:
            C_dataset = self._dataset_file["estimates"]["C"]
            YrA_dataset = self._dataset_file["estimates"]["YrA"]

            # Check if datasets are valid (not scalar)
            if C_dataset.shape == () or YrA_dataset.shape == ():
                warn("C or YrA dataset is scalar, cannot compute raw traces")
                return None

            # Check if shapes match traces
            if C_dataset.shape != YrA_dataset.shape:
                warn(f"C and YrA shapes don't match: {C_dataset.shape} vs {YrA_dataset.shape}")
                return None

            C = C_dataset[:]
            YrA = YrA_dataset[:]
            roi_response_raw = C + YrA
            return np.array(roi_response_raw.T)
        except (KeyError, ValueError) as e:
            warn(f"Error reading raw traces: {e}")
            return None

    def _correlation_image_read(self):
        """
        Read the mean summary image ("mean image") from the CaImAn output.

        The mean image is a 2D array representing the spatial distribution of background fluorescence
        across the field of view (FOV). In CaImAn, this is computed by summing all background spatial
        components (the "b" matrix in the estimates group) and reshaping the result to match the FOV shape.
        This image is useful for visualizing the overall background signal and for quality control.

        Returns
        -------
        np.ndarray or None
            The mean background image as a 2D array (height, width), or None if not available.
        """
        if "Cn" in self._dataset_file["estimates"]:
            try:
                cn_dataset = self._dataset_file["estimates"]["Cn"]
                # Check if it's a scalar dataset
                if cn_dataset.shape == ():
                    warn("Cn dataset is scalar, skipping")
                    return None
                return np.array(cn_dataset)
            except Exception as e:
                warn(f"Error reading correlation image: {e}")
                return None
        return None

    def _summary_image_read(self):
        """
        Return a 2D mean background image for the field of view (FOV).

        In CaImAn, the background is modeled as spatial components in the "b" matrix.
        This function sums all columns of "b" and reshapes the result to the FOV shape,
        providing a visualization of overall background fluorescence.

        Returns
        -------
        np.ndarray or None
            2D mean background image (height, width), or None if not available.
        """
        if "b" in self._dataset_file["estimates"]:
            try:
                b_dataset = self._dataset_file["estimates"]["b"]
                # Check if it's a scalar dataset
                if b_dataset.shape == () or b_dataset.dtype == object:
                    warn("b dataset is scalar/object (thus empty), skipping mean image")
                    return None

                # Get FOV shape safely
                try:
                    FOV_shape = self._dataset_file["params"]["data"]["dims"][()]
                    if isinstance(FOV_shape, np.ndarray) and FOV_shape.shape == ():
                        FOV_shape = FOV_shape.item()
                    if not isinstance(FOV_shape, (tuple, list, np.ndarray)):
                        warn(f"Invalid FOV shape format: {FOV_shape}")
                        return None
                except (KeyError, ValueError) as e:
                    warn(f"Could not get FOV shape: {e}")
                    return None

                b_sum = b_dataset[:].sum(axis=1)
                return np.array(b_sum).reshape(FOV_shape, order="F")
            except Exception as e:
                warn(f"Error reading summary image: {e}")
                return None
        return None

    def _image_mask_sparse_read(self):
        """Read the image masks from the h5py file.

        Returns
        -------
        image_masks: numpy.ndarray
            The image masks for each ROI.
        """
        roi_ids = self._dataset_file["estimates"]["A"]["indices"]
        masks = self._dataset_file["estimates"]["A"]["data"]
        ids = self._dataset_file["estimates"]["A"]["indptr"]

        frame_shape = self.get_frame_shape()
        image_mask_in = csc_matrix(
            (masks, roi_ids, ids),
            shape=(np.prod(frame_shape), self.get_num_rois()),
        ).toarray()
        image_masks = np.reshape(image_mask_in, (*frame_shape, -1), order="F")
        return image_masks

    def _background_image_mask_read(self):
        """Read the image masks from the h5py file.

        Returns
        -------
        image_masks: numpy.ndarray
            The image masks for each background components.
        """
        b_dataset = self._dataset_file["estimates"].get("b")

        if b_dataset is None:
            return None
        if b_dataset.shape == () or b_dataset.dtype == object:
            return None

        try:
            background_image_masks = np.reshape(b_dataset, (*self.get_frame_shape(), -1), order="F")
            return background_image_masks
        except ValueError as e:
            warn(f"Cannot reshape background components (size {b_dataset.size}): {e}")
            return None

    def get_accepted_list(self):
        """Get list of accepted ROI indices."""
        try:
            accepted_dataset = self._dataset_file["estimates"]["idx_components"]
            # Handle scalar datasets
            if accepted_dataset.shape == ():
                # Scalar dataset - default to all ROIs
                return list(range(self.get_num_rois()))
            else:
                accepted = list(accepted_dataset[:])
                return accepted
        except (KeyError, AttributeError):
            return list(range(self.get_num_rois()))

    def get_rejected_list(self):
        """Get list of rejected ROI indices."""
        try:
            rejected_dataset = self._dataset_file["estimates"]["idx_components_bad"]
            # Handle scalar datasets
            if rejected_dataset.shape == ():
                # Scalar dataset - default to no rejected ROIs
                return []
            else:
                rejected = list(rejected_dataset[:])
                return rejected
        except (KeyError, AttributeError):
            return []

    def get_frame_shape(self):
        """Get frame shape from HDF5 file."""
        try:
            # Try root level dims first
            dims_dataset = self._dataset_file["dims"]

            # Handle scalar datasets
            if dims_dataset.shape == ():
                dims = dims_dataset[()]
            else:
                dims = dims_dataset[:]
        except KeyError:
            # Fallback to params/data/dims
            try:
                dims_dataset = self._dataset_file["params"]["data"]["dims"]
                if dims_dataset.shape == ():
                    dims = dims_dataset[()]
                else:
                    dims = dims_dataset[:]
            except KeyError:
                raise ValueError("Could not find dimensions in HDF5 file")

        # Handle different formats of dims to ensure the returned value is a tuple of two integers
        if isinstance(dims, np.ndarray):
            if dims.ndim == 1 and dims.size == 2:  # 1D array with two elements
                return tuple(int(x) for x in dims.astype(int))
            elif dims.ndim == 0:  # scalar array
                dims_val = dims.item()
                if isinstance(dims_val, (list, tuple, np.ndarray)) and len(dims_val) == 2:
                    return tuple(int(x) for x in dims_val)
        elif isinstance(dims, (tuple, list)) and len(dims) == 2:
            return tuple(int(x) for x in dims)

        raise ValueError(f"Invalid dims format in HDF5 file: {dims}")

    def get_num_rois(self):
        """Get number of ROIs by counting columns in the image mask matrix.

        Returns
        -------
        int
            Number of ROIs (regions of interest).
        """
        try:
            return self._dataset_file["estimates"]["A"]["indptr"].shape[0] - 1
        except (KeyError, AttributeError):
            if hasattr(self, "_roi_response_raw") and self._roi_response_raw is not None:
                return self._roi_response_raw.shape[1]
            return 0

    def get_num_samples(self):
        """Get number of time samples from traces or estimates fields.

        Returns
        -------
        int
            Number of time samples (frames).
        """
        try:
            if hasattr(self, "_roi_response_raw") and self._roi_response_raw is not None:
                return self._roi_response_raw.shape[0]
            else:
                for field in ["C", "S", "F_dff"]:
                    if field in self._dataset_file["estimates"]:
                        return self._dataset_file["estimates"][field].shape[1]
                return 0
        except Exception:
            return 0

    def __del__(self):
        """Close the h5py file when the object is deleted."""
        if hasattr(self, "_dataset_file") and self._dataset_file is not None:
            try:
                self._dataset_file.close()
            except:
                pass

    @staticmethod
    def write_segmentation(segmentation_object: SegmentationExtractor, save_path: PathType, overwrite: bool = True):
        """Write a segmentation object to a .hdf5 or .h5 file specified by save_path.

        Parameters
        ----------
        segmentation_object: SegmentationExtractor
            The segmentation object to be written to file.
        save_path: str
            The path to the file to be written.
        overwrite: bool
            If True, overwrite the file if it already exists.

        Raises
        ------
        FileExistsError
            If the file already exists and overwrite is False.
        """
        warn(
            "The write_segmentation function is deprecated and will be removed on or after September 2025. "
            "ROIExtractors is no longer supporting write operations.",
            DeprecationWarning,
            stacklevel=2,
        )
        save_path = Path(save_path)
        assert save_path.suffix in [".hdf5", ".h5"], "'save_path' must be a *.hdf5 or *.h5 file"

        if save_path.is_file():
            if not overwrite:
                raise FileExistsError("The specified path exists! Use overwrite=True to overwrite it.")
            else:
                save_path.unlink()

        folder_path = save_path.parent
        file_name = save_path.name

        # Handle MultiSegmentationExtractor
        if isinstance(segmentation_object, MultiSegmentationExtractor):
            segext_objs = segmentation_object.segmentations
            for plane_num, segext_obj in enumerate(segext_objs):
                save_path_plane = folder_path / f"Plane_{plane_num}" / file_name
                CaimanSegmentationMultiExtractor.write_segmentation(segext_obj, save_path_plane)
            return

        if not folder_path.is_dir():
            folder_path.mkdir(parents=True)

        with h5py.File(save_path, "a") as f:
            # Create base groups
            estimates = f.create_group("estimates")
            params = f.create_group("params")

            # Add traces
            if segmentation_object.get_traces(name="denoised") is not None:
                estimates.create_dataset("C", data=segmentation_object.get_traces(name="denoised"))
            if segmentation_object.get_traces(name="neuropil") is not None:
                estimates.create_dataset("f", data=segmentation_object.get_traces(name="neuropil"))
            if segmentation_object.get_traces(name="dff") is not None:
                estimates.create_dataset("F_dff", data=segmentation_object.get_traces(name="dff"))
            if segmentation_object.get_traces(name="deconvolved") is not None:
                estimates.create_dataset("S", data=segmentation_object.get_traces(name="deconvolved"))
            if segmentation_object.get_image("correlation") is not None:
                estimates.create_dataset("Cn", data=segmentation_object.get_image("correlation"))

            estimates.create_dataset(
                "idx_components",
                data=np.array(
                    [] if segmentation_object.get_accepted_list() is None else segmentation_object.get_accepted_list()
                ),
            )
            estimates.create_dataset(
                "idx_components_bad",
                data=np.array(
                    [] if segmentation_object.get_rejected_list() is None else segmentation_object.get_rejected_list()
                ),
            )

            # Add image masks
            image_mask_data = np.reshape(
                segmentation_object.get_roi_image_masks(),
                [-1, segmentation_object.get_num_rois()],
                order="F",
            )
            image_mask_csc = csc_matrix(image_mask_data)
            estimates.create_dataset("A/data", data=image_mask_csc.data)
            estimates.create_dataset("A/indptr", data=image_mask_csc.indptr)
            estimates.create_dataset("A/indices", data=image_mask_csc.indices)
            estimates.create_dataset("A/shape", data=image_mask_csc.shape)

            # Add params
            params.create_dataset("data/fr", data=segmentation_object.get_sampling_frequency())
            params.create_dataset("data/dims", data=segmentation_object.get_frame_shape())
            f.create_dataset("dims", data=segmentation_object.get_frame_shape())
