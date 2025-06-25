"""A SegmentationExtractor for CaImAn.

Classes
-------
CaimanSegmentationExtractor
    A class for extracting segmentation from CaImAn output.
"""

from pathlib import Path
from warnings import warn
import warnings

import h5py

from scipy.sparse import csc_matrix
import numpy as np

from ...extraction_tools import PathType, get_package
from ...multisegmentationextractor import MultiSegmentationExtractor
from ...segmentationextractor import SegmentationExtractor


class CaimanSegmentationExtractor(SegmentationExtractor):
    """A SegmentationExtractor for CaImAn.

    This class inherits from the SegmentationExtractor class, having all
    its functionality specifically applied to the dataset output from
    the 'CaImAn' ROI segmentation method.

    CaImAn (Calcium Imaging Analysis) is a computational toolbox for large scale
    calcium imaging data analysis and behavioral analysis. This extractor provides
    access to the rich output of CaImAn's analysis pipeline stored in HDF5 format.

    The CaImAn estimates object contains the following key components:

    Spatial and Temporal Components:
        A : scipy.sparse.csc_matrix (# pixels x # components)
            Spatial footprints of identified components. Each column represents
            a component's spatial footprint, flattened with order='F'.
        C : np.ndarray (# components x # timesteps)
            Temporal traces (denoised and deconvolved) for each component.
        b : np.ndarray (# pixels x # background components)
            Spatial background components, flattened with order='F'.
        f : np.ndarray (# background components x # timesteps)
            Temporal background components.

    Neural Activity:
        S : np.ndarray (# components x # timesteps)
            Deconvolved neural activity (spikes) for each component.
        F_dff : np.ndarray (# components x # timesteps)
            DF/F normalized temporal components (2p data only).
        YrA : np.ndarray (# components x # timesteps)
            Residual traces after denoising.

    Quality Assessment:
        SNR_comp : np.ndarray (# components,)
            Signal-to-noise ratio for each component.
        r_values : np.ndarray (# components,)
            Spatial correlation values for each component.
        cnn_preds : np.ndarray (# components,)
            CNN-based classifier predictions (0-1, neuron-like probability).
        idx_components : list
            Indices of accepted components.
        idx_components_bad : list
            Indices of rejected components.

    Component Properties:
        center : list (# components,)
            Centroid coordinates for each spatial footprint.
        coordinates : list (# components,)
            Contour coordinates for each spatial footprint.
        g : np.ndarray (# components, p)
            Autoregressive time constants for each trace.
        bl : np.ndarray (# components,)
            Baseline values for each trace.
        c1 : np.ndarray (# components,)
            Initial calcium concentration for each trace.
        neurons_sn : np.ndarray (# components,)
            Noise standard deviation for each trace.

    Background and Noise:
        b0 : np.ndarray (# pixels,)
            Constant baseline for each pixel (1p data).
        sn : np.ndarray (# pixels,)
            Noise standard deviation for each pixel.
        W : scipy.sparse matrix (# pixels x # pixels)
            Ring model matrix for background computation (1p data).

    Summary Images:
        Cn : np.ndarray (height, width)
            Local correlation image.

    Caiman parameters:
        The params group contains all analysis parameters organized by category:
        - data: Dataset properties (dimensions, frame rate, decay time)
        - init: Component initialization parameters
        - motion: Motion correction parameters
        - quality: Component evaluation thresholds
        - spatial/temporal: Processing parameters
        - online: OnACID algorithm parameters

    Notes
    -----
        Some fields may be stored as scalar values in the HDF5 file when they
        are not available or not computed. This extractor will detect such cases
        and return None for those fields.

        At the moment (June, 2025), Caimn does not keep documentation of their output format. Looking at the
        source what they do is to transform the cnmfe class to a dict with the dunder method (`__dict__`) and
        save this as an HDF5 file:

        https://github.com/flatironinstitute/CaImAn/blob/881e627adf951dde25d3839953c98acf6b4adab0/caiman/source_extraction/cnmf/cnmf.py#L655-L667

        This might change in the future, so please check the CaImAn documentation.
    """

    extractor_name = "CaimanSegmentation"
    mode = "file"

    def __init__(self, file_path: PathType):
        """Initialize a CaimanSegmentationExtractor instance.

        Parameters
        ----------
        file_path: str
            The location of the HDF5 file containing CaImAn analysis output.

        Notes
        -----
        The extractor will automatically detect which data types are available
        in the HDF5 file. This allows for compatibility with different CaImAn
        versions and analysis configurations.
        """
        SegmentationExtractor.__init__(self)
        self.file_path = file_path
        self._dataset_file = self._file_extractor_read()

        # Create handles to main groups for better readability
        self._estimates = self._dataset_file["estimates"]
        self._params = self._dataset_file["params"]

        # Core traces and images
        self._roi_response_raw = self._raw_trace_extractor_read()
        self._roi_response_dff = self._trace_extractor_read("F_dff")
        self._roi_response_denoised = self._trace_extractor_read("C")
        self._roi_response_neuropil = self._trace_extractor_read("f")
        self._roi_response_deconvolved = self._trace_extractor_read("S")
        self._image_correlation = self._correlation_image_read()
        self._image_mean = self._summary_image_read()

        # Sampling frequency and spatial information
        self._sampling_frequency = self._params["data"]["fr"][()]
        self._image_masks = self._image_mask_sparse_read()
        self._background_image_masks = self._background_image_mask_read()

    def __del__(self):  # TODO: refactor segmentation extractors who use __del__ together into a base class
        """Close the h5py file when the object is deleted."""
        self._dataset_file.close()

    def _is_scalar_dataset(self, dataset) -> bool:
        """Check if a dataset in the HDF5 file is a scalar value.

        Parameters
        ----------
        dataset : h5py.Dataset
            The HDF5 dataset to check.

        Returns
        -------
        bool
            True if the dataset is scalar, False otherwise.
        """
        return len(dataset.shape) == 0 or (len(dataset.shape) == 1 and dataset.shape[0] == 0)

    def _get_sparse_dataset_safe(self, base_path: str):
        """Get sparse matrix dataset, returning None for scalar values.

        Parameters
        ----------
        base_path : str
            Base path to the sparse matrix group in HDF5 file.

        Returns
        -------
        scipy.sparse.csc_matrix or None
            The sparse matrix if available, None if scalar or missing.
        """
        if (
            self._is_scalar_dataset(self._dataset_file[f"{base_path}/data"])
            or self._is_scalar_dataset(self._dataset_file[f"{base_path}/indices"])
            or self._is_scalar_dataset(self._dataset_file[f"{base_path}/indptr"])
        ):
            return None

        data = self._dataset_file[f"{base_path}/data"][:]
        indices = self._dataset_file[f"{base_path}/indices"][:]
        indptr = self._dataset_file[f"{base_path}/indptr"][:]
        shape = tuple(self._dataset_file[f"{base_path}/shape"][:])

        return csc_matrix((data, indices, indptr), shape=shape)

    def _file_extractor_read(self):
        """Read the h5py file.

        Returns
        -------
        h5py.File
            The h5py file object specified by self.file_path.
        """
        return h5py.File(self.file_path, "r")

    def _image_mask_sparse_read(self):
        """Read the image masks from the h5py file.

        Returns
        -------
        image_masks: numpy.ndarray or None
            The image masks for each ROI, or None if not available.
        """
        sparse_matrix = self._get_sparse_dataset_safe("estimates/A")
        if sparse_matrix is not None:
            image_mask_in = sparse_matrix.toarray()
            image_masks = np.reshape(image_mask_in, (*self.get_frame_shape(), -1), order="F")
            return image_masks
        return None

    def _background_image_mask_read(self):
        """Read the background image masks from the h5py file.

        Returns
        -------
        image_masks: numpy.ndarray or None
            The image masks for each background component, or None if not available.
        """
        if "b" in self._estimates and not self._is_scalar_dataset(self._estimates["b"]):
            background_data = np.array(self._estimates["b"])
            background_image_masks = np.reshape(background_data, (*self.get_frame_shape(), -1), order="F")
            return background_image_masks
        return None

    def _trace_extractor_read(self, field):
        """Read the traces specified by the field from the estimates dataset of the h5py file.

        Parameters
        ----------
        field: str
            The field to read from the estimates object.

        Returns
        -------
        lazy_ops.DatasetView or None
            The traces specified by the field, or None if not available.
        """
        lazy_ops = get_package(package_name="lazy_ops")

        # Check if field exists and is not scalar
        if field in self._estimates and not self._is_scalar_dataset(self._estimates[field]):
            return lazy_ops.DatasetView(self._estimates[field]).lazy_transpose()

        return None

    def _raw_trace_extractor_read(self):
        """Read the denoised trace and the residual trace from the h5py file and sum them to obtain the raw roi response trace.

        Returns
        -------
        roi_response_raw: numpy.ndarray or None
            The raw roi response trace, or None if required data is not available.
        """
        # Check if both required datasets are available and not scalar
        if (
            "C" in self._estimates
            and not self._is_scalar_dataset(self._estimates["C"])
            and "YrA" in self._estimates
            and not self._is_scalar_dataset(self._estimates["YrA"])
        ):

            denoised_traces = self._estimates["C"][:]
            residual_traces = self._estimates["YrA"][:]
            roi_response_raw = denoised_traces + residual_traces
            return np.array(roi_response_raw.T)

        return None

    def _correlation_image_read(self):
        """Read correlation image Cn.

        Returns
        -------
        numpy.ndarray or None
            Local correlation image, or None if not available.
        """
        if "Cn" in self._estimates and not self._is_scalar_dataset(self._estimates["Cn"]):
            return np.array(self._estimates["Cn"])
        return None

    def _summary_image_read(self):
        """Read summary image from background components.

        Returns
        -------
        numpy.ndarray or None
            Summary image computed from background components, or None if not available.
        """
        if "b" in self._estimates and not self._is_scalar_dataset(self._estimates["b"]):
            background_data = np.array(self._estimates["b"])
            FOV_shape = self._params["data"]["dims"][()]
            b_sum = background_data.sum(axis=1)
            return np.array(b_sum).reshape(FOV_shape, order="F")
        return None

    def get_accepted_list(self):
        """Get list of accepted component indices.

        Returns
        -------
        list
            List of indices for components that passed quality assessment.
            If no quality assessment was performed, returns all component indices.
        """
        if "idx_components" in self._estimates and not self._is_scalar_dataset(self._estimates["idx_components"]):
            return list(self._estimates["idx_components"][:])
        # If no quality assessment, assume all components are accepted
        return list(range(self.get_num_rois()))

    def get_rejected_list(self):
        """Get list of rejected component indices.

        Returns
        -------
        list
            List of indices for components that failed quality assessment.
            Returns empty list if no quality assessment was performed.
        """
        if "idx_components_bad" in self._estimates and not self._is_scalar_dataset(
            self._estimates["idx_components_bad"]
        ):
            return list(self._estimates["idx_components_bad"][:])
        return []

    def get_frame_shape(self) -> tuple:
        return tuple(self._params["data"]["dims"][()])

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
            "The write_segmentation function is deprecated and will be removed on or after September 2025. ROIExtractors is no longer supporting write operations.",
            DeprecationWarning,
            stacklevel=2,
        )
        save_path = Path(save_path)
        assert save_path.suffix in [
            ".hdf5",
            ".h5",
        ], "'save_path' must be a *.hdf5 or *.h5 file"
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
                CaimanSegmentationExtractor.write_segmentation(segext_obj, save_path_plane)
        if not folder_path.is_dir():
            folder_path.mkdir(parents=True)

        with h5py.File(save_path, "a") as f:
            # create base groups:
            estimates = f.create_group("estimates")
            params = f.create_group("params")
            # adding to estimates:
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

            # adding image_masks:
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

            # adding params:
            params.create_dataset("data/fr", data=segmentation_object.get_sampling_frequency())
            params.create_dataset("data/dims", data=segmentation_object.get_image_size())
            f.create_dataset("dims", data=segmentation_object.get_image_size())

    def get_image_size(self):
        warnings.warn(
            "get_image_size is deprecated and will be removed on or after January 2026. "
            "Use get_frame_shape instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_frame_shape()
