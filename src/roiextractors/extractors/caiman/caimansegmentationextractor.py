"""A SegmentationExtractor for CaImAn.

Classes
-------
CaimanSegmentationExtractor
    A class for extracting segmentation from CaImAn output.
"""

import warnings

import h5py
import numpy as np
from scipy.sparse import csc_matrix

from ...extraction_tools import PathType, get_package
from ...segmentationextractor import (
    SegmentationExtractor,
    _ROIMasks,
    _RoiResponse,
)


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

        Quality metrics (SNR, spatial correlation values, CNN predictions) are
        automatically stored as properties during initialization if available.
        """
        SegmentationExtractor.__init__(self)
        self.file_path = file_path
        self._dataset_file = self._file_extractor_read()

        # Create handles to main groups for better readability
        self._estimates = self._dataset_file["estimates"]
        self._params = self._dataset_file["params"]

        # Core traces and images
        cell_ids: list[int] | None = None

        raw_traces = self._raw_trace_extractor_read()
        if raw_traces is not None:
            cell_ids = list(range(raw_traces.shape[1]))
            self._roi_responses.append(_RoiResponse("raw", raw_traces, cell_ids))

        dff_traces = self._trace_extractor_read("F_dff")
        if dff_traces is not None:
            if cell_ids is None:
                cell_ids = list(range(dff_traces.shape[1]))
            self._roi_responses.append(_RoiResponse("dff", dff_traces, cell_ids))

        denoised_traces = self._trace_extractor_read("C")
        if denoised_traces is not None:
            if cell_ids is None:
                cell_ids = list(range(denoised_traces.shape[1]))
            self._roi_responses.append(_RoiResponse("denoised", denoised_traces, cell_ids))

        deconvolved_traces = self._trace_extractor_read("S")
        if deconvolved_traces is not None:
            if cell_ids is None:
                cell_ids = list(range(deconvolved_traces.shape[1]))
            self._roi_responses.append(_RoiResponse("deconvolved", deconvolved_traces, cell_ids))

        background_traces = self._trace_extractor_read("f")
        if background_traces is not None:
            background_ids = [f"background{index}" for index in range(background_traces.shape[1])]
            self._roi_responses.append(_RoiResponse("background", background_traces, background_ids))

        if cell_ids is not None:
            self._roi_ids = list(cell_ids)

        correlation_image = self._correlation_image_read()
        if correlation_image is not None:
            self._summary_images["correlation"] = correlation_image

        mean_image = self._summary_image_read()
        if mean_image is not None:
            self._summary_images["mean"] = mean_image

        # Sampling frequency and spatial information
        self._sampling_frequency = self._params["data"]["fr"][()]

        # Create ROI representations from CaImAn sparse matrices
        self._roi_masks = self._create_roi_masks()

        # Store quality metrics as properties
        self._set_quality_metrics_as_properties()

    def __del__(self):  # TODO: refactor segmentation extractors who use __del__ together into a base class
        """Close the h5py file when the object is deleted."""
        self._dataset_file.close()

    def _create_roi_masks(self) -> _ROIMasks | None:
        """Create ROI representations from CaImAn CSC sparse matrices.

        Converts CaImAn's native CSC matrix format to NWB-compatible pixel mask format.
        Combines cell and background ROIs into a single container.

        Returns
        -------
        _ROIMasks or None
            Container with all ROI masks in nwb-pixel_mask format, or None if no masks available.
        """
        # Get cell masks from sparse matrix A
        cell_sparse_matrix = self._get_sparse_dataset_safe("estimates/A")
        if cell_sparse_matrix is None:
            return None

        height, width = self.get_frame_shape()
        num_cells = cell_sparse_matrix.shape[1]

        # Convert CSC matrix to per-ROI pixel masks
        pixel_masks = []
        roi_id_map = {}

        # Process cell ROIs
        for index in range(num_cells):
            col = cell_sparse_matrix[:, index]
            nonzero_flat_indices = col.nonzero()[0]
            weights = col.data

            # Convert flat Fortran-order indices to (y, x) coordinates
            # In Fortran order: flat_index = y + x * height
            y_coords = nonzero_flat_indices % height
            x_coords = nonzero_flat_indices // height

            pixel_mask = np.column_stack([y_coords, x_coords, weights])
            pixel_masks.append(pixel_mask)

            # Map cell_id to index
            if self._roi_ids is not None and index < len(self._roi_ids):
                cell_id = self._roi_ids[index]
            else:
                cell_id = index
            roi_id_map[cell_id] = index

        # Process background components if available
        if "b" in self._estimates and not self._is_scalar_dataset(self._estimates["b"]):
            background_data = np.array(self._estimates["b"])  # Shape: (n_pixels, n_backgrounds)
            num_backgrounds = background_data.shape[1] if len(background_data.shape) > 1 else 1

            if num_backgrounds == 1 and len(background_data.shape) == 1:
                # Single background component as 1D array
                background_data = background_data.reshape(-1, 1)

            for bg_index in range(num_backgrounds):
                bg_flat = background_data[:, bg_index]
                nonzero_indices = np.nonzero(bg_flat)[0]

                # Convert flat Fortran-order indices to (y, x) coordinates
                y_coords = nonzero_indices % height
                x_coords = nonzero_indices // height
                weights = bg_flat[nonzero_indices]

                pixel_mask = np.column_stack([y_coords, x_coords, weights])
                pixel_masks.append(pixel_mask)

                # Background IDs match trace naming (e.g., "background0", "background1")
                bg_id = f"background{bg_index}"
                roi_id_map[bg_id] = len(pixel_masks) - 1

        return _ROIMasks(
            data=pixel_masks,
            mask_tpe="nwb-pixel_mask",
            field_of_view_shape=(height, width),
            roi_id_map=roi_id_map,
        )

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

    # Quality Metrics
    def _get_snr_values(self) -> np.ndarray | None:
        """Get signal-to-noise ratio for each component.

        Returns
        -------
        numpy.ndarray or None
            SNR values for each component, or None if not available.
        """
        if self._dataset_file["estimates"].get("SNR_comp"):
            snr_data = self._dataset_file["estimates"]["SNR_comp"]
            if snr_data.shape != ():
                return np.array(snr_data)
        return None

    def _get_spatial_correlation_values(self) -> np.ndarray | None:
        """Get spatial correlation values (r_values) for each component.

        Returns
        -------
        numpy.ndarray or None
            Spatial correlation values for each component, or None if not available.
        """
        if self._dataset_file["estimates"].get("r_values"):
            r_data = self._dataset_file["estimates"]["r_values"]
            if r_data.shape != ():
                return np.array(r_data)
        return None

    def _get_cnn_predictions(self) -> np.ndarray | None:
        """Get CNN classifier predictions for component quality.

        Note
        ----
        CNN predictions require special handling because CaImAn stores
        a Python None object when CNN classification is not used or unavailable.
        HDF5 serializes this as a string 'NoneType', which h5py reads back as
        array(b'NoneType', dtype=object).

        Returns
        -------
        numpy.ndarray or None
            CNN predictions for each component, or None if not available.
        """
        if self._dataset_file["estimates"].get("cnn_preds"):
            cnn_data = self._dataset_file["estimates"]["cnn_preds"]
            if cnn_data.size > 0:  # Check if not empty
                data_array = np.array(cnn_data)
                # Check if the data is actually a serialized 'NoneType'
                if (
                    data_array.shape == ()
                    and isinstance(data_array.item(), (bytes, str))
                    and str(data_array.item()).lower() in ["b'nonetype'", "nonetype", "b'nonetype'"]
                ):
                    return None
                return data_array
        return None

    def _set_quality_metrics_as_properties(self):
        """Store quality metrics as properties if available.

        This method is called during initialization to automatically store
        any available quality metrics (SNR, spatial correlation values, CNN predictions)
        as properties that can be accessed via the property interface.
        """
        roi_ids = self.get_roi_ids()

        # Set SNR values as property if available
        snr_values = self._get_snr_values()
        if snr_values is not None and len(snr_values) == len(roi_ids):
            self.set_property(key="snr", values=snr_values, ids=roi_ids)

        # Set spatial correlation values as property if available
        r_values = self._get_spatial_correlation_values()
        if r_values is not None and len(r_values) == len(roi_ids):
            self.set_property(key="r_values", values=r_values, ids=roi_ids)

        # Set CNN predictions as property if available
        cnn_preds = self._get_cnn_predictions()
        if cnn_preds is not None and len(cnn_preds) == len(roi_ids):
            self.set_property(key="cnn_preds", values=cnn_preds, ids=roi_ids)

    def get_image_size(self):
        warnings.warn(
            "get_image_size is deprecated and will be removed on or after January 2026. "
            "Use get_frame_shape instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_frame_shape()

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        """Retrieve the original unaltered timestamps for the data in this interface.

        Returns
        -------
        timestamps: numpy.ndarray or None
            The timestamps for the data stream, or None if native timestamps are not available.
        """
        # CaImAn segmentation data does not have native timestamps
        return None
