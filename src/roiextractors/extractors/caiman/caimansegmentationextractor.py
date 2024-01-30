"""A SegmentationExtractor for CaImAn.

Classes
-------
CaimanSegmentationExtractor
    A class for extracting segmentation from CaImAn output.
"""

from pathlib import Path

try:
    import h5py

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

try:
    from scipy.sparse import csc_matrix

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

import numpy as np

from ...extraction_tools import PathType, get_package
from ...multisegmentationextractor import MultiSegmentationExtractor
from ...segmentationextractor import SegmentationExtractor


class CaimanSegmentationExtractor(SegmentationExtractor):
    """A SegmentationExtractor for CaImAn.

    This class inherits from the SegmentationExtractor class, having all
    its functionality specifically applied to the dataset output from
    the 'CaImAn' ROI segmentation method.
    """

    extractor_name = "CaimanSegmentation"
    installed = HAVE_H5PY and HAVE_SCIPY  # check at class level if installed or not
    is_writable = True
    mode = "file"
    # error message when not installed
    installation_mesg = "To use the CaimanSegmentationExtractor install h5py and scipy: \n\n pip install scipy/h5py\n\n"

    def __init__(self, file_path: PathType):
        """Initialize a CaimanSegmentationExtractor instance.

        Parameters
        ----------
        file_path: str
            The location of the folder containing caiman *.hdf5 output file.
        """
        SegmentationExtractor.__init__(self)
        self.file_path = file_path
        self._dataset_file = self._file_extractor_read()
        self._roi_response_dff = self._trace_extractor_read("F_dff")
        self._roi_response_neuropil = self._trace_extractor_read("C")
        self._roi_response_deconvolved = self._trace_extractor_read("S")
        self._image_correlation = self._summary_image_read()
        self._sampling_frequency = self._dataset_file["params"]["data"]["fr"][()]
        self._image_masks = self._image_mask_sparse_read()

    def __del__(self):  # TODO: refactor segmentation extractors who use __del__ together into a base class
        """Close the h5py file when the object is deleted."""
        self._dataset_file.close()

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
        image_masks: numpy.ndarray
            The image masks for each ROI.
        """
        roi_ids = self._dataset_file["estimates"]["A"]["indices"]
        masks = self._dataset_file["estimates"]["A"]["data"]
        ids = self._dataset_file["estimates"]["A"]["indptr"]
        image_mask_in = csc_matrix(
            (masks, roi_ids, ids),
            shape=(np.prod(self.get_image_size()), self.get_num_rois()),
        ).toarray()
        image_masks = np.reshape(image_mask_in, (*self.get_image_size(), -1), order="F")
        return image_masks

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
            return lazy_ops.DatasetView(self._dataset_file["estimates"][field]).lazy_transpose()

    def _summary_image_read(self):
        """Read the summary image (Cn) from the estimates dataset of the h5py file."""
        if self._dataset_file["estimates"].get("Cn"):
            return np.array(self._dataset_file["estimates"]["Cn"])

    def get_accepted_list(self):
        accepted = self._dataset_file["estimates"]["idx_components"]
        if len(accepted.shape) == 0:
            accepted = list(range(self.get_num_rois()))
        else:
            accepted = list(accepted[:])
        return accepted

    def get_rejected_list(self):
        rejected = self._dataset_file["estimates"]["idx_components_bad"]
        if len(rejected.shape) == 0:
            rejected = list()
        else:
            rejected = list(rejected[:])
        return rejected

    @staticmethod
    def write_segmentation(segmentation_object, save_path, overwrite=True):
        """Write a segmentation object to a *.hdf5 or *.h5 file specified by save_path.

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
            if segmentation_object.get_traces(name="neuropil") is not None:
                estimates.create_dataset("C", data=segmentation_object.get_traces(name="neuropil"))
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
        return self._dataset_file["params"]["data"]["dims"][()]
