"""MultiTiffMultiPageImagingExtractor class."""

from .tiffimagingextractor import TiffImagingExtractor
from ...multiimagingextractor import MultiImagingExtractor
from ...utils import match_paths


class MultiTiffMultiPageImagingExtractor(MultiImagingExtractor):
    """A ImagingExtractor for multiple TIFF files that each have multiple pages."""

    extractor_name = "multi-tiff multi-page Imaging Extractor"
    is_writable = False

    def __init__(self, folder_path: str, pattern: str, sampling_frequency: float):
        """Create a MultiTiffMultiPageImagingExtractor instance.

        Parameters
        ----------
        folder_path : str
            List of path to each TIFF file.
        pattern : str
            F-string-style pattern to match the TIFF files.
        sampling_frequency : float
            The frequency at which the frames were sampled, in Hz.
        """

        self.folder_path = folder_path
        self.tif_paths = match_paths(folder_path, pattern)
        if len(self.tif_paths) == 0:
            raise ValueError("No TIFF files found in the folder_path with the given pattern.")
        imaging_extractors = [
            TiffImagingExtractor(file_path=x, sampling_frequency=sampling_frequency) for x in self.tif_paths
        ]
        super().__init__(imaging_extractors=imaging_extractors)
        self._kwargs.update({"folder_path": folder_path})
