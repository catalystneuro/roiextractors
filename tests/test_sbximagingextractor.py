import numpy as np
from numpy.testing import assert_array_equal

from roiextractors import SbxImagingExtractor

from .extractor_mixins import ImagingExtractorTestMixin
from .setup_paths import OPHYS_DATA_PATH


class TestSbxImagingExtractorSample(ImagingExtractorTestMixin):
    extractor_cls = SbxImagingExtractor
    extractor_kwargs = dict(file_path=OPHYS_DATA_PATH / "imaging_datasets" / "Scanbox" / "sample.sbx")
    expected_num_samples = 10  # the size of the .sbx file over the bytes of one 512 x 796 uint16 frame
    expected_sample_shape = (512, 796)  # info.sz in the .mat file
    expected_dtype = np.dtype("uint16")
    expected_native_timestamps = None
    # info.resfreq / info.config.lines, as info.scanmode is 1 (unidirectional)
    expected_sampling_frequency = 7916 / 512
    expected_metadata = {}

    def check_series(self, series):
        """The series is the .sbx file read with `numpy`, frame by frame, with Scanbox's inverted values."""
        file_path = self.extractor_kwargs["file_path"]
        raw = np.fromfile(file_path, dtype="uint16").reshape(self.expected_num_samples, *self.expected_sample_shape)
        expected_series = np.iinfo("uint16").max - raw
        assert_array_equal(series, expected_series)
