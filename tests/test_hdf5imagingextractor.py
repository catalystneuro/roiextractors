"""Tests for `Hdf5ImagingExtractor`."""

from roiextractors import Hdf5ImagingExtractor

from .setup_paths import OPHYS_DATA_PATH

HDF5_FILE_PATH = OPHYS_DATA_PATH / "imaging_datasets" / "hdf5" / "demoMovie.hdf5"


def test_metadata_argument_in_extra_metadata():
    """The `metadata` argument of the constructor is under `"hdf5_user_metadata"` in `extra_metadata`."""
    user_metadata = {"experimenter": "Experimenter name"}

    extractor = Hdf5ImagingExtractor(file_path=HDF5_FILE_PATH, metadata=user_metadata)

    assert extractor.metadata.extra_metadata == {"hdf5_user_metadata": user_metadata}


def test_metadata_without_user_metadata():
    """Without a `metadata` argument or attribute, `extra_metadata` is empty."""
    # The `mov` dataset of the fixture has only the `fr` attribute
    extractor = Hdf5ImagingExtractor(file_path=HDF5_FILE_PATH)

    assert extractor.metadata.extra_metadata == {}
