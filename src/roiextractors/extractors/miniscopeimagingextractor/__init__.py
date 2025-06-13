"""An ImagingExtractor for the Miniscope video (.avi) format.

Modules
-------
miniscopeimagingextractor
    An ImagingExtractor for the Miniscope video (.avi) format.

Classes
-------
MiniscopeImagingExtractor
    An ImagingExtractor for the Miniscope video (.avi) format.
"""

from .miniscopeimagingextractor import MiniscopeImagingExtractor, MiniscopeMultiRecordingImagingExtractor
from .miniscope_utils import (
    get_miniscope_files_from_multi_recordings_subfolders,
    get_miniscope_files_from_direct_folder,
    validate_miniscope_files,
    load_miniscope_config,
)
