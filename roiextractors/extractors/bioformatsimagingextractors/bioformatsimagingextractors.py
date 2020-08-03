import numpy as np
from .bioformatsimagingbaseextractor import BioformatsImagingExtractor


class FliImagingExtractor(BioformatsImagingExtractor):
    def __init__(self, file_path):
        BioformatsImagingExtractor.__init__(self, file_path)
