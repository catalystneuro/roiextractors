"""Imaging and Segmenation Extractors for .npy files.

Modules
-------
numpyextractors
    Imaging and Segmenation Extractors for .npy files.

Classes
-------
NumpyImagingExtractor
    An ImagingExtractor specified by timeseries .npy file, sampling frequency, and channel names.
NumpySegmentationExtractor
    A Segmentation extractor specified by image masks and traces .npy files.
"""

from .numpyextractors import NumpyImagingExtractor, NumpySegmentationExtractor
