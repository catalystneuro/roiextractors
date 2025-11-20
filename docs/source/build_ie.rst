Build an ImagingExtractor:
==========================

To build your custom ImagingExtractor to interface with a new raw image storage format from your microscope, build a custom class inheriting from a base class that enforces certain methods that need to be defined:

* `get_num_samples()`: the number of image frames recorded
* `get_image_shape()`: the y,x dimensions of a single frame
* `get_series()`: return a continuous slice of imaging data



.. code-block:: python

    from roiextractors import ImagingExtractor

    class MyFormatImagingExtractor(ImagingExtractor):
        def __init__(self, file_path, sampling_frequency: float = None):
            ImagingExtractor.__init__(self)

            ## All file specific initialization code can go here.

            self.sampling_frequency = sampling_frequency or # define your own method of extraction
            self._data = self._load_data()  # groups is a list or a np.array with length num_channels

        def _load_data(self):

            # define a function that retrieves an in memory array from your native imaging format
            # returns: np.ndarray (shape: frames, image_dims)

        def get_num_samples(self) -> int:

            # Fill code to get the number of frames (samples) in the recordings.
            # Note that if the data is volumetric, num_samples corresponds to the number of volumes, not the number of 2D images.

            return num_samples

        def get_sampling_frequency(self) -> float:

            return self.sampling_frequency

        def get_series(self, start_sample: Optional[int] = None, end_sample: Optional[int] = None) -> NumpyArray:

            # define a method to read a continuous slice of imaging data
            return self._data[start_sample:end_sample]

        def get_image_shape(self) -> Tuple[int, int]:

            # returns the (height, width) dimensions of a single frame
            # this is the abstract method that must be implemented
            return self._data.shape[1:]
