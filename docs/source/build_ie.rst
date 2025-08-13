Build an ImagingExtractor:
==========================

To build your custom ImagingExtractor to interface with a new raw image storage format from your microscope, build a custom class inheriting from a base class that enforces certain methods that need to be defined:

* `get_channel_names()`: returns a list of channel names
* `get_num_samples()`: the number of image frames recorded
* `get_image_size()`: the y,x dim of the image(the resolution)
* `get_frames()`: return specific requested image frames



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

        def get_channel_names(self):

            # Fill code to get a list of channel_ids. If channel ids are not specified, you can use:
            # channel_ids = range(num_channels)

            return self._channel_names

        def get_num_samples(self):

            # Fill code to get the number of frames (samples) in the recordings.

            return num_frames

        def get_sampling_frequency():

            return self.sampling_frequency

        def get_frames(self, frame_idxs: ArrayType, channel: int = 0) -> NumpyArray:

            # define a method to read a frame from the given frame numbers requested.
            return self._data[frame_idxs]

        def get_frame_shape(self)

            # returns something like self._data.shape[1:]
