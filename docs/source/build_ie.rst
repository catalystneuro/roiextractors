Build an ImagingExtractor:
==========================

To build your custom ImagingExtractor to interface with a new raw image storage format from your microscope, build a custom class inheriting from a base class that enforces certain methods that need to be defined:

* `get_channel_names()`: returns a list of channel names
* `get_num_channels()`: returns the number of channels used in the recording
* `get_num_frames()`: the number of image frames recorded
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
            self._channel_names = [f'channel_{ch}' for ch in range(self.get_num_channels())] # write logic to get channel names

        def _load_data(self):

            # define a function that retrives an in memory array from your native imaging format
            # returns: np.ndarray (shape: frames, image_dims)

        def get_channel_names(self):

            # Fill code to get a list of channel_ids. If channel ids are not specified, you can use:
            # channel_ids = range(num_channels)

            return self._channel_names

        def get_num_channels(self) -> int:
            # define method to find the number of channels
            # returns int

        def get_num_frames(self):

            # Fill code to get the number of frames (samples) in the recordings.

            return num_frames

        def get_sampling_frequency():

            return self.sampling_frequency

        def get_frames(self, frame_idxs: ArrayType, channel: int = 0) -> NumpyArray:

            # define a method to read a frame from the given frame numbers requested.
            return self._data[frame_idxs]

        def get_image_size(self)

            # returns something like self._data.shape[1:]

        @staticmethod
        def write_imaging(imaging_obj, save_path, other_params):
            '''
            This is an example of a function that is not abstract so it is optional if you want to override it.
            It allows other ImageExtractors to use your new ImageExtractors to convert their  data into your this file format.
            '''
