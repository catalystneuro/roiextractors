
Build a SegmentationExtractor:
==============================

To build a custom SegmentationExtractor that interfaces with the output of a custom segmentation pipeline used for post processing the raw image data.

* `get_accepted_list()`: returns a list of accepted ROIs' id numbers
* `get_image_size()`: the y,x dim of the image(the resolution)


.. code-block:: python

    from roiextractors import SegmentationExtractor

    class MyFormatSegmentationExtractor(SegmentationExtractor):
        def __init__(self, file_path):
            RecordingExtractor.__init__(self)

            ## All file specific initialization code can go here.

            self._sampling_frequency = # logic to extract sampling frequency here
            self._channel_names = ['OpticalChannel']
            self._num_planes = 1
            self._image_masks = self._load_rois()
            self._roi_response_raw = self._load_traces()# define a method to extract Flourescence traces
            self._roi_response_dff = self._load_traces()# define a method to extract dF/F traces if any else None
            self._roi_response_neuropil = self._load_traces()# define a method to extract neuropil info if any else None
            self._roi_response_deconvolved = self._load_traces() # define a method to extract deconvolved traces if any else None
            self._image_correlation = self._load_summary_images()# define method to extract a correlation image else None
            self._image_mean = self._load_summary_images() # define method to extract a mean image else None

        def _load_traces(self, name):

            # define your logic to extract roi time traces
            # return a np.ndarray, None

        def _load_rois(self):

            # define logic to extract the image masks used to define the extracted regions of interest
            # will return a np.ndarray with 0,1 values. 1: the ROI. Shape as image shape.

        def _load_summary_images(self):

            # define method to extract and return summary images like mean/correlation etc.
            # return np.ndarray , shape: x,y pixels

        def get_accepted_list(self):

            # define method to get all the accepted ROIs after the segmentation operation of pipeline

        def get_rejected_list(self):

            # define method to get all the rejected ROIs after the segmentation operation of pipeline
            # this can also be the compliment of accepted list

        def get_sampling_frequency(self):

            return self._sampling_frequency

        def get_roi_ids(self):

            # define logic to get the numerical id of the ROI output by the segmentation pipeline.

        def get_image_size(self)

            # returns something like self._image_mean.shape

        @staticmethod
        def write_segmentation(segmentation_obj, save_path, other_params):
            '''
            This is an example of a function that is not abstract so it is optional if you want to override it.
            It allows other SegmentationExtractor to use your new SegmentationExtractor to convert their data into your format
            '''