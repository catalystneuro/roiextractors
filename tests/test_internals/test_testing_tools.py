import numpy as np
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal

import numpy as np
from numpy.testing import assert_array_equal

from roiextractors.testing import (
    generate_dummy_segmentation_extractor,
    _assert_iterable_complete,
)


class TestDummySegmentationExtractor(TestCase):
    def setUp(self) -> None:
        self.num_rois = 10
        self.num_frames = 30
        self.num_rows = 25
        self.num_columns = 25
        self.sampling_frequency = 30.0

        self.raw = True
        self.dff = True
        self.deconvolved = True
        self.neuropil = True

    def test_default_values(self):
        segmentation_extractor = generate_dummy_segmentation_extractor()

        # Test basic shape
        assert segmentation_extractor.get_num_rois() == self.num_rois
        assert segmentation_extractor.get_num_frames() == self.num_frames
        assert segmentation_extractor.get_image_size() == (self.num_rows, self.num_columns)
        assert segmentation_extractor.get_sampling_frequency() == self.sampling_frequency
        assert segmentation_extractor.get_roi_ids() == list(range(self.num_rois))
        assert segmentation_extractor.get_accepted_list() == segmentation_extractor.get_roi_ids()
        assert segmentation_extractor.get_rejected_list() == []
        assert segmentation_extractor.get_roi_locations().shape == (2, self.num_rois)

        # Test frame_to_time
        times = np.round(np.arange(self.num_frames) / self.sampling_frequency, 6)
        assert_array_equal(segmentation_extractor.frame_to_time(frame_indices=np.arange(self.num_frames)), times)
        self.assertEqual(segmentation_extractor.frame_to_time(frame_indices=8), times[8])

        # Test image masks
        assert segmentation_extractor.get_roi_image_masks().shape == (self.num_rows, self.num_columns, self.num_rois)
        # TO-DO Missing testing of pixel masks

        # Test summary images
        assert segmentation_extractor.get_image(name="mean").shape == (self.num_rows, self.num_columns)
        assert segmentation_extractor.get_image(name="correlation").shape == (self.num_rows, self.num_columns)

        # Test signals
        assert segmentation_extractor.get_traces(name="raw").shape == (self.num_rois, self.num_frames)
        assert segmentation_extractor.get_traces(name="dff").shape == (self.num_rois, self.num_frames)
        assert segmentation_extractor.get_traces(name="deconvolved").shape == (self.num_rois, self.num_frames)
        assert segmentation_extractor.get_traces(name="neuropil").shape == (self.num_rois, self.num_frames)

    def test_passing_parameters(self):

        segmentation_extractor = generate_dummy_segmentation_extractor()

        # Test basic shape
        assert segmentation_extractor.get_num_rois() == self.num_rois
        assert segmentation_extractor.get_num_frames() == self.num_frames
        assert segmentation_extractor.get_image_size() == (self.num_rows, self.num_columns)
        assert segmentation_extractor.get_sampling_frequency() == self.sampling_frequency
        assert segmentation_extractor.get_roi_ids() == list(range(self.num_rois))
        assert segmentation_extractor.get_accepted_list() == segmentation_extractor.get_roi_ids()
        assert segmentation_extractor.get_rejected_list() == []
        assert segmentation_extractor.get_roi_locations().shape == (2, self.num_rois)

        # Test image masks
        assert segmentation_extractor.get_roi_image_masks().shape == (self.num_rows, self.num_columns, self.num_rois)
        # TO-DO Missing testing of pixel masks

        # Test summary images
        assert segmentation_extractor.get_image(name="mean").shape == (self.num_rows, self.num_columns)
        assert segmentation_extractor.get_image(name="correlation").shape == (self.num_rows, self.num_columns)

        # Test signals
        assert segmentation_extractor.get_traces(name="raw").shape == (self.num_rois, self.num_frames)
        assert segmentation_extractor.get_traces(name="dff").shape == (self.num_rois, self.num_frames)
        assert segmentation_extractor.get_traces(name="deconvolved").shape == (self.num_rois, self.num_frames)
        assert segmentation_extractor.get_traces(name="neuropil").shape == (self.num_rois, self.num_frames)

    def test_set_times(self):
        """Test that set_times sets the times in the expected way."""

        segmentation_extractor = generate_dummy_segmentation_extractor()

        num_frames = segmentation_extractor.get_num_frames()
        sampling_frequency = segmentation_extractor.get_sampling_frequency()

        # Check that times have not been set yet
        assert segmentation_extractor._times is None

        # Set times with an array that has the same length as the number of frames
        times_to_set = np.round(np.arange(num_frames) / sampling_frequency, 6)
        segmentation_extractor.set_times(times_to_set)

        assert_array_equal(segmentation_extractor._times, times_to_set)

        _assert_iterable_complete(
            iterable=segmentation_extractor._times,
            dtypes=np.ndarray,
            element_dtypes=np.float64,
            shape=(num_frames,),
        )

        # Set times with an array that is too short
        times_to_set = np.round(np.arange(num_frames - 1) / sampling_frequency, 6)
        with self.assertRaisesWith(
            exc_type=AssertionError,
            exc_msg="'times' should have the same length of the number of frames!",
        ):
            segmentation_extractor.set_times(times_to_set)

    def test_frame_to_time_no_sampling_frequency(self):
        segmentation_extractor = generate_dummy_segmentation_extractor(
            sampling_frequency=None,
        )

        times = np.arange(self.num_frames) / self.sampling_frequency
        segmentation_extractor._times = times

        self.assertEqual(segmentation_extractor.frame_to_time(frame_indices=2), times[2])
        assert_array_equal(
            segmentation_extractor.frame_to_time(
                frame_indices=np.arange(self.num_frames),
            ),
            times,
        )
