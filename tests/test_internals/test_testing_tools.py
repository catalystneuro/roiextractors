import numpy as np
from hdmf.testing import TestCase
from numpy.testing import assert_array_equal

from roiextractors.testing import (
    _assert_iterable_complete,
    generate_dummy_segmentation_extractor,
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
        assert segmentation_extractor.get_num_samples() == self.num_frames
        assert segmentation_extractor.get_frame_shape() == (self.num_rows, self.num_columns)
        assert segmentation_extractor.get_sampling_frequency() == self.sampling_frequency
        expected_roi_ids = [f"roi_{i}" for i in range(self.num_rois)]
        assert segmentation_extractor.get_roi_ids() == expected_roi_ids
        assert segmentation_extractor.get_accepted_list() == segmentation_extractor.get_roi_ids()
        assert segmentation_extractor.get_rejected_list() == []
        assert segmentation_extractor.get_roi_locations().shape == (2, self.num_rois)

        # Test frame_to_time
        times = np.arange(self.num_frames) / self.sampling_frequency
        assert_array_equal(segmentation_extractor.frame_to_time(frames=np.arange(self.num_frames)), times)
        self.assertEqual(segmentation_extractor.frame_to_time(frames=8), times[8])

        # Test image masks
        assert segmentation_extractor.get_roi_image_masks().shape == (self.num_rows, self.num_columns, self.num_rois)
        # TO-DO Missing testing of pixel masks

        # Test summary images
        assert segmentation_extractor.get_image(name="mean").shape == (self.num_rows, self.num_columns)
        assert segmentation_extractor.get_image(name="correlation").shape == (self.num_rows, self.num_columns)

        # Test signals
        assert segmentation_extractor.get_traces(name="raw").shape == (self.num_frames, self.num_rois)
        assert segmentation_extractor.get_traces(name="dff").shape == (self.num_frames, self.num_rois)
        assert segmentation_extractor.get_traces(name="deconvolved").shape == (self.num_frames, self.num_rois)
        assert segmentation_extractor.get_traces(name="neuropil").shape == (self.num_frames, self.num_rois)

    def test_passing_parameters(self):
        segmentation_extractor = generate_dummy_segmentation_extractor()

        # Test basic shape
        assert segmentation_extractor.get_num_rois() == self.num_rois
        assert segmentation_extractor.get_num_samples() == self.num_frames
        assert segmentation_extractor.get_frame_shape() == (self.num_rows, self.num_columns)
        assert segmentation_extractor.get_sampling_frequency() == self.sampling_frequency
        expected_roi_ids = [f"roi_{i}" for i in range(self.num_rois)]
        assert segmentation_extractor.get_roi_ids() == expected_roi_ids
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
        assert segmentation_extractor.get_traces(name="raw").shape == (self.num_frames, self.num_rois)
        assert segmentation_extractor.get_traces(name="dff").shape == (self.num_frames, self.num_rois)
        assert segmentation_extractor.get_traces(name="deconvolved").shape == (self.num_frames, self.num_rois)
        assert segmentation_extractor.get_traces(name="neuropil").shape == (self.num_frames, self.num_rois)

    def test_set_times(self):
        """Test that set_times sets the times in the expected way."""

        segmentation_extractor = generate_dummy_segmentation_extractor()

        num_frames = segmentation_extractor.get_num_samples()
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
            exc_msg="'times' should have the same length of the number of samples!",
        ):
            segmentation_extractor.set_times(times_to_set)

    def test_frame_to_time_no_sampling_frequency(self):
        segmentation_extractor = generate_dummy_segmentation_extractor(
            sampling_frequency=None,
        )

        times = np.arange(self.num_frames) / self.sampling_frequency
        segmentation_extractor._times = times

        self.assertEqual(segmentation_extractor.frame_to_time(frames=2), times[2])
        assert_array_equal(
            segmentation_extractor.frame_to_time(frames=np.arange(self.num_frames)),
            times,
        )
