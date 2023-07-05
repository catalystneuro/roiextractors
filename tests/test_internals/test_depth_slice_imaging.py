from hdmf.testing import TestCase
from roiextractors.testing import (
    generate_dummy_volumetric_imaging_extractor,
    generate_dummy_imaging_extractor,
)


class TestDepthSliceImaging(TestCase):
    @classmethod
    def setUpClass(cls):
        """Use a toy example of ten frames of a 5 x 4 grayscale image with 8 planes."""
        cls.num_frames = 10
        cls.num_rows = 5
        cls.num_columns = 4
        cls.num_planes = 8
        cls.sampling_frequency = 30.0

        cls.toy_volumetric_imaging_example = generate_dummy_volumetric_imaging_extractor(
            num_frames=cls.num_frames,
            num_rows=cls.num_rows,
            num_columns=cls.num_columns,
            num_z_planes=cls.num_planes,
            sampling_frequency=cls.sampling_frequency,
        )

        start_plane = 2
        end_plane = 7
        cls.depth_sliced_planes = end_plane - start_plane

        cls.depth_sliced_imaging = cls.toy_volumetric_imaging_example.depth_slice(start_plane=start_plane, end_plane=end_plane)

    def test_frame_slice_image_size(self):
        depth_sliced_image_size = (self.num_rows, self.num_columns, self.depth_sliced_planes)
        self.assertEqual(self.depth_sliced_imaging.get_image_size(), depth_sliced_image_size)

    def test_frame_slice_num_frames(self):
        self.assertEqual(self.depth_sliced_imaging.get_num_frames(), self.num_frames)

    def test_get_sampling_frequency(self):
        self.assertEqual(self.depth_sliced_imaging.get_sampling_frequency(), self.sampling_frequency)

    def test_get_channel_names(self):
        parent_imaging_channel_names = self.toy_volumetric_imaging_example.get_channel_names()
        self.assertEqual(self.depth_sliced_imaging.get_channel_names(), parent_imaging_channel_names)

    def test_get_num_channels(self):
        parent_imaging_num_channels = self.toy_volumetric_imaging_example.get_num_channels()
        self.assertEqual(self.depth_sliced_imaging.get_num_channels(), parent_imaging_num_channels)

    def test_depth_slice_with_non_volumetric_imaging(self):
        non_volumetric_imaging = generate_dummy_imaging_extractor(
            num_frames=self.num_frames,
            num_rows=self.num_rows,
            num_columns=self.num_columns,
        )

        with self.assertRaisesWith(
            exc_type=AssertionError, exc_msg="DepthSliceImagingExtractor can be only used for volumetric imaging data."
        ):
            non_volumetric_imaging.depth_slice()

    def test_depth_slice_on_depth_slice(self):
        depth_sliced_imaging = self.depth_sliced_imaging.depth_slice(start_plane=2, end_plane=5)

        image_size = (self.num_rows, self.num_columns, 3)
        self.assertEqual(depth_sliced_imaging.get_image_size(), image_size)

    def test_frame_slice_with_depth_slice(self):
        depth_sliced_imaging = self.depth_sliced_imaging.depth_slice(start_plane=2, end_plane=5)
        frame_sliced_imaging = depth_sliced_imaging.frame_slice(start_frame=3, end_frame=7)

        self.assertEqual(frame_sliced_imaging.get_num_frames(), 4)
        self.assertEqual(frame_sliced_imaging.get_image_size(), (5, 4, 3))

        depth_sliced_imaging = frame_sliced_imaging.depth_slice(start_plane=1, end_plane=3)
        self.assertEqual(depth_sliced_imaging.get_num_frames(), 4)
        self.assertEqual(depth_sliced_imaging.get_image_size(), (5, 4, 2))

    def test_get_dtype(self):
        assert self.depth_sliced_imaging.get_dtype() == self.toy_volumetric_imaging_example.get_dtype()