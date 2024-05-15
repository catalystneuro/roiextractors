from roiextractors import MultiTiffMultiPageImagingExtractor


def test_init_multitiff_multipage_imaging_extractor():
    extractor = MultiTiffMultiPageImagingExtractor(
        folder_path="tests/test_files", pattern="split_{:d}.tif", sampling_frequency=1.0
    )

    assert extractor.get_num_channels() == 1
    assert extractor.get_num_frames() == 2000
    assert extractor.get_sampling_frequency() == 1.0
    assert extractor.get_channel_names() == ["channel_0"]
    assert extractor.get_dtype() == "uint16"
    assert extractor.get_image_size() == (512, 512)
    assert extractor.get_video().shape == (2000, 512, 512)
