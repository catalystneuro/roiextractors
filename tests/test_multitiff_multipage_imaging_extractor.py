from roiextractors import MultiTiffMultiPageImagingExtractor

from tests.setup_paths import OPHYS_DATA_PATH


def test_init_multitiff_multipage_imaging_extractor():
    extractor = MultiTiffMultiPageImagingExtractor(
        folder_path=OPHYS_DATA_PATH / "imaging_datasets" / "Tif" / "splits",
        pattern="split_{split:d}.tif",
        sampling_frequency=1.0,
    )

    assert extractor.get_num_channels() == 1
    assert extractor.get_num_frames() == 2000
    assert extractor.get_sampling_frequency() == 1.0
    assert extractor.get_channel_names() == ["channel_0"]
    assert extractor.get_dtype() == "uint16"
    assert extractor.get_image_size() == (60, 80)
    assert extractor.get_video().shape == (2000, 60, 80)
    assert list(extractor.tif_paths.keys()) == [
        str(OPHYS_DATA_PATH / "imaging_datasets" / "Tif" / "splits" / x)
        for x in (
            "split_1.tif",
            "split_2.tif",
            "split_3.tif",
            "split_4.tif",
            "split_5.tif",
            "split_6.tif",
            "split_7.tif",
            "split_8.tif",
            "split_9.tif",
            "split_10.tif",
        )
    ]
