from roiextractors import MultiTiffImagingExtractor, FolderTiffImagingExtractor

from tests.setup_paths import OPHYS_DATA_PATH


def test_init_folder_tiff_imaging_extractor_multi_page():
    extractor = FolderTiffImagingExtractor(
        folder_path=OPHYS_DATA_PATH / "imaging_datasets" / "Tif" / "splits",
        pattern="split_{split:d}.tif",
        sampling_frequency=1.0,
    )

    assert extractor.get_num_channels() == 1
    assert extractor.get_num_frames() == 2000
    assert extractor.get_sampling_frequency() == 1.0
    assert extractor.get_channel_names() is None
    assert extractor.get_dtype() == "uint16"
    assert extractor.get_image_size() == (60, 80)
    assert extractor.get_video().shape == (2000, 60, 80)
    assert list(extractor.file_paths) == [
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


def test_init_multitiff_imaging_extractor_multi_page():
    extractor = MultiTiffImagingExtractor(
        file_paths=[
            OPHYS_DATA_PATH / "imaging_datasets" / "Tif" / "splits" / f"split_{i}.tif"
            for i in range(1, 11)
        ],
        sampling_frequency=1.0,
    )

    assert extractor.get_num_channels() == 1
    assert extractor.get_num_frames() == 2000
    assert extractor.get_sampling_frequency() == 1.0
    assert extractor.get_channel_names() is None
    assert extractor.get_dtype() == "uint16"
    assert extractor.get_image_size() == (60, 80)
    assert extractor.get_video().shape == (2000, 60, 80)
    assert list(extractor.file_paths) == [
        OPHYS_DATA_PATH / "imaging_datasets" / "Tif" / "splits" / x
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
