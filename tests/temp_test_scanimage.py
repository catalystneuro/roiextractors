# scratch file to test scanimage tiff extractor
from roiextractors import ScanImageTiffImagingExtractor, MultiImagingExtractor
from roiextractors.extractors.tiffimagingextractors.scanimagetiffimagingextractor import (
    extract_extra_metadata,
    parse_metadata,
    ScanImageTiffMultiPlaneImagingExtractor,
)


def main():
    example_test = "/Users/pauladkisson/Documents/CatalystNeuro/ROIExtractors/ophys_testing_data/imaging_datasets/ScanImage/sample_scanimage_version_3_8.tiff"
    example_holo = "/Volumes/T7/CatalystNeuro/NWB/MouseV1/raw-tiffs/2ret/20230119_w57_1_2ret_00001.tif"
    example_single = "/Users/pauladkisson/Documents/CatalystNeuro/ROIExtractors/ophys_testing_data/imaging_datasets/ScanImage/scanimage_20220801_single.tif"
    example_volume = "/Users/pauladkisson/Documents/CatalystNeuro/ROIExtractors/ophys_testing_data/imaging_datasets/ScanImage/scanimage_20220801_volume.tif"
    example_multivolume = "/Users/pauladkisson/Documents/CatalystNeuro/ROIExtractors/ophys_testing_data/imaging_datasets/ScanImage/scanimage_20220801_multivolume.tif"
    multi_example_holo = [
        f"/Volumes/T7/CatalystNeuro/NWB/MouseV1/raw-tiffs/2ret/20230119_w57_1_2ret_{i:05d}.tif" for i in range(1, 11)
    ]

    extractor = ScanImageTiffImagingExtractor(file_path=example_test, sampling_frequency=30.0)
    print("Example test file loads!")

    extractor = ScanImageTiffMultiPlaneImagingExtractor(file_path=example_holo)
    print("Example holographic file loads!")
    video = extractor.get_video()
    print("Video shape:", video.shape)

    extractors = [ScanImageTiffMultiPlaneImagingExtractor(file_path=f) for f in multi_example_holo]
    extractor = MultiImagingExtractor(imaging_extractors=extractors)
    print("Example multi-holographic file loads!")
    video = extractor.get_video()
    print("Video shape:", video.shape)

    extractor = ScanImageTiffMultiPlaneImagingExtractor(file_path=example_volume)
    print("Example volume file loads!")
    video = extractor.get_video()
    print("Video shape:", video.shape)

    extractor = ScanImageTiffMultiPlaneImagingExtractor(file_path=example_multivolume)
    print("Example multi-volume file loads!")
    video = extractor.get_video()
    print("Video shape:", video.shape)


if __name__ == "__main__":
    main()
