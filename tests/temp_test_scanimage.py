# scratch file to test scanimage tiff extractor
from roiextractors import ScanImageTiffImagingExtractor
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

    extractor = ScanImageTiffImagingExtractor(file_path=example_test, sampling_frequency=30.0)
    print("Example test file loads!")

    metadata = extract_extra_metadata(example_holo)
    metadata_parsed = parse_metadata(metadata)
    extractor = ScanImageTiffMultiPlaneImagingExtractor(file_path=example_holo, **metadata_parsed)
    print("Example holographic file loads!")

    metadata = extract_extra_metadata(example_single)
    metadata_parsed = parse_metadata(metadata)
    metadata_parsed["frames_per_slice"] = 1  # Multiple frames per slice is not supported yet
    extractor = ScanImageTiffImagingExtractor(file_path=example_single, **metadata_parsed)
    print("Example single-plane file loads!")

    metadata = extract_extra_metadata(example_volume)
    metadata_parsed = parse_metadata(metadata)
    metadata_parsed["frames_per_slice"] = 1  # Multiple frames per slice is not supported yet
    extractor = ScanImageTiffImagingExtractor(file_path=example_volume, **metadata_parsed)
    print("Example volume file loads!")

    metadata = extract_extra_metadata(example_multivolume)
    metadata_parsed = parse_metadata(metadata)
    metadata_parsed["frames_per_slice"] = 1  # Multiple frames per slice is not supported yet
    extractor = ScanImageTiffImagingExtractor(file_path=example_multivolume, **metadata_parsed)
    print("Example multivolume file loads!")


if __name__ == "__main__":
    main()
