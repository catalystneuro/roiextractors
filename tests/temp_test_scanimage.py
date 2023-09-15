# scratch file to test scanimage tiff extractor
from roiextractors import ScanImageTiffImagingExtractor


def main():
    example_holo = "/Volumes/T7/CatalystNeuro/NWB/MouseV1/raw-tiffs/2ret/20230119_w57_1_2ret_00001.tif"
    example_single = "/Users/pauladkisson/Documents/CatalystNeuro/ROIExtractors/ophys_testing_data/imaging_datasets/ScanImage/scanimage_20220801_single.tif"
    example_volume = "/Users/pauladkisson/Documents/CatalystNeuro/ROIExtractors/ophys_testing_data/imaging_datasets/ScanImage/scanimage_20220801_volume.tif"
    example_multivolume = "/Users/pauladkisson/Documents/CatalystNeuro/ROIExtractors/ophys_testing_data/imaging_datasets/ScanImage/scanimage_20220801_multivolume.tif"

    extractor = ScanImageTiffImagingExtractor(file_path=example_holo)
    print("Example holographic file loads!")
    extractor = ScanImageTiffImagingExtractor(file_path=example_single)
    print("Example single-plane file loads!")
    extractor = ScanImageTiffImagingExtractor(file_path=example_volume)
    print("Example volume file loads!")
    extractor = ScanImageTiffImagingExtractor(file_path=example_multivolume)
    print("Example multivolume file loads!")


if __name__ == "__main__":
    main()
