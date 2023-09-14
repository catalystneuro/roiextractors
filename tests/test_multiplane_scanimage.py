from roiextractors.extractors.tiffimagingextractors.scanimagetiffimagingextractor import (
    ScanImageTiffImagingExtractor,
)
import matplotlib.pyplot as plt


def main():
    file_path = "/Volumes/T7/CatalystNeuro/NWB/MouseV1/raw-tiffs/2ret/20230119_w57_1_2ret_00001.tif"
    extractor = ScanImageTiffImagingExtractor(file_path=file_path)
    print(f"num_frames: {extractor.get_num_frames()}")
    print(f"num_planes: {extractor.get_num_planes()}")
    print(f"num_channels: {extractor.get_num_channels()}")
    print(f"sampling_frequency: {extractor.get_sampling_frequency()}")
    print(f"channel_names: {extractor.get_channel_names()}")
    print(f"image_size: {extractor.get_image_size()}")
    first_frame = extractor._get_single_frame(frame=0, channel=1, plane=2)
    several_frames = extractor.get_frames(frame_idxs=[0, 10, 2], channel=1, plane=2)
    video = extractor.get_video(channel=0, plane=2)
    print(several_frames.shape)
    plt.imshow(first_frame[0])
    plt.show()
    plt.imshow(several_frames[0])
    plt.show()
    plt.imshow(video[0])
    plt.show()


if __name__ == "__main__":
    main()
