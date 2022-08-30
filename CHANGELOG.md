# Upcoming

### Back-compatability break
* The orientation of traces in all `SegmentationExtractor`s has been standardized to have time (frames) as the first axis, and ROIs as the final axis. [PR #200](https://github.com/catalystneuro/roiextractors/pull/200)

### Features
* Add support for newer versions of EXTRACT output files. [PR #170](https://github.com/catalystneuro/roiextractors/pull/170)
The `ExtractSegmentationExtractor` class is now abstract and redirects to the newer or older
extractor depending on the version of the file. [PR #170](https://github.com/catalystneuro/roiextractors/pull/170)
* The `ExtractSegmentationExtractor.write_segmentation` method has now been deprecated. [PR #170](https://github.com/catalystneuro/roiextractors/pull/170)

### Improvements
* Add `frame_to_time` to `SegmentationExtractor`, `get_roi_ids` is now a class method. [PR #187](https://github.com/catalystneuro/roiextractors/pull/187)
* Add `set_times` to `SegmentationExtractor`. [PR #188](https://github.com/catalystneuro/roiextractors/pull/188)
* Updated the test for segmentation images to check all images for the given segmentation extractors. [PR #190](https://github.com/catalystneuro/roiextractors/pull/190)
* Refactored the `NwbSegmentationExtractor` to be more flexible with segmentation images and keep up
  with the change in [catalystneuro/neuoroconv#41](https://github.com/catalystneuro/neuroconv/pull/41)
  of trace names. [PR #191](https://github.com/catalystneuro/roiextractors/pull/191)
* Implemented a more efficient case of the base `ImagingExtractor.get_frames` through `get_video` when the indices are contiguous. [PR #195](https://github.com/catalystneuro/neuroconv/pull/195)
* Removed `max_frame` check on `MultiImagingExtractor.get_video()` to adhere to upper-bound slicing semantics. [PR #195](https://github.com/catalystneuro/neuroconv/pull/195)
* Improved the `MultiImagingExtractor.get_video()` to no longer rely on `get_frames`. [PR #195](https://github.com/catalystneuro/neuroconv/pull/195)
* Adding `dtype` consistency check across `MultiImaging` components as well as a direct override method. [PR #195](https://github.com/catalystneuro/neuroconv/pull/195)

### Fixes
* Fixed the reference to the proper `mov_field` in `Hdf5ImagingExtractor`. [PR #195](https://github.com/catalystneuro/neuroconv/pull/195)

### Testing



# v0.4.18

### Improvements
* `get_video` is now an abstract method in `ImagingExtractor` [PR #180](https://github.com/catalystneuro/roiextractors/pull/180)

### Features
* Add dummy segmentation extractor [PR #176](https://github.com/catalystneuro/roiextractors/pull/176)

### Testing
* Added unittests to the `get_frames` method from `ImagingExtractors` to assert that they are consistent with numpy
indexing behavior. [PR #154](https://github.com/catalystneuro/roiextractors/pull/154)
* Tests for spikeinterface like-behavior for the `get_video` funtiction [PR #181](https://github.com/catalystneuro/roiextractors/pull/181)



# v0.4.17

### Depreceations
- Suite2P argument has become `folder_path` instead of `file_path`, `file_path` deprecation scheduled for august or later.

### Documentation
- Improved docstrings across many extractors.

### Features
- Adds MultiImagingExtractor for combining multiple imaging extractors.
- Adds ScanImageTiffExtractor for reading .tiff files output from ScanImage
- Adds NumpyImagingExtractor for extracting raw video data as memmaps.
- Added frame slicing capabilities for imaging extractors.

### Testing
- Added checks and debugs that all sampling frequencies returns are floats
- Round trip testing working for all extractors that have a working write method.
