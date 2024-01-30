# Upcoming

### Improvements
* Improved xml parsing with Bruker [PR #267](https://github.com/catalystneuro/roiextractors/pull/267)


# v0.5.5

### Features

* Updated `Suite2pSegmentationExtractor` to support multi channel and multi plane data. [PR #242](https://github.com/catalystneuro/roiextractors/pull/242)

### Fixes

* Fixed `MicroManagerTiffImagingExtractor` private extractor's dtype to not override the parent's dtype. [PR #257](https://github.com/catalystneuro/roiextractors/pull/257)
* Fixed override of `channel_name` in `Suite2pSegmentationExtractor`. [PR #263](https://github.com/catalystneuro/roiextractors/pull/263)


# v0.5.4

### Features

* Added volumetric and multi-channel support for Bruker format. [PR #230](https://github.com/catalystneuro/roiextractors/pull/230)



# v0.5.3

### Features

* Added support for Miniscope AVI files with the `MiniscopeImagingExtractor`. [PR #225](https://github.com/catalystneuro/roiextractors/pull/225)



# v0.5.2

### Features

* Added support for MicroManager TIFF files with the `MicroManagerTiffImagingExtractor`. [PR #222](https://github.com/catalystneuro/roiextractors/pull/222)

* Added support for Bruker TIFF files with the `BrukerTiffImagingExtractor`. [PR #220](https://github.com/catalystneuro/roiextractors/pull/220)



# v0.5.1

### Features

* Added a `has_time_vector` function for ImagingExtractors and SegmentationExtractors, similar to the SpikeInterface API for detecting if timestamps have been set. [PR #216](https://github.com/catalystneuro/roiextractors/pull/216)

### Fixes

* Fixed two issues with the `SubFrameSegementation` class: (i) attempting to set the private attribute `_image_masks` even when this was not present in the parent, and (ii) not calling the parent function for `get_pixel_masks` and instead using the base method even in cases where this had been overridden by the parent. [PR #215](https://github.com/catalystneuro/roiextractors/pull/215)



# v0.5.0

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
* Added `dtype` consistency check across `MultiImaging` components as well as a direct override method. [PR #195](https://github.com/catalystneuro/neuroconv/pull/195)
* Added the `FrameSliceSegmentationExtractor` class and corresponding `Segmentation.frame_slice(...)` method. [PR #201](https://github.com/catalystneuro/neuroconv/pull/201)
* Changed the `output_struct_name` argument to optional in `ExtractSegmentationExtractor`.
  to allow more flexible usage for the user and better error message when it cannot be found in the file.
  For consistency, `output_struct_name` argument has been also added to the legacy extractor.
  The orientation of segmentation images are transposed for consistency in image orientation (height x width). [PR #210](https://github.com/catalystneuro/roiextractors/pull/210)
* Relaxed rounding of `ImagingExtractor.frame_to_time(...)` and `SegmentationExtractor.frame_to_time(...)` to be more consistent with SpikeInterface. [PR #212](https://github.com/catalystneuro/roiextractors/pull/212)

### Fixes
* Fixed the reference to the proper `mov_field` in `Hdf5ImagingExtractor`. [PR #195](https://github.com/catalystneuro/neuroconv/pull/195)
* Updated the name of the ROICentroids column for the `NwbSegmentationExtractor` to be up-to-date with NeuroConv v0.2.0 `write_segmentation`. [PR #208](https://github.com/catalystneuro/roiextractors/pull/208)
* Updated the trace orientation for the `NwbSegmentationExtractor`. [PR #208](https://github.com/catalystneuro/roiextractors/pull/208)



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
