# v0.6.2 (Upcoming)

### Features

### Fixes

### Deprecations And Removals

### Improvements
* Refactor Femtonics Imaging Extractor session, munit and channel selection logic. Added missing stub files (`single_channel.mesc` and `single_m_unit_index`)[PR #479](https://github.com/catalystneuro/roiextractors/pull/479)
* Added tests for unused stub files (`dual_color_movie_with_dropped_frames.isxd` and `multiplane_movie.isxd`) in Inscopix imaging extractor [PR #478](https://github.com/catalystneuro/roiextractors/pull/478)

# v0.6.1 (July 7th 2025)

### Features
* Added native timestamp support with automatic fallback hierarchy: `get_native_timestamps()` abstract method and `get_timestamps()` concrete method to all imaging and segmentation extractors. The `get_timestamps()` method follows priority order: cached times → native timestamps → inferred from sampling frequency. This enables automatic native timestamp usage when available (e.g., ScanImage TIFF files) while maintaining backward compatibility. [PR #465](https://github.com/catalystneuro/roiextractors/pull/465)
* Added property management methods to `SegmentationExtractor`: `set_property()`, `get_property()`, and `get_property_keys()` for storing and retrieving custom ROI properties [PR #467](https://github.com/catalystneuro/roiextractors/pull/467)
* Added `MinianSegmentationExtractor` for reading and extracting metadata and segmentation data from Minian output files.[PR #368](https://github.com/catalystneuro/roiextractors/pull/368)

### Fixes
* Fixed a bug in CaimanSegmentationExtractor where empty components of the segmentation traces will throw an error
[PR #452](https://github.com/catalystneuro/roiextractors/pull/452)

### Deprecations And Removals
* Remove deprecated arguments `combined` and `plane_no` from `Suite2pSegmentationExtractor` [PR #457](https://github.com/catalystneuro/roiextractors/pull/457)
* `check_imaging_equal` and `check_segmentation_equal` no longer compare the number of channels in an extractor as that attribute has been deprecated [PR #470](https://github.com/catalystneuro/roiextractors/pull/470)
* `exclude_channel_comparison` in `check_imaging_equal` is deprecated and will be removed in or after January 2026. This parameter is no longer needed as extractors do not have multiple channels. [PR #470](https://github.com/catalystneuro/roiextractors/pull/470)
* The `num_frames` parameter in `generate_dummy_imaging_extractor` and `generate_dummy_segmentation_extractor` is now deprecated and will be removed on or after January 2026. Use the new `num_samples` parameter instead. `num_samples` is now keyword-only and will become positional-only in a future release. [PR #470](https://github.com/catalystneuro/roiextractors/pull/470)
* The `sample_indices_to_time()` method in both ImagingExtractor and SegmentationExtractor is deprecated and will be removed on or after January 2026. Use `get_timestamps()` instead. [PR #448](https://github.com/catalystneuro/roiextractors/pull/448)
Remove deprecated arguments `combined` and `plane_no` from `Suite2pSegmentationExtractor` [PR #457](https://github.com/catalystneuro/roiextractors/pull/457)

### Improvements
* Updated GitHub Actions workflows to use cross-OS cache sharing for multi-OS testing data, reducing redundant downloads and improving CI efficiency. Added reusable data loading action pattern similar to neuroconv. [PR #459](https://github.com/catalystneuro/roiextractors/pull/459)
* Improved writing to NWB documentation to point out users to neuroconv [PR #468](https://github.com/catalystneuro/roiextractors/pull/468)
* Add comprehensive test suite for CaimanSegmentationExtractor covering basic properties, ROI masks, traces, accepted/rejected lists, and different CaImAn dataset formats [PR #464](https://github.com/catalystneuro/roiextractors/pull/464)
* Add isort and remove unused imports to pre-commit [PR #471](https://github.com/catalystneuro/roiextractors/pull/471)
* Updated tests for `CaimanSegmentationExtractor` to cover all stub files, and enhanced the extractor to return quality metrics (`r_values`, `SNR_comp`, `cnn_preds`). [PR #466](https://github.com/catalystneuro/roiextractors/pull/466)

# v0.6.0 (June 17th, 2025)

### Features
* Added `FemtonicsImagingExtractor` for reading and extracting metadata and imaging data from Femtonics MESc files.[PR #440](https://github.com/catalystneuro/roiextractors/pull/440)
* Added `get_original_frame_indices()` method to `ScanImageImagingExtractor` for mapping extractor samples back to original frame indices in raw microscopy data. This method can be used to synchronize with other data sources or for advanced analysis that requires knowledge of the original frame indices. [PR #445](https://github.com/catalystneuro/roiextractors/pull/445)

### Fixes
* Fixed InscopixSegmentationExtractor with support for string/integer ROI IDs, NWB-compliant pixel masks, standardized ROI lists, improved macOS compatibility, enhanced metadata extraction, and general code cleanup.
[PR #435](https://github.com/catalystneuro/roiextractors/pull/435)
* Added missing function to retrieve inscopix metadata [PR #436](https://github.com/catalystneuro/roiextractors/pull/436)
* Fixed deprecated scipy import warning by updating `mat_struct` import path [PR #442](https://github.com/catalystneuro/roiextractors/pull/442)
* Fixed deprecation warning about invalid escape sequence in micromanager TIFF extractor [PR #442](https://github.com/catalystneuro/roiextractors/pull/442)
* Removed integer-only assumption for ROI IDs in segmentation extractors. ROI IDs can now be strings or any type. Updated `generate_dummy_segmentation_extractor` to produce string ROI IDs in format `roi_00`, `roi_01`, etc. [PR #449](https://github.com/catalystneuro/roiextractors/pull/449)


### Deprecations And Removals
* The `get_image_size()` method in SegmentationExtractor is deprecated and will be removed on or after January 2026. Use `get_frame_shape()` instead. [PR #443](https://github.com/catalystneuro/roiextractors/pull/443)
* The `get_num_frames()` method in SegmentationExtractor is deprecated and will be removed on or after January 2026. Use `get_num_samples()` instead. [PR #443](https://github.com/catalystneuro/roiextractors/pull/443)
* The `frame_to_time()` method in SegmentationExtractor is deprecated and will be removed on or after January 2026. Use `sample_indices_to_time()` instead. [PR #447](https://github.com/catalystneuro/roiextractors/pull/447)
* Removed unused `is_writable` class attributes from all extractor classes [PR #442](https://github.com/catalystneuro/roiextractors/pull/442)
* `get_frames` is now `get_samples`. `get_frames` will be deprecated in or after January 2026 [PR #444](https://github.com/catalystneuro/roiextractors/pull/444)
* The `frame_slice()` method in SegmentationExtractor is deprecated and will be removed on or after January 2026. Use ``slice_samples`()` instead. [PR #451](https://github.com/catalystneuro/roiextractors/pull/451)
* The `FrameSliceSegmentationExtractor` class is deprecated and will be removed on or after January 2026. Use `SampleSlicedSegmentationExtractor` instead. [PR #451](https://github.com/catalystneuro/roiextractors/pull/451)

### Improvements
* Bruker series can now read sequences of type `BrightnessOverTime` [PR #448](https://github.com/catalystneuro/roiextractors/pull/448)

# v0.5.13 (May 12th, 2025)

### Features
* Added `ScanImageImagingExtractor` for simplifying reading ScanImage data [PR #412](https://github.com/catalystneuro/roiextractors/pull/412)
* Added volumetric imaging support with `is_volumetric` flag, `get_frame_shape`, `get_num_planes`, and `get_volume_shape` methods [PR #418](https://github.com/catalystneuro/roiextractors/pull/418)
* Added support for multiple samples per slice to `ScanImageIMagingExtractor` [PR # 417](https://github.com/catalystneuro/roiextractors/pull/417)
* Added support for flyback frames to `ScanImageImagingExtractor` [PR #419](https://github.com/catalystneuro/roiextractors/pull/419)
* Added InscopixSegmentationExtractor for reading .isxd segmentation files [#407](https://github.com/catalystneuro/roiextractors/pull/407)
* Add `plane_index` to `ScanImageImagingExtractor` to obtain a planar extractor across a plane [PR #424](https://github.com/catalystneuro/roiextractors/pull/424)
* Add testing to timestamp extraction on `ScanImageImagingExtractor` [PR #426](https://github.com/catalystneuro/roiextractors/pull/426)
* Add paths as string support to `ScanImageImagingExtractor` [PR #427](https://github.com/catalystneuro/roiextractors/pull/427)
* Add informative error for old ScanImage files with `ScanImageImagingExtractor` [PR #427](https://github.com/catalystneuro/roiextractors/pull/427)
* Add the option to read interleaved data in `ScanImageImagingExtractor` [PR #428](https://github.com/catalystneuro/roiextractors/pull/428)
* Add detection of missing files in a sequence and excess file for `ScanImageImagingExtractor` file find heuristic [PR #429](https://github.com/catalystneuro/roiextractors/pull/429)

### Fixes
* Fixed `get_series` method in `MemmapImagingExtractor` to preserve channel dimension [PR #416](https://github.com/catalystneuro/roiextractors/pull/416)
* Fix memory estimation for volumetric imaging extractors in their `_repr_` [PR #422](https://github.com/catalystneuro/roiextractors/pull/433)


### Deprecations And Removals
* The `get_video(start_frame, end_frame)` method is deprecated and will be removed in or after September 2025. Use `get_series(start_sample, end_sample)` instead for consistent naming with `get_num_samples`. [PR #416](https://github.com/catalystneuro/roiextractors/pull/416)
* Python 3.9 is no longer supported [PR #423](https://github.com/catalystneuro/roiextractors/pull/423)
* The `time_to_frame()` method is deprecated and will be removed in or after October 2025. Use `time_to_sample_indices()` instead for consistent terminology between planar and volumetric data. [PR #430](https://github.com/catalystneuro/roiextractors/pull/430)
* The `frame_to_time()` method is deprecated and will be removed in or after October 2025. Use `sample_indices_to_time()` instead for consistent terminology between planar and volumetric data. [PR #430](https://github.com/catalystneuro/roiextractors/pull/430)
* The `frame_slice()` method is deprecated and will be removed in or after October 2025. Use `slice_samples()` instead for consistent terminology between planar and volumetric data. [PR #430](https://github.com/catalystneuro/roiextractors/pull/430)
* The `FrameSliceImagingExtractor` class is deprecated and will be removed in or after October 2025. Use `SampleSlicedImagingExtractor` instead for consistent terminology between planar and volumetric data.
[PR #430](https://github.com/catalystneuro/roiextractors/pull/430)
* Long deprecated `ScanImageTiffImagingExtractor` is removed. For ScanImage legacy data use `ScanImageLegacyImagingExtractor` [PR #431](https://github.com/catalystneuro/roiextractors/pull/431)
* Deprecated `ScanImageTiffMultiPlaneMultiFileImagingExtractor`, `ScanImageTiffSinglePlaneMultiFileImagingExtractor`, `ScanImageTiffMultiPlaneImagingExtractor`, and `ScanImageTiffSinglePlaneImagingExtractor` classes. These will be removed on or after October 2025. Use `ScanImageImagingExtractor` instead. [PR #432](https://github.com/catalystneuro/roiextractors/pull/432)

### Improvements
* Improved criteria for determining if a ScanImage dataset is volumetric by checking both `SI.hStackManager.enable` and `SI.hStackManager.numSlices > 1`[PR #425](https://github.com/catalystneuro/roiextractors/pull/425)


# v0.5.12 (April 18th, 2025)

### Features
* New `read_scanimage_metadata` for reading scanimage metadata from a file directly as a python dict [PR #405](https://github.com/catalystneuro/roiextractors/pull/401)

### Fixes
* Use `SI.hChannels.channelSave` or `SI.hChannels.channelsave` to determine number of channels for ScanImage extractors when available [PR #401](https://github.com/catalystneuro/roiextractors/pull/401)
* Fixes the sampling rate for volumetric `ScanImage` [#405](https://github.com/catalystneuro/roiextractors/pull/401)

### Deprecations
* Deprecated `write_imaging` and `write_segmentation` methods: [#403](https://github.com/catalystneuro/roiextractors/pull/403)
* The `get_image_size()` method is deprecated and will be removed in or after September 2025. Use `get_image_shape()` instead for consistent behavior across all extractors. [#409](https://github.com/catalystneuro/roiextractors/pull/409)
* Change `get_num_frames` for `get_num_samples` [#411](https://github.com/catalystneuro/roiextractors/pull/411)

### Improvements
* Removed unused installed attribute [#410](https://github.com/catalystneuro/roiextractors/pull/410)

# v0.5.11 (March 5th, 2025)

### Features
* Added ThorTiffImagingExtractor for reading TIFF files produced via Thor [#395](https://github.com/catalystneuro/roiextractors/pull/395)

### Fixes
* Use tifffile.imwrite instead of tifffile.imsave for TiffImagingExtractor: [#390](https://github.com/catalystneuro/roiextractors/pull/390)

### Deprecations
* The 'channel' parameter in get_frames() and get_video() methods is deprecated and will be removed in August 2025.  [#388](https://github.com/catalystneuro/roiextractors/pull/388)
* Removed get_num_channels from the base ImagingExtractor as an abstract class. Implementations remain in concrete classes until deprecation on August 2025 [#392](https://github.com/catalystneuro/roiextractors/pull/392)

### Improvements
* Use `pyproject.toml` for project metadata and installation requirements [#382](https://github.com/catalystneuro/roiextractors/pull/382)
* Added `__repr__` and  methods to ImagingExtractor for better display in terminals and Jupyter notebooks [#393](https://github.com/catalystneuro/roiextractors/pull/393) and [#396](https://github.com/catalystneuro/roiextractors/pull/396)
* Removed deprecated np.product from the library [#397](https://github.com/catalystneuro/roiextractors/pull/397)


# v0.5.10 (November 6th, 2024)

### Features
* Added a seed to dummy generators [#361](https://github.com/catalystneuro/roiextractors/pull/361)
* Added depth_slice for VolumetricImagingExtractors [PR #363](https://github.com/catalystneuro/roiextractors/pull/363)

### Fixes
* Added specific error message for single-frame scanimage data [PR #360](https://github.com/catalystneuro/roiextractors/pull/360)
* Fixed bug with ScanImage's parse_metadata so that it works properly when hStackManager is disabled [PR #373](https://github.com/catalystneuro/roiextractors/pull/373)
* Add support for background components in FrameSliceSegmentationExtractor [PR #378](https://github.com/catalystneuro/roiextractors/pull/378)

### Improvements
* Removed unnecessary import checks for scipy, h5py, and zarr [PR #364](https://github.com/catalystneuro/roiextractors/pull/364)
* Improved the error message for the `set_timestamps` method in the `ImagingExtractor` class[PR #377](https://github.com/catalystneuro/roiextractors/pull/377)
* Renamed `MiniscopeImagingExtractor` to`MiniscopeMultiRecordingImagingExtractor` class[PR #374](https://github.com/catalystneuro/roiextractors/pull/374)


# v0.5.9

### Deprecations

* Remove support for Python 3.8: [PR #325](https://github.com/catalystneuro/roiextractors/pull/325)

### Features

* Add InscopixImagingExtractor: [#276](https://github.com/catalystneuro/roiextractors/pull/276)
* Updated testing workflows to include python 3.12, m1/intel macos, and dev tests to check neuroconv: [PR #317](https://github.com/catalystneuro/roiextractors/pull/317)
* Added API documentation: [#337](https://github.com/catalystneuro/roiextractors/pull/337)
* Optimized `get_streams` for `BrukerTiffSinglePlaneImagingExtractor` by introducing a static function  `get_available_channels` which uses lazy parsing of the XML to fetch the available channels: [#344](https://github.com/catalystneuro/roiextractors/pull/344)

### Fixes

* Remove unnecessary `scipy` import error handling: [#315](https://github.com/catalystneuro/roiextractors/pull/315)
* Fixed the typing returned by the `InscopixImagingExtractor.get_dtype` method: [#326](https://github.com/catalystneuro/roiextractors/pull/326)
* Detect Changelog Updates was moved to its own dedicated workflow to avoid daily testing failures: [#336](https://github.com/catalystneuro/roiextractors/pull/336)
* Fixed the Daily testing workflows by passing along the appropriate secrets: [#340](https://github.com/catalystneuro/roiextractors/pull/340)
* Change the criteria of determining if Bruker data is volumetric [#342](https://github.com/catalystneuro/roiextractors/pull/342)
* Fixes a bug that assumes the channel name is is on the tiff file for `BrukerTiffSinglePlaneImagingExtractor` [#343](https://github.com/catalystneuro/roiextractors/pull/343)
* Including `packaging` explicitly in minimal requirements [#347](https://github.com/catalystneuro/roiextractors/pull/347)
* Updated requirements to include cv2 and update dev testing locations for neuroconv: [#357](https://github.com/catalystneuro/roiextractors/pull/357)

### Improvements

* The `Suite2PSegmentationExtractor` now produces an error when a required sub-file is missing: [#330](https://github.com/catalystneuro/roiextractors/pull/330)
* Added `_image_mask` initialization in `BaseSegmentationExtractor`; combined `abstractmethod`s into top of file: [#327](https://github.com/catalystneuro/roiextractors/pull/327)
* Optimize parsing of xml with `lxml` library for Burker extractors: [#346](https://github.com/catalystneuro/roiextractors/pull/346)
* Protect sima and dill export [#351](https://github.com/catalystneuro/roiextractors/pull/351)
* Improve error message when `TiffImagingExtractor` is not able to form memmap [#353](https://github.com/catalystneuro/roiextractors/pull/353)
* Updated Check Docstrings workflow to use new github action: [#354](https://github.com/catalystneuro/roiextractors/pull/354)

### Testing

* Updated testing workflows to include python 3.12, m1/intel macos, and dev tests to check neuroconv: [PR #317](https://github.com/catalystneuro/roiextractors/pull/317)
* Added daily testing workflow and fixed bug with python 3.12 by upgrading scanimage-tiff-reader version: [PR #321](https://github.com/catalystneuro/roiextractors/pull/321)
* Remove wheel from requirements and move CI dependencies to test requirements [PR #348](https://github.com/catalystneuro/roiextractors/pull/348)
* Use Spikeinterface instead of Spikeextractors for toy_example [PR #349](https://github.com/catalystneuro/roiextractors/pull/349)


# v0.5.8

### Fixes

* The triggering workflow name for update version was incorrectly set to `auto-publish` (the name of the yaml file).  It has been renamed to `Upload Package to PyPI` (the name field of the workflow): [PR #304](https://github.com/catalystneuro/roiextractors/pull/304).

* Fixed bug with automatic changelog test that was causing it to fail for daily tests: [PR #310](https://github.com/catalystneuro/roiextractors/pull/310)

* Updated zenodo to get a DOI on each release: No PR



# v0.5.7

### Features

* Add support to get background components: add `get_background_ids()`, `get_background_image_masks()`, `get_background_pixel_masks()` to `SegmentationExtractor`. [PR #291](https://github.com/catalystneuro/roiextractors/pull/291)

* Add distinction for raw roi response and denoised roi response in `CaimanSegmentationExtractor`: [PR #291](https://github.com/catalystneuro/roiextractors/pull/291)

* Bug fix for the `CaimanSegmentationExtractor`: correctly extract temporal and spatial background components [PR #291](https://github.com/catalystneuro/roiextractors/pull/291)

* Added automatic version update workflow file that will run after publishing a new release to pypi: [PR #290](https://github.com/catalystneuro/roiextractors/pull/290)

* Added `ScanImageTiffSinglePlaneMultiFileImagingExtractor` and `ScanImageTiffMultiPlaneMultiFileImagingExtractor`: [PR #297](https://github.com/catalystneuro/roiextractors/pull/297/files)

* Added automatic changelog checking in the test workflow: [PR #302](https://github.com/catalystneuro/roiextractors/pull/302)

### Fixes

* Improved xml parsing with Bruker [PR #267](https://github.com/catalystneuro/roiextractors/pull/267)

* Fixed a bug with `ScanImageTiffSinglePlaneImagingExtractor` in which `frames_per_slice` would be set to `_num_frames`: [PR #294](https://github.com/catalystneuro/roiextractors/pull/294)



# v0.5.6

### Features

* Added support for red channel (anatomical) ROIs from suite2p in Suite2pSegmentationExtractor: [PR #270](https://github.com/catalystneuro/roiextractors/pull/270)

* Added support for RoiGroup metadata in the `extract_extra_metadata` function for ScanImageTiff files: [PR #272](https://github.com/catalystneuro/roiextractors/pull/272)

* Updated documentation and Readme PRs: [#283](https://github.com/catalystneuro/roiextractors/pull/283) [#282](https://github.com/catalystneuro/roiextractors/pull/282) [#280](https://github.com/catalystneuro/roiextractors/pull/280)



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

* Added support for incomplete file ingestion for the `Suite2pSegmentationExtractor`. [PR #227](https://github.com/catalystneuro/roiextractors/pull/227)

* Bug fix for the `CaimanSegmentationExtractor`: Change reshaping from 'C' to 'F' (Fortran). [PR #227](https://github.com/catalystneuro/roiextractors/pull/227)

* Bug fix for the `CaimanSegmentationExtractor`: Added importing of `self._image_correlation` and changed how `self._image_mean` to import the background component image. [PR #227](https://github.com/catalystneuro/roiextractors/pull/227)


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
