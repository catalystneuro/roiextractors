[![PyPI version](https://badge.fury.io/py/roiextractors.svg)](https://badge.fury.io/py/roiextractors)
![Full Tests](https://github.com/catalystneuro/roiextractors/actions/workflows/testing.yml/badge.svg)
![Auto-release](https://github.com/catalystneuro/roiextractors/actions/workflows/auto-publish.yml/badge.svg)
[![codecov](https://codecov.io/github/catalystneuro/roiextractors/coverage.svg?branch=master)](https://codecov.io/github/catalystneuro/roiextractors?branch=master)
[![documentation](https://readthedocs.org/projects/roiextractors/badge/?version=latest)](https://roiextractors.readthedocs.io/en/latest/)
[![License](https://img.shields.io/pypi/l/pynwb.svg)](https://github.com/catalystneuro/roiextractors/license.txt)

# ROI Extractors
Python-based module for extracting from, converting between, and handling recorded and optical imaging data from several file formats. Inspired by [SpikeExtractors](https://github.com/SpikeInterface/spikeextractors).

Developed by [CatalystNeuro](http://catalystneuro.com/).

## Getting Started:
#### Installation:
`pip install roiextractors`

## Usage:
### Supported file types:
#### Imaging
1. HDF5
2. TIFF
3. STK
4. FLI
5. SBX

#### Segmentation
1. [calciumImagingAnalysis](https://github.com/bahanonu/calciumImagingAnalysis) (CNMF-E, EXTRACT)
2. [SIMA](http://www.losonczylab.org/sima/1.3.2/)
3. [NWB](https://pynwb.readthedocs.io/en/stable/)
4. [suite2p](https://github.com/MouseLand/suite2p)
45. Numpy (a data format for manual input of optical physiology data as various numpy datasets)

#### Functionality:
Interconversion amongst the various data formats as well as conversion to the NWB format and back.

#### Features:
1. SegmentationExtractor object:
    * `seg_obj.get_channel_names()` :
    List of optical channel names
    * `seg_obj.get_num_channels()` :
    Number of channels
    * `seg_obj.get_movie_framesize()`:
    (height, width) of raw movie
    * `seg_obj.get_movie_location()`:
    Location of storage of movie/tiff images
    * `seg_obj.get_image_masks(self, roi_ids=None)`:
    Image masks as (ht, wd, num_rois) with each value as the weight given during segmentation operation.
    * `seg_obj.get_pixel_masks(roi_ids=None)`:
    Get pixel masks as (total_pixels(ht*wid), no_rois)
    * `seg_obj.get_traces(self, roi_ids=None, start_frame=None, end_frame=None)`:
    df/F trace as (num_rois, num_frames)
    * `seg_obj.get_sampling_frequency()`:
    Sampling frequency of movie/df/F trace.
    * `seg_obj.get_roi_locations()`:
    Centroid pixel location of the ROI (Regions Of Interest) as (x,y).
    * `seg_obj.get_num_rois()`:
    Total number of ROIs after segmentation operation.
    * `seg_obj.get_roi_ids()`:
    Any integer tags associated with an ROI, defaults to `0:num_of_rois`

#### SegmentationExtractor object creation:
```python
import roiextractors
import numpy as np

seg_obj_cnmfe = roiextractors.CnmfeSegmentationExtractor('cnmfe_filename.mat') # cnmfe
seg_obj_extract = roiextractors.ExtractSegmentationExtractor('extract_filename.mat') # extract
seg_obj_sima = roiextractors.SimaSegmentationExtractor('sima_filename.sima') # SIMA
seg_obj_numpy = roiextractors.NumpySegmentationExtractor(
        filepath = 'path-to-file',
        masks=np.random.rand(movie_size[0],movie_size[1],no_rois),
        signal=np.random.randn(num_rois,num_frames),
        roi_idx=np.random.randint(no_rois,size=[1,no_rois]),
        no_of_channels=None,
        summary_image=None,
        channel_names=['Blue'],
) # Numpy object
seg_obj_nwb = roiextractors.NwbSegmentationExtractor(
                    filepath_of_nwb, optical_channel_name=None, # optical channel to extract and store info from
                    imaging_plane_name=None, image_series_name=None, # imaging plane to extract and store data from
                    processing_module_name=None,
                    neuron_roi_response_series_name=None, # roi_response_series name to extract and store data from
                    background_roi_response_series_name=None) # nwb object
```
#### Data format conversion: SegmentationExtractor to NWB:
```python
roiextractors.NwbSegmentationExtractor.write_segmentation(seg_obj, saveloc,
                    propertydict=[{'name': 'ROI feature 1,
                                   'description': 'additional attribute of each ROI',
                                   'data': np.random.rand(1,no_rois),
                                   'id': seg_obj.get_roi_ids()},
                                  {'name': 'ROI feature 2,
                                   'description': 'additional attribute of each ROI',
                                   'data': np.random.rand(1,no_rois),
                                   'id': seg_obj.get_roi_ids()}],
                    nwbfile_kwargs={'session_description': 'nwbfiledesc',
                                    'experimenter': 'experimenter name',
                                    'lab': 'test lab',
                                    'session_id': 'test sess id'},
                    emission_lambda=400.0, excitation_lambda=500.0)
```
## Example Datasets:
Example datasets are maintained at https://gin.g-node.org/CatalystNeuro/ophys_testing_data.

To download test data on your machine,

1. Install the gin client (instructions [here](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Setup#linux))
2. Use gin to download data.
```shell
gin get CatalystNeuro/ophys_testing_data
cd ophys_testing_data
gin get-content
```

3. Change the file at `roiextractors/tests/gin_test_config.json` to point to the path of this test data

To update data later, `cd` into the test directory and run `gin get-content`


## Class descriptions:

*  **SegmentationExtractor:** An abstract class that contains all the meta-data and output data from the ROI segmentation operation when applied to the pre-processed data. It also contains methods to read from and write to various data formats ouput from  the processing pipelines like SIMA, CaImAn, Suite2p, CNNM-E.

*  **NumpySegmentationExtractor:** Contains all data coming from a file format for which there is currently no support. To construct this, all data must be entered manually as arguments.

*  **CnmfeSegmentationExtractor:** This class inherits from the SegmentationExtractor class, having all its funtionality specifically applied to the dataset output from the 'CNMF-E' ROI segmentation method.

*  **ExtractSegmentationExtractor:** This class inherits from the SegmentationExtractor class, having all its funtionality specifically applied to the dataset output from the 'EXTRACT' ROI segmentation method.

*  **SimaSegmentationExtractor:** This class inherits from the SegmentationExtractor class, having all its funtionality specifically applied to the dataset output from the 'SIMA' ROI segmentation method.

*  **NwbSegmentationExtractor:** Extracts data from the NWB data format. Also implements a static method to write any format specific object to NWB.

* **Suite2PSegmentationExtractor:** Extracts data from suite2p format.

## Troubleshooting
##### Installing SIMA with python>=3.7:
Will need a manual installation for package dependency **SIMA** since it does not currently support python 3.7:
1.   Download SIMA wheels distribution [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#sima).
2.  `pip install <download-path-to-wheels.whl>`
3.  `pip install roiextractors`

### Funded by
* Stanford University as part of the Ripple U19 project (U19NS104590).
* LBNL as part of the NWB U24 (U24NS120057).
