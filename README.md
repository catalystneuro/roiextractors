[![PyPI version](https://badge.fury.io/py/roiextractors.svg)](https://badge.fury.io/py/roiextractors)
![Full Tests](https://github.com/catalystneuro/roiextractors/actions/workflows/run-tests.yml/badge.svg)
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

See [documentation](https://roiextractors.readthedocs.io/en/latest/) for details.

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
This package provides a common API for various optical imaging and segmentation formats streamline conversion and data analysis.

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

## Troubleshooting
##### Installing SIMA with python>=3.7:
Will need a manual installation for package dependency **SIMA** since it does not currently support python 3.7:
1.   Download SIMA wheels distribution [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#sima).
2.  `pip install <download-path-to-wheels.whl>`
3.  `pip install roiextractors`

### Funded by
* Stanford University as part of the Ripple U19 project (U19NS104590).
* LBNL as part of the NWB U24 (U24NS120057).
