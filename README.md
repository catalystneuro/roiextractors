[![PyPI version](https://badge.fury.io/py/roiextractors.svg)](https://badge.fury.io/py/roiextractors)
![Full Tests](https://github.com/catalystneuro/roiextractors/actions/workflows/run-tests.yml/badge.svg)
![Auto-release](https://github.com/catalystneuro/roiextractors/actions/workflows/auto-publish.yml/badge.svg)
[![codecov](https://codecov.io/github/catalystneuro/roiextractors/coverage.svg?branch=master)](https://codecov.io/github/catalystneuro/roiextractors?branch=master)
[![documentation](https://readthedocs.org/projects/roiextractors/badge/?version=latest)](https://roiextractors.readthedocs.io/en/latest/)
[![License](https://img.shields.io/pypi/l/pynwb.svg)](https://github.com/catalystneuro/roiextractors/license.txt)

# ROIExtractors
<p align="center">
  <h3 align="center">Automatically read optical imaging/segmentation data into a common API</h3>
</p>
<p align="center">
   <a href="roiextractors.readthedocs.io"><strong>Explore our documentation »</strong></a>
</p>

<!-- TABLE OF CONTENTS -->

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Documentation](#documentation)
- [Funding](#funding)
- [License](#license)

## About

ROIExtractors provides a common API for various optical imaging and segmentation formats to streamline conversion and data analysis. ROI stands for Region Of Interest, which is the region in a set of acquired fluorescence images which the segmentation software has determined as a neuron.

Features:

* Reads data from 10+ popular optical imaging and segmentation data formats into a common API.
* Extracts relevant metadata from each format.
* Provides a handy set of methods to analyze data such as `get_roi_locations()`

## Installation

To install the latest stable release of **roiextractors** though PyPI, type:
```shell
pip install roiextractors
```

For more flexibility we recommend installing the latest version directly from GitHub. The following commands create an environment with all the required dependencies and the latest updates:

```shell
git clone https://github.com/catalystneuro/roiextractors
cd roiextractors
conda env create roiextractors_env
conda activate roiextractors_env
pip install -e .
```
Note that this will install the package in [editable mode](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs).

Finally, if you prefer to avoid `conda` altogether, the following commands provide a clean installation within the current environment:
```shell
pip install git+https://github.com/catalystneuro/roiextractors.git@main
```

## Documentation
See our [ReadTheDocs page](https://roiextractors.readthedocs.io/en/latest/) for full documentation, including a gallery of all supported formats.

## Funding
ROIExtractors is funded by
* Stanford University as part of the Ripple U19 project (U19NS104590).
* LBNL as part of the NWB U24 (U24NS120057).

## License
ROIExtractors is distributed under the BSD3 License. See [LICENSE](https://github.com/catalystneuro/roiextractors/blob/main/LICENSE.txt) for more information.
