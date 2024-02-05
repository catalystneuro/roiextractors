.. RoiExtractors documentation master file, created by
   sphinx-quickstart on Thu Jan  7 20:59:45 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _indextag:

Welcome to RoiExtractors's documentation!
=========================================
Roiextractors is a library that helps in analyzing, visualizing and interacting with optical physiology data acquired from various acquisition systems.

**ROI**
   Stands for Region Of Interest, which is the region in a set of acquired fluorescence images which the segmentation software has determined as a neuron.

With this package, a user can:

- Work with imaging data in formats like: TIFF, HDF5, STK, FLI.
- Work with post-processed data (after application of cell extraction/segmentation) output from various commonly used cell extraction/segmentation packages like Suite2p, CNMF-E, Caiman, SIMA.
- Save all this data into NWB format and share it with the community!

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:
   :numbered:

   gettingstarted
   compatible
   usage
   usecase
   contribute
   licence
   contact
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
