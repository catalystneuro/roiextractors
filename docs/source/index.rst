.. RoiExtractors documentation master file, created by
   sphinx-quickstart on Thu Jan  7 20:59:45 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _indextag:

Welcome to RoiExtractors's documentation!
=========================================
Roiextractors is a library that helps in analyzing, visualizing and interacting with optical physiology data acquired from various acquisition systems.

**ROI**
   Stands for Region Of Interest, which is the set of pixels from acquired fluorescence images which the segmentation software has determined to follow a particular cellular structure.

With this package, a user can:

- Work with imaging data in formats like: ScanImage TIFF, HDF5, Miniscope, Scanbox and more.
- Work with post-processed data (after application of cell extraction/segmentation) output from various commonly used cell extraction/segmentation packages like Suite2p, CNMF-E, Caiman, SIMA.
- Leverage a common API to streamline analysis and visualization of data from different acquisition systems and cell extraction/segmentation pipelines.

.. seealso::

   If you want to write data to NWB, you can check out our primary dependent: `NeuroConv <https://neuroconv.readthedocs.io/en/main/index.html>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   gettingstarted
   compatible
   usage
   usecase
   contribute
   licence
   contact
