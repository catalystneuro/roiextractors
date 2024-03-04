Contribute
===========

Example Datasets
----------------

Example datasets are maintained at https://gin.g-node.org/CatalystNeuro/ophys_testing_data.

To download test data on your machine,

1. Install the gin client (instructions `here <https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Setup#linux>`_)
2. Use gin to download data:

   .. code-block:: bash

      gin get CatalystNeuro/ophys_testing_data
      cd ophys_testing_data
      gin get-content

3. Change the file at ``roiextractors/tests/gin_test_config.json`` to point to the path of this test data

To update data later, ``cd`` into the test directory and run ``gin get-content``

Troubleshooting
---------------

Installing SIMA with python>=3.7:
________________________________
Will need a manual installation for package dependency **SIMA** since it does not currently support python 3.7:

1.   Download SIMA wheels distribution `here <https://www.lfd.uci.edu/~gohlke/pythonlibs/#sima>`_.
2.  `pip install <download-path-to-wheels.whl>`
3.  `pip install roiextractors`

More details on how to construct individual extractors can be found here:

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :numbered:

   build_ie
   build_re
