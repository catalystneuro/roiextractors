Getting Started
================

Installation
------------

#. Using Pip:
    .. code-block:: shell

       $ pip install roiextractors


#. **Cloning the github repo**:
    .. code-block:: shell

        $ git clone https://github.com/SpikeInterface/spikeinterface.git
        $ cd spikeinterface
        $ pip install --editable .

You can also install the optional dependencies by installing the package with the following command:

.. code-block:: shell

    $ pip install "roiextractors[full]"

The `full` extra installs format-specific dependencies.

For development, testing, or building documentation, use dependency groups:

.. code-block:: shell

    $ pip install --group test
    $ pip install --group docs

These commands install the test and documentation dependencies, respectively.

What is RoiExtractors
---------------------
Python-based module for extracting from, converting between, and handling recorded and optical imaging data from several file formats. Inspired by `SpikeExtractors <https://github.com/SpikeInterface/spikeextractors/>`_.

.. image:: ./_images/roiextractors_overview.jpg
