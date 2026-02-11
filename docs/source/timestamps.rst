Timestamp Handling
==================

Getting Timestamps
------------------

To get the timestamps associated with any extractor, use ``get_timestamps()``:

.. code-block:: python

    from roiextractors import NwbImagingExtractor

    extractor = NwbImagingExtractor(file_path="my_data.nwb")

    # For each sample, get the timestamp (in seconds)
    timestamps = extractor.get_timestamps()

    # Get timestamps for a specific range of frames
    timestamps = extractor.get_timestamps(start_sample=100, end_sample=200)

This method is available on both ``ImagingExtractor`` and ``SegmentationExtractor``.
It **always returns an array** of times in seconds. You never need to check whether timestamps
exist before calling it. The source of those timestamps depends on the extractor:

- If the user has set custom timestamps via ``set_times()``, those take priority.
- Otherwise, if the data format stores hardware timestamps (e.g., DAQ-recorded times), those
  are returned.
- If neither exists, timestamps are reconstructed from the sampling frequency as
  ``np.arange(start, end) / sampling_frequency``.

This means every extractor provides timestamps, regardless of whether the underlying
format includes them.


Setting Custom Timestamps
--------------------------

You can override the timestamps with ``set_times()``:

.. code-block:: python

   import numpy as np

   # Set custom timestamps (must match the number of frames)
   custom_times = np.linspace(0, 10, extractor.get_num_samples())
   extractor.set_times(custom_times)

   # Check if custom times have been set
   extractor.has_time_vector()  # True

``set_times()`` validates that the length of the provided array matches ``get_num_samples()``.


.. seealso::

    To implement timestamps in a custom extractor, see :doc:`build_ie` or :doc:`build_re`.

    For the rationale behind the timestamp API design, see :doc:`design_decisions`.

    For the full method signatures and parameters, see the API reference for
    :doc:`api/base_imagingextractors` and :doc:`api/base_segmentationextractors`.
