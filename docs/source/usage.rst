===============
Step wise usage
===============

Functionality
==============
Interconversion amongst the various data formats as well as conversion to the NWB format and back.

Features
---------
1. **SegmentationExtractor object**:

- ``seg_obj.get_image_masks(self, roi_ids=None)``:Image masks as (ht, wd, num_rois) with each value as the weight given during segmentation operation.
- ``seg_obj.get_pixel_masks(roi_ids=None)``: Get pixel masks as (total_pixels(ht*wid), no_rois)
- ``seg_obj.get_traces(self, roi_ids=None, start_frame=None, end_frame=None)``: df/F trace as (num_rois, num_frames)
- ``seg_obj.get_sampling_frequency()``: Sampling frequency of movie/df/F trace.
- ``seg_obj.get_roi_locations()``: Centroid pixel location of the ROI (Regions Of Interest) as (x,y).
- ``seg_obj.get_num_rois()``: Total number of ROIs after segmentation operation.
- ``seg_obj.get_roi_ids()``: Any integer tags associated with an ROI, defaults to `0:num_of_rois`

SegmentationExtractor object creation
--------------------------------------

.. code-block:: python
    :linenos:

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
                        channel_names=['Blue']) # Numpy object
    seg_obj_nwb = roiextractors.NwbSegmentationExtractor(
                        filepath_of_nwb, optical_channel_name=None, # optical channel to extract and store info from
                        imaging_plane_name=None, image_series_name=None, # imaging plane to extract and store data from
                        processing_module_name=None,
                        neuron_roi_response_series_name=None, # roi_response_series name to extract and store data from
                        background_roi_response_series_name=None) # nwb object

Data format conversion: SegmentationExtractor to NWB
-----------------------------------------------------

.. note::
   The ``roiextractors.NwbSegmentationExtractor.write_segmentation`` method has been deprecated.
   Please use ``neuroconv`` for writing segmentation data to NWB format.

.. code-block:: python
    :linenos:

    # Import write_segmentation_to_nwbfile from neuroconv instead of roiextractors
    from neuroconv.tools.roiextractors import write_segmentation_to_nwbfile

    write_segmentation(seg_obj, saveloc)

    See the `neuroconv documentation <https://neuroconv.readthedocs.io/en/stable/api/tools.roiextractors.html>`_ for more details.
