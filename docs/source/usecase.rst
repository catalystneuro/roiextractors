Use Case
========
This will take you though a use case of working with raw image data in TIFF format as well as post processed data using Caiman ROI extraction software.

Download a TIF file and a extract pipeline output .mat file from `here. <https://gin.g-node.org/CatalystNeuro/ophys_testing_data/src/master/segmentation_datasets/>`_

Create an Imaging Extractor Object:
-----------------------------------

.. jupyter-execute::

    from notebook.services.config import ConfigManager
    cm = ConfigManager().update('notebook', {'limit_output': 1000})
    import roiextractors
    import matplotlib.pyplot as plt
    %matplotlib inline
    import numpy as np
    img_ext = roiextractors.TiffImagingExtractor(r'source\demoMovie.tif',sampling_frequency=100)

.. jupyter-execute::

    img_ext.get_frames(10)

.. jupyter-execute::

    img_ext.get_image_size()

.. jupyter-execute::

    img_ext.get_num_frames()

.. jupyter-execute::

    img_ext.get_sampling_frequency()

.. jupyter-execute::

    img_ext.get_channel_names()

.. jupyter-execute::

    img_ext.get_num_channels()

.. jupyter-execute::

    vid_fra = img_ext.get_video(start_frame=0,end_frame=1)
    plt.imshow(vid_fra)
    plt.show()


Create a SegmentationExtractor Object
-------------------------------------

.. jupyter-execute::

    seg_ext = roiextractors.CaimanSegmentationExtractor(r'source\caiman.hdf5')

.. jupyter-execute::

    # will output a list of ids of all accepted rois
    seg_ext.get_accepted_list()[:5]

.. jupyter-execute::

    seg_ext.get_num_frames()

.. jupyter-execute::

    seg_ext.get_roi_locations(roi_ids=[2])[:,:10]

.. jupyter-execute::

    plt.plot(seg_ext.get_sampling_frequency()*np.arange(10,100),seg_ext.get_traces(roi_ids=[2],start_frame=10,end_frame=100, name='dff').squeeze())
    plt.show()

.. jupyter-execute::

    plt.imshow(seg_ext.get_roi_image_masks(roi_ids=[5]).squeeze())
    plt.show()

.. jupyter-execute::

    plt.imshow(seg_ext.get_image())
    plt.show()

.. jupyter-execute::

    seg_ext.get_image_size()

.. jupyter-execute::

    seg_ext.get_num_rois()





