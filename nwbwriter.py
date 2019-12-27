from pynwb import NWBHDF5IO, TimeSeries, NWBFile
from pynwb.base import Images
from pynwb.image import GrayscaleImage
from pynwb.ophys import ImageSegmentation, Fluorescence, OpticalChannel, ImageSeries
from pynwb.device import Device
import os
import uuid
from datetime import datetime
from dateutil.tz import tzlocal
import time
import logging
import numpy as np
import re


def set_dynamic_table_property(dynamic_table, row_ids, property_name, values, index=False,
                               default_value=np.nan, description='no description'):
    check_nwb_install()
    if not isinstance(row_ids, list) or not all(isinstance(x, int) for x in row_ids):
        raise TypeError("'ids' must be a list of integers")
    ids = list(dynamic_table.id[:])
    if any([i not in ids for i in row_ids]):
        raise ValueError("'ids' contains values outside the range of existing ids")
    if not isinstance(property_name, str):
        raise TypeError("'property_name' must be a string")
    if len(row_ids) != len(values) and index is False:
        raise ValueError("'ids' and 'values' should be lists of same size")

    if index is False:
        if property_name in dynamic_table:
            for (row_id, value) in zip(row_ids, values):
                dynamic_table[property_name].data[ids.index(row_id)] = value
        else:
            col_data = [default_value] * len(ids)  # init with default val
            for (row_id, value) in zip(row_ids, values):
                col_data[ids.index(row_id)] = value
            dynamic_table.add_column(
                name=property_name,
                description=description,
                data=col_data,
                index=index
            )
    else:
        if property_name in dynamic_table:
            # TODO
            raise NotImplementedError
        else:
            dynamic_table.add_column(
                name=property_name,
                description=description,
                data=values,
                index=index
            )


def get_dynamic_table_property(dynamic_table, *, row_ids=None, property_name):
    all_row_ids = list(dynamic_table.id[:])
    if row_ids is None:
        row_ids = all_row_ids
    return [dynamic_table[property_name][all_row_ids.index(x)] for x in row_ids]


def write_recording(TraceExtractor, propertydict=[], filename, imaging_plane_name=None, imaging_series_name=None, identifier=None,
                    starting_time=0., session_start_time=datetime.now(tzlocal()), excitation_lambda=np.nan, imaging_plane_description='no description',
                    emission_lambda=np.nan, indicator='none', location='brain', device_name='MyDevice',
                    optical_channel_name='MyOpticalChannel', optical_channel_description='MyOpticalChannelDescription',
                    imaging_plane_name='MyImagingPlane', imaging_plane_description='MyImagingPlaneDescription',
                    image_series_name='MyImageSeries', image_series_description='MyImageSeriesDescription',
                    processing_module_name='Ophys', processing_module_description='ContainsProcessedData',
                    neuron_roi_response_series_name='NeuronTimeSeriesData',
                    background_roi_response_series_name='BackgroundTimeSeriesData', **nwbfile_kwargs):
    """writes NWB file
    Args:
        filename: str
        imaging_plane_name: str, optional
        imaging_series_name: str, optional
        sess_desc: str, optional
        exp_desc: str, optional
        identifier: str, optional
        imaging_rate: float, optional
            default: 30 (Hz)
        starting_time: float, optional
            default: 0.0 (seconds)
        location: str, optional
        session_start_time: datetime.datetime, optional
            Only required for new files
        excitation_lambda: float
        imaging_plane_description: str
        emission_lambda: float
        indicator: str
        location: str
    """
    imaging_rate = TraceExtractor.get_sampling_frequency()
    raw_data_file = TraceExtractor.get_raw_file()

    if identifier is None:
        identifier = uuid.uuid1().hex

    if '.nwb' != os.path.splitext(filename)[-1].lower():
        raise Exception("Wrong filename")

    if os.path.exists(save_path):
        read_mode = 'r+'
        _nwbfile_exist = True
    else:
        _nwbfile_exist = False
        read_mode = 'w'

    with NWBHDF5IO(save_path, mode=read_mode) as io:
        if _nwbfile_exist:
            nwbfile = io.read()
            _nwbchildren_type = [type(i).__name__ for i in nwbfile.all_children()]
            _nwbchildren_name = [i.name for i in nwbfile.all_children()]
            _custom_names = [device_name, optical_channel_name, imaging_plane_name,
                             image_series_name]
            # ADDING ACQUISITION DATA:
            # check existance of device, optical channel, imaging plane, image series:
            _device_exist = [i for i, e in enumerate(_nwbchildren_name) if e == device_name]
            if not _device_exist:
                nwbfile.add_device(Device(device_name))
            elif len(_device_exist > 1):
                raise Exception('Multiple Devices exist, provide name of one')

            _optical_channel_exist = [i for i, e in enumerate(
                _nwbchildren_name) if e == optical_channel_name]
            if not _optical_channel_exist:
                optical_channel = OpticalChannel(optical_channel_name,
                                                 optical_channel_description,
                                                 emission_lambda=emission_lambda)
             elif len(_optical_channel_exist)==1:
                 optical_channel=nwbfile.all_children()[_optical_channel_exist[0]]
             else:
                 raise Exception('Multiple Optical Channels exist, provide name of one')

            _imaging_plane_exist = [i for i, e in enumerate(_nwbchildren_name) if e == imaging_plane_name]
            if not _imaging_plane_exist:
                nwbfile.create_imaging_plane(name=imaging_plane_name,
                                             optical_channel=optical_channel,
                                             description=imaging_plane_description,
                                             device=device,
                                             excitation_lambda=excitation_lambda,
                                             imaging_rate=imaging_rate,
                                             indicator=indicator,
                                             location=location)
            elif len(_imaging_plane_exist)==1:
                optical_channel=nwbfile.all_children()[_imaging_plane_exist[0]]
            else:
                raise Exception('Multiple Imaging Planes exist, provide name of one')

            _image_series_exist = [i for i, e in enumerate(_nwbchildren_name) if e == image_series_name]
            if not _image_series_exist:
                nwbfile.add_acquisition(ImageSeries(name=image_series_name,
                                                    description=image_series_description,
                                                    external_file=[raw_data_file],
                                                    format='external',
                                                    rate=imaging_rate,
                                                    starting_frame=[0]))
            elif len(_image_series_exist)==1:
                optical_channel=nwbfile.all_children()[_image_series_exist[0]]
            else:
                raise Exception('Multiple Imaging Series exist, provide name of one')

        else:
            kwargs = {'session_description': 'No description',
                      'identifier': str(uuid.uuid4()),
                      'session_start_time': datetime.now()}
            kwargs.update(**nwbfile_kwargs)
            nwbfile = NWBFile(**kwargs)
            nwbfile.add_device(Device(device_name))

            optical_channel = OpticalChannel(optical_channel_name,
                                             optical_channel_description,
                                             emission_lambda=emission_lambda)
            nwbfile.create_imaging_plane(name=imaging_plane_name,
                                         optical_channel=optical_channel,
                                         description=imaging_plane_description,
                                         device=device,
                                         excitation_lambda=excitation_lambda,
                                         imaging_rate=imaging_rate,
                                         indicator=indicator,
                                         location=location)
            nwbfile.add_acquisition(ImageSeries(name=image_series_name,
                                                description=image_series_description,
                                                external_file=[raw_data_file],
                                                format='external',
                                                rate=imaging_rate,
                                                starting_frame=[0]))
        with NWBHDF5IO(filename, 'w') as io:
            io.write(nwbfile)

    time.sleep(4)  # ensure the file is fully closed before opening again in append mode
    logging.info('Saving the results in the NWB file...')

    with NWBHDF5IO(filename, 'r+') as io:
        nwbfile = io.read()
        # ADDING PROCESSING DATA:
        # Adding Ophys
        _ophys_module_exist = [i for i, e in enumerate(_nwbchildren_name) if e == processing_module_name]
        if not _ophys_module_exist:
            mod = nwbfile.create_processing_module(
                processing_module_name, processing_module_description)
        elif len(_ophys_module_exist)==1:
            mod = nwbfile.all_children()[_ophys_module_exist[0]]
        else:
            raise Exception('Multiple Ophys modules exist, provide name of one')
        # Adding ImageSegmentation:
        _image_segmentation_exist = [i for i, e in enumerate(_nwbchildren_name) if e == 'ImageSegmentation']
        if not _image_segmentation_exist:
            img_seg = ImageSegmentation()
            mod.add_data_interface(img_seg)
        else:
            img_seg = nwbfile.all_children()[_image_segmentation_exist[0]]
        # Adding Fluorescence:
        _fluorescence_module_exist = [i for i, e in enumerate(_nwbchildren_name) if e == 'Fluorescence']
        if not _fluorescence_module_exist:
            fl = Fluorescence()
            mod.add_data_interface(fl)
        else:
            fl = nwbfile.all_children()[_fluorescence_module_exist[0]]

        # Adding the ROI-related stuff:
        imaging_plane = nwbfile.imaging_planes[imaging_plane_name]
        image_series = nwbfile.acquisition[imaging_series_name]
        # Adding PlaneSegmentation:
        _plane_segmentation_exist = [i for i, e in enumerate(_nwbchildren_name) if e == 'PlaneSegmentation']
        if not _plane_segmentation_exist:
            ps = img_seg.create_plane_segmentation(
                'ROIs', imaging_plane, 'PlaneSegmentation', image_series)
        else:
            ps = nwbfile.all_children()[_plane_segmentation_exist[0]]

        # Adding custom columns and their values to the PlaneSegmentation table:
        # propertydict is a list of dictionaries containing [{'name':'','discription':''}, {}..]
        if len(propertydict):
            trace_exctractor_attrs=dir(TraceExtractor)
            for i in range(len(propertydict)):
                _property_name = propertydict[i]['name']
                _property_desc = propertydict[i].get('description','no description'])
                _property_row_ids = propertydict[i].get('id',list(np.arange(TraceExtractor.no_rois))])
                _property_name_attr=[re.findall('^' + _property_name,i,re.I) for i in trace_exctractor_attrs]
                assert not _property_name_attr or len(_property_name_attr)>1, \
                print('Enter one of the Names and their descriptions in the Property Dictionary:\n'
                      '\'r\' : Residual Noise \n'
                      '\'cnn\' : CNN predictions for each component \n'
                      '\'keep\' : ROIs to keep \n'
                      '\'accepted\' : accepted ROIs \n'
                      '\'rejected\' : rejected ROIs \n')
                value = getattr(TraceExtractor,_property_name_attr,False)
                if value:
                    set_dynamic_table_property(ps, _property_row_ids, _property_name, value, index=False,
                                               description=_property_desc)

       # Adding Image and Pixel Masks(default colnames in PlaneSegmentation):
        if hasattr(TraceExtractor, 'pixel_masks'):
            for i, (img_roi, pix_roi) in enumerate(zip(TraceExtractor.image_masks.T, TraceExtractor.pixel_masks):
                ps.add_roi(image_mask=img_roi.T.reshape(TraceExtractor.dims),
                           pixel_mask=pix_roi)
        else:
            for i, img_roi in enumerate(TraceExtractor.image_masks.T):
                ps.add_roi(image_mask=img_roi.T.reshape(TraceExtractor.dims))

        # Background components addition:
        if hasattr(TraceExtractor, 'image_masks_bk'):
            if hasattr(TraceExtractor, 'pixel_masks_bk'):
                for i, (img_roi_bk, pix_roi_bk) in enumerate(zip(TraceExtractor.image_masks_bk.T, TraceExtractor.pixel_masks_bk):
                    ps.add_roi(image_mask=img_roi_bk.T.reshape(TraceExtractor.dims),
                               pixel_mask=pix_roi_bk)
            else:
                for i, img_roi_bk in enumerate(TraceExtractor.image_masks_bk.T):
                    ps.add_roi(image_mask=img_roi_bk.T.reshape(TraceExtractor.dims))

        # Add Traces
        no_neurons_rois = TraceExtractor.image_masks.shape[-1]
        no_background_rois = len(TraceExtractor.roi_response_bk)
        neuron_rois = ps.create_roi_table_region(
            'NeuronROIs', region=list(range(no_neurons_rois)))

        background_rois = ps.create_roi_table_region(
            'BackgroundROIs', region=list(range(no_neurons_rois, no_neurons_rois+no_background_rois)))

        timestamps = np.arange(
            TraceExtractor.roi_response_bk.shape[1]) / imaging_rate + starting_time

        # Neurons TimeSeries
        neuron_roi_response_exist = [i for i, e in enumerate(_nwbchildren_name) if e == neuron_roi_response_series_name]
        if not neuron_roi_response_exist:
            fl.create_roi_response_series(name=neuron_roi_response_series_name, data=TraceExtractor.roi_response.T,
                                          rois=neuron_rois, unit='lumens', timestamps=timestamps,
                                          starting_time=starting_time, rate=imaging_rate)
        else:
            raise Exception('Time Series for' + neuron_roi_response_series_name +
                            ' already exists , provide another name')

        # Background TimeSeries
        background_roi_response_exist = [i for i, e in enumerate(_nwbchildren_name) if e == background_roi_response_series_name]
        if not background_roi_response_exist:
            fl.create_roi_response_series(name=background_roi_response_series_name, data=TraceExtractor.roi_response.T,
                                          rois=neuron_rois, unit='lumens', timestamps=timestamps,
                                          starting_time=starting_time, rate=imaging_rate)
        else:
            raise Exception('Time Series for' + background_roi_response_series_name +
                            ' already exists , provide another name')

        # Residual TimeSeries
        mod.add(TimeSeries(name='residuals', description='residuals', data=TraceExtractor.roi_response_residual.T, timestamps=timestamps,
                           unit='NA'))
        if hasattr(TraceExtractor, 'cn'):
            images = Images('summary_images')
            images.add_image(GrayscaleImage(name='local_correlations', data=TraceExtractor.cn))

            # Add MotionCorreciton
    #            create_corrected_image_stack(corrected, original, xy_translation, name='CorrectedImageStack')
            io.write(nwbfile)
