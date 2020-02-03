import os
import uuid
from datetime import datetime
from dateutil.tz import tzlocal
import numpy as np
import re

try:
    from pynwb import NWBHDF5IO, TimeSeries, NWBFile
    from pynwb.base import Images
    from pynwb.image import GrayscaleImage
    from pynwb.ophys import ImageSegmentation, Fluorescence, OpticalChannel, TwoPhotonSeries
    from pynwb.device import Device

    HAVE_NWB = True
except ModuleNotFoundError:
    HAVE_NWB = False


def check_nwb_install():
    assert HAVE_NWB, "To use the Nwb extractors, install pynwb: \n\n pip install pynwb\n\n"


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


class NwbSegmentationExtractor(segmentation_extractor_obj):

    def __init__(self, filepath, optical_channel_name=None,
                 imaging_plane_name=None, image_series_name=None,
                 processing_module_name=None,
                 neuron_roi_response_series_name=None,
                 background_roi_response_series_name=None):

        check_nwb_install()
        if not os.path.exists(filepath):
            raise Exception('file does not exist')

        self.filepath = filepath

        with NWBHDF5IO(filepath, mode='r+') as io:
            nwbfile = io.read()
            _nwbchildren_type = [type(i).__name__ for i in nwbfile.all_children()]
            _nwbchildren_name = [i.name for i in nwbfile.all_children()]
            mod = nwbfile.processing
            if len(mod) > 1:
                print('multiple processing modules found, picking the first one')
                mod = list(nwbfile.processing.values())[0]
            elif not mod:
                raise Exception('no processing module found')

            # Extract image_mask/background:
            _plane_segmentation_exist = [i for i, e in enumerate(
                _nwbchildren_name) if e == 'PlaneSegmentation']
            if not _plane_segmentation_exist:
                print('could not find a plane segmentation to contain image mask')
            else:
                ps = nwbfile.all_children()[_plane_segmentation_exist[0]]
            self.image_masks = np.moveaxis(ps['image_mask'].data, [0, 1, 2], [2, 0, 1])
            self.raw_images = self.image_masks

            # Extract pixel_mask/background:
            self.pixel_masks = np.moveaxis(ps['pixel_mask'].data, [0, 1, 2], [2, 0, 1])
            # Extract Image dimensions:
            self.extimage_dims = ps['image_mask'].data.shape[1::]

            # Extract roi_response:
            self._no_background_comps =
            _roi_exist = [_nwbchildren_name[val]
                          for val, i in enumerate(_nwbchildren_type) if i == 'RoiResponseSeries']

            if not _roi_exist:
                raise Exception('no ROI response series found')
            else len(_roi_exist) > 0:
                rrs_neurons = mod['Fluorescence'].get_roi_response_series[_roi_exist[0]]
                self.roi_response = rrs_neurons.data
                self.roi_response_bk = np.nan * np.ones(
                                        [self._no_background_comps, self.roi_response.shape[1]])
                if len(_roi_exist) > 1:
                    rrs_bk = mod['Fluorescence'].get_roi_response_series[_roi_exist[1]
                    self.roi_response_bk = rrs_bk.data


            # Extract planesegmentation dictionary values:
            _new_columns = [i for i in ps.colnames if i not in ['image_mask', 'pixel_mask']]
            for i in _new_columns:
                setattr(self, i, ps[i].data)

            # Extract samp_freq:
            self.samp_freq = rrs_neurons.rate
            self.total_time = rrs_neurons.rate * rrs_neurons.num_samples
            # Extract no_rois/ids:
            self.roi_idx = np.array(rrs_neurons.rois.data)
            self.no_rois = self.roi_idx.size


    def get_traces(self, ROI_ids=None, start_frame=None, end_frame=None):

    def get_num_frames(self):

    def get_sampling_frequency(self):

    def get_roi_locations(self):

    def get_roi_ids(self):

    def get_num_rois(self):

    def get_pixel_masks(self, ROI_ids=None):

    def get_image_masks(self, ROI_ids=None):

    def get_movie_framesize(self):

    def get_movie_location(self):

    def get_channel_names(self):

    def get_num_channels(self):

    @staticmethod
    def write_nwb(segmentation_extractor_obj, filename, propertydict=[], identifier=None,
                  starting_time=0., session_start_time=datetime.now(tzlocal()), excitation_lambda=np.nan,
                  emission_lambda=np.nan, indicator='none', location='brain', device_name='MyDevice',
                  optical_channel_name='MyOpticalChannel', optical_channel_description='MyOpticalChannelDescription',
                  imaging_plane_name='MyImagingPlane', imaging_plane_description='MyImagingPlaneDescription',
                  image_series_name='MyTwoPhotonSeries', image_series_description='MyTwoPhotonSeriesDescription',
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
        imaging_rate = segmentation_extractor_obj.get_sampling_frequency()
        raw_movie_file_location = segmentation_extractor_obj.get_movie_location()

        if identifier is None:
            identifier = uuid.uuid1().hex

        if '.nwb' != os.path.splitext(filename)[-1].lower():
            raise Exception("Wrong filename")

        if os.path.exists(filename):
            read_mode = 'r+'
            _nwbfile_exist = True
        else:
            _nwbfile_exist = False
            read_mode = 'w'

        with NWBHDF5IO(filename, mode=read_mode) as io:
            if _nwbfile_exist:
                nwbfile = io.read()
                _nwbchildren_type = [type(i).__name__ for i in nwbfile.all_children()]
                _nwbchildren_name = [i.name for i in nwbfile.all_children()]
                _custom_names = [device_name, optical_channel_name, imaging_plane_name,
                                 image_series_name]
                # ADDING ACQUISITION DATA: device, optical channel, imaging plane, image series

                # check existence of device:
                _device_exist = [i for i, e in enumerate(_nwbchildren_name) if e == device_name]
                if not _device_exist:
                    nwbfile.add_device(Device(device_name))
                elif len(_device_exist > 1):
                    raise Exception('Multiple Devices exist, provide name of one')
                device = list(nwbfile.devices.values())[0]

                # check existence of optical channel:
                _optical_channel_exist = [i for i, e in enumerate(
                    _nwbchildren_name) if e == optical_channel_name]
                if not _optical_channel_exist:
                    optical_channel = OpticalChannel(optical_channel_name,
                                                     optical_channel_description,
                                                     emission_lambda=emission_lambda)
                elif len(_optical_channel_exist) == 1:
                    optical_channel = nwbfile.all_children()[_optical_channel_exist[0]]
                else:
                    raise Exception('Multiple Optical Channels exist, provide name of one')

                # check existence of imaging plane:
                _imaging_plane_exist = [i for i, e in enumerate(
                    _nwbchildren_name) if e == imaging_plane_name]
                if not _imaging_plane_exist:
                    nwbfile.create_imaging_plane(name=imaging_plane_name,
                                                 optical_channel=optical_channel,
                                                 description=imaging_plane_description,
                                                 device=device,
                                                 excitation_lambda=excitation_lambda,
                                                 imaging_rate=imaging_rate,
                                                 indicator=indicator,
                                                 location=location)
                elif len(_imaging_plane_exist) == 1:
                    imaging_plane = nwbfile.all_children()[_imaging_plane_exist[0]]
                else:
                    raise Exception('Multiple Imaging Planes exist, provide name of one')

                # check existence of image series:
                _image_series_exist = [i for i, e in enumerate(
                    _nwbchildren_name) if e == image_series_name]
                if not _image_series_exist:
                    nwbfile.add_acquisition(TwoPhotonSeries(name=image_series_name,
                                                            description=image_series_description,
                                                            external_file=[raw_movie_file_location],
                                                            format='external',
                                                            rate=imaging_rate,
                                                            starting_frame=[0]))
                elif len(_image_series_exist) == 1:
                    image_series = nwbfile.all_children()[_image_series_exist[0]]
                else:
                    raise Exception('Multiple Imaging Series exist, provide name of one')

                # ADDING PROCESSING DATA:
                # Adding Ophys
                _ophys_module_exist = [i for i, e in enumerate(
                    _nwbchildren_name) if e == processing_module_name]
                if not _ophys_module_exist:
                    mod = nwbfile.create_processing_module(
                        processing_module_name, processing_module_description)
                elif len(_ophys_module_exist) == 1:
                    mod = nwbfile.all_children()[_ophys_module_exist[0]]
                else:
                    raise Exception('Multiple Ophys modules exist, provide name of one')

                # Adding ImageSegmentation:
                _image_segmentation_exist = [i for i, e in enumerate(
                    _nwbchildren_name) if e == 'ImageSegmentation']
                if not _image_segmentation_exist:
                    img_seg = ImageSegmentation()
                    mod.add_data_interface(img_seg)
                else:
                    img_seg = nwbfile.all_children()[_image_segmentation_exist[0]]

                # Adding Fluorescence:
                _fluorescence_module_exist = [i for i, e in enumerate(
                    _nwbchildren_name) if e == 'Fluorescence']
                if not _fluorescence_module_exist:
                    fl = Fluorescence()
                    mod.add_data_interface(fl)
                else:
                    fl = nwbfile.all_children()[_fluorescence_module_exist[0]]

                # Adding PlaneSegmentation:
                _plane_segmentation_exist = [i for i, e in enumerate(
                    _nwbchildren_name) if e == 'PlaneSegmentation']
                if not _plane_segmentation_exist:
                    ps = img_seg.create_plane_segmentation(
                        'ROIs', imaging_plane, 'PlaneSegmentation', image_series)
                else:
                    ps = nwbfile.all_children()[_plane_segmentation_exist[0]]

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
                imaging_plane = nwbfile.create_imaging_plane(name=imaging_plane_name,
                                                             optical_channel=optical_channel,
                                                             description=imaging_plane_description,
                                                             device=device,
                                                             excitation_lambda=excitation_lambda,
                                                             imaging_rate=imaging_rate,
                                                             indicator=indicator,
                                                             location=location)
                image_series = TwoPhotonSeries(name=image_series_name,
                                               description=image_series_description,
                                               external_file=[raw_movie_file_location],
                                               format='external',
                                               rate=imaging_rate,
                                               starting_frame=[0])
                nwbfile.add_acquisition(image_series)

                mod = nwbfile.create_processing_module(processing_module_name,
                                                       processing_module_description)

                img_seg = ImageSegmentation()
                mod.add_data_interface(img_seg)
                fl = Fluorescence()
                mod.add_data_interface(fl)
                ps = img_seg.create_plane_segmentation(
                    'ROIs', imaging_plane, 'PlaneSegmentation', image_series)

            # Adding the ROI-related stuff:
            # Adding custom columns and their values to the PlaneSegmentation table:
            # propertydict is a list of dictionaries containing:
            # [{'name':'','discription':'', 'data':'', id':''}, {}..]
            if len(propertydict):
                _segmentation_exctractor_attrs = dir(segmentation_extractor_obj)
                for i in range(len(propertydict)):
                    excep_str = 'enter argument propertydict as list of dictionaries:\n'\
                                '[{\'name\':str(required), '\
                                '\'description\':str(optional), '\
                                '\'data\':array_like(required), '\
                                '\'id\':int(optional)}, {..}]'
                    if propertydict[i].get('name', None):
                        raise Exception(excep_str)
                    _property_name = propertydict[i].['name']
                    if propertydict[i].get('data', None):
                        raise Exception(excep_str)
                    _property_values = propertydict[i].['data']
                    _property_desc = propertydict[i].get('description', 'no description')
                    _property_row_ids = propertydict[i].get('id', list(
                        np.arange(segmentation_extractor_obj.no_rois)))
                    # stop execution if the property name is non existant OR there are multiple such properties:
                    _property_name_exist = [i for i in _segmentation_exctractor_attrs if len(
                        re.findall('^' + _property_name, i, re.I))]
                    if len(_property_name_exist) == 1:
                        print(f'adding {_property_name_exist} with supplied data')
                    elif len(_property_name_exist) == 0:
                        print(f'creating table for {_property_name_exist} with supplied data')
                    else:
                        raise Exception('multiple variables found for supplied name\n enter'
                                        f' one of {_property_name_exist}')
                    set_dynamic_table_property(ps, _property_row_ids, _property_name, _property_values,
                                               index=False, description=_property_desc)

            # Adding Image and Pixel Masks(default colnames in PlaneSegmentation):
            for i, roiid in enumerate(segmentation_extractor_obj.roi_idx):
                img_roi = segmentation_extractor_obj.image_masks[:, i]
                pix_roi = segmentation_extractor_obj.pixel_masks[segmentation_extractor_obj.pixel_masks[:, 3] == roiid, :]
                ps.add_roi(image_mask=img_roi.reshape(segmentation_extractor_obj.image_dims, order='F'),
                           pixel_mask=pix_roi)

            # Background components addition:
            if hasattr(segmentation_extractor_obj, 'image_masks_bk'):
                if hasattr(segmentation_extractor_obj, 'pixel_masks_bk'):
                    for i, (img_roi_bk, pix_roi_bk) in enumerate(zip(segmentation_extractor_obj.image_masks_bk.T, segmentation_extractor_obj.pixel_masks_bk)):
                        ps.add_roi(image_mask=img_roi_bk.reshape(segmentation_extractor_obj.image_dims, order='F'),
                                   pixel_mask=pix_roi_bk)
                else:
                    for i, img_roi_bk in enumerate(segmentation_extractor_obj.image_masks_bk.T):
                        ps.add_roi(image_mask=img_roi_bk.reshape(
                            segmentation_extractor_obj.image_dims, order='F'))

            # Add Traces
            no_neurons_rois = segmentation_extractor_obj.image_masks.shape[-1]
            no_background_rois = len(segmentation_extractor_obj.roi_response_bk)
            neuron_rois = ps.create_roi_table_region(
                'NeuronROIs', region=list(range(no_neurons_rois)))

            background_rois = ps.create_roi_table_region(
                'BackgroundROIs', region=list(range(no_neurons_rois, no_neurons_rois + no_background_rois)))

            timestamps = np.arange(
                segmentation_extractor_obj.roi_response_bk.shape[1]) / imaging_rate + starting_time

            # Neurons TimeSeries
            neuron_roi_response_exist = [i for i, e in enumerate(
                _nwbchildren_name) if e == neuron_roi_response_series_name]
            if not neuron_roi_response_exist:
                fl.create_roi_response_series(name=neuron_roi_response_series_name, data=segmentation_extractor_obj.roi_response.T,
                                              rois=neuron_rois, unit='lumens'
                                              starting_time=starting_time, rate=imaging_rate)
            else:
                raise Exception('Time Series for' + neuron_roi_response_series_name +
                                ' already exists , provide another name')

            # Background TimeSeries
            background_roi_response_exist = [i for i, e in enumerate(
                _nwbchildren_name) if e == background_roi_response_series_name]
            if not background_roi_response_exist:
                fl.create_roi_response_series(name=background_roi_response_series_name, data=segmentation_extractor_obj.roi_response.T,
                                              rois=background_rois, unit='lumens',
                                              starting_time=starting_time, rate=imaging_rate)
            else:
                raise Exception('Time Series for' + background_roi_response_series_name +
                                ' already exists , provide another name')

            # Residual TimeSeries
            mod.add(TimeSeries(name='residuals', description='residuals', data=segmentation_extractor_obj.roi_response_residual.T, timestamps=timestamps,
                               unit='NA'))
            if hasattr(segmentation_extractor_obj, 'cn'):
                images = Images('summary_images')
                images.add_image(GrayscaleImage(name='local_correlations',
                                                data=segmentation_extractor_obj.cn))

                # Add MotionCorreciton
        #            create_corrected_image_stack(corrected, original, xy_translation, name='CorrectedImageStack')
            io.write(nwbfile)
