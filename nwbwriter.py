from pynwb import NWBHDF5IO, TimeSeries, NWBFile
from pynwb.base import Images
from pynwb.image import GrayscaleImage
from pynwb.ophys import ImageSegmentation, Fluorescence, OpticalChannel, ImageSeries
from pynwb.device import Device
import os
from datetime import datetime
from dateutil.tz import tzlocal
import time,logging
import numpy as np
def save_NWB(CIDataContainerObj,filename,imaging_plane_name=None,imaging_series_name=None,sess_desc='TraceExtractor Results',exp_desc=None,identifier=None,
             starting_time=0.,session_start_time=datetime.now(tzlocal()),excitation_lambda=488.0,imaging_plane_description='some imaging plane description',emission_lambda=520.0,indicator='OGB-1',
             location='brain'):
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
        imaging_rate=CIDataContainerObj.SampFreq
        raw_data_file=CIDataContainerObj.raw_data_file

        if identifier is None:
            import uuid
            identifier = uuid.uuid1().hex

        if '.nwb' != os.path.splitext(filename)[-1].lower():
            raise Exception("Wrong filename")

        if not os.path.isfile(filename):  # if the file doesn't exist create new and add the original data path
            print('filename does not exist. Creating new NWB file with only estimates output')

            nwbfile = NWBFile(sess_desc, identifier, session_start_time, experiment_description=exp_desc)
            device = Device('imaging_device')
            nwbfile.add_device(device)
            optical_channel = OpticalChannel('OpticalChannel',
                                             'main optical channel',
                                             emission_lambda=emission_lambda)
            nwbfile.create_imaging_plane(name='ImagingPlane',
                                         optical_channel=optical_channel,
                                         description=imaging_plane_description,
                                         device=device,
                                         excitation_lambda=excitation_lambda,
                                         imaging_rate=imaging_rate,
                                         indicator=indicator,
                                         location=location)
            if raw_data_file:
                nwbfile.add_acquisition(ImageSeries(name='TwoPhotonSeries',
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
            # Add processing results

            # Create the module as 'ophys' unless it is taken and append 'ophysX' instead
            ophysmodules = [s[5:] for s in list(nwbfile.modules) if s.startswith('ophys')]
            if any('' in s for s in ophysmodules):
                if any([s for s in ophysmodules if s.isdigit()]):
                    nummodules = max([int(s) for s in ophysmodules if s.isdigit()])+1
                    print('ophys module previously created, writing to ophys'+str(nummodules)+' instead')
                    mod = nwbfile.create_processing_module('ophys'+str(nummodules), 'contains caiman estimates for '
                                                                                    'the main imaging plane')
                else:
                    print('ophys module previously created, writing to ophys1 instead')
                    mod = nwbfile.create_processing_module('ophys1', 'contains caiman estimates for the main '
                                                                     'imaging plane')
            else:
                mod = nwbfile.create_processing_module('ophys', 'contains caiman estimates for the main imaging plane')

            img_seg = ImageSegmentation()
            mod.add_data_interface(img_seg)
            fl = Fluorescence()
            mod.add_data_interface(fl)
#            mot_crct = MotionCorrection()
#            mod.add_data_interface(mot_crct)

            # Add the ROI-related stuff
            if imaging_plane_name is not None:
                imaging_plane = nwbfile.imaging_planes[imaging_plane_name]
            else:
                if len(nwbfile.imaging_planes) == 1:
                    imaging_plane = list(nwbfile.imaging_planes.values())[0]
                else:
                    raise Exception('There is more than one imaging plane in the file, you need to specify the name'
                                    ' via the "imaging_plane_name" parameter')

            if imaging_series_name is not None:
                image_series = nwbfile.acquisition[imaging_series_name]
            else:
                if not len(nwbfile.acquisition):
                    image_series = None
                elif len(nwbfile.acquisition) == 1:
                    image_series = list(nwbfile.acquisition.values())[0]
                else:
                    raise Exception('There is more than one imaging plane in the file, you need to specify the name'
                                    ' via the "imaging_series_name" parameter')

            ps = img_seg.create_plane_segmentation('CNMF_ROIs', imaging_plane, 'PlaneSegmentation', image_series)

            ps.add_column('r', 'description of r values')
            ps.add_column('snr', 'signal to noise ratio')
            ps.add_column('cnn', 'description of CNN')
            ps.add_column('keep', 'in idx_components')
            ps.add_column('accepted', 'in accepted list')
            ps.add_column('rejected', 'in rejected list')

            # Add ROIs
            if not hasattr(CIDataContainerObj, 'accepted_list'):
                for i, (roi, snr, r, cnn) in enumerate(zip(CIDataContainerObj.A.T, CIDataContainerObj.SNR_comp, CIDataContainerObj.r_values, CIDataContainerObj.cnn_preds)):
                    ps.add_roi(image_mask=roi.T.reshape(CIDataContainerObj.dims), r=r, snr=snr, cnn=cnn,
                               keep=i in CIDataContainerObj.idx_components, accepted=False, rejected=False)
            else:
                for i, (roi, snr, r, cnn) in enumerate(zip(CIDataContainerObj.A.T, CIDataContainerObj.SNR_comp, CIDataContainerObj.r_values, CIDataContainerObj.cnn_preds)):
                    ps.add_roi(image_mask=roi.T.reshape(CIDataContainerObj.dims), r=r, snr=snr, cnn=cnn,
                               keep=i in CIDataContainerObj.idx_components, accepted=i in CIDataContainerObj.accepted_list, rejected=i in CIDataContainerObj.rejected_list)

            for bg in CIDataContainerObj.b.T:  # Backgrounds
                ps.add_roi(image_mask=bg.reshape(CIDataContainerObj.dims), r=np.nan, snr=np.nan, cnn=np.nan, keep=False, accepted=False, rejected=False)
            # Add Traces
            n_rois = CIDataContainerObj.A.shape[-1]
            n_bg = len(CIDataContainerObj.f)
            rt_region_roi = ps.create_roi_table_region(
                'ROIs', region=list(range(n_rois)))

            rt_region_bg = ps.create_roi_table_region(
                'Background', region=list(range(n_rois, n_rois+n_bg)))

            timestamps = np.arange(CIDataContainerObj.f.shape[1]) / imaging_rate + starting_time

            # Neurons
            fl.create_roi_response_series(name='RoiResponseSeries', data=CIDataContainerObj.C.T, rois=rt_region_roi, unit='lumens', timestamps=timestamps)
            # Background
            fl.create_roi_response_series(name='Background_Fluorescence_Response', data=CIDataContainerObj.f.T, rois=rt_region_bg, unit='lumens',
                                          timestamps=timestamps)

            mod.add(TimeSeries(name='residuals', description='residuals', data=CIDataContainerObj.YrA.T, timestamps=timestamps,
                               unit='NA'))
            if hasattr(CIDataContainerObj, 'Cn'):
                images = Images('summary_images')
                images.add_image(GrayscaleImage(name='local_correlations', data=CIDataContainerObj.Cn))

                # Add MotionCorreciton
    #            create_corrected_image_stack(corrected, original, xy_translation, name='CorrectedImageStack')
                io.write(nwbfile)
