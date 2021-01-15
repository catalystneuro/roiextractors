from ...imagingextractor import ImagingExtractor
from ...extraction_tools import check_keys
from ...extraction_tools import PathType, FloatType, ArrayType
import numpy as np
from pathlib import Path
import os

try:
    import scippy.io as spio

    HAVE_Scipy = True
except ImportError:
    HAVE_Scipy = False


class SgxImagingExtractor(ImagingExtractor):
    extractor_name = 'SgxImaging'
    installed = HAVE_Scipy  # check at class level if installed or not
    is_writable = True
    mode = 'folder'
    installation_mesg = "To use the Sgx Extractor run:\n\n pip install scipy\n\n"  # error message when not installed

    def __init__(self, file_path: PathType, k, N):
        self.file_name = file_path

    def loadmat(self):
        """
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects.
        Mimics implementation @ https://github.com/GiocomoLab/TwoPUtils/blob/main/scanner_tools/sbx_utils.py
        """
        data = spio.loadmat(self.file_name, struct_as_record=False, squeeze_me=True)
        info = check_keys(data)['info']
        # Defining number of channels/size factor
        if info['channels'] == 1:
            info['nChan'] = 2
            factor = 1
        elif info['channels'] == 2:
            info['nChan'] = 1
            factor = 2
        elif info['channels'] == 3:
            info['nChan'] = 1
            factor = 2
        else:
            raise UserWarning("wrong 'channels' argument")

        if info['scanmode'] == 0:
            info['recordsPerBuffer'] *= 2

        if 'fold_lines' in info.keys():
            if info['fold_lines'] > 0:
                info['fov_repeats'] = int(info['config']['lines']/info['fold_lines'])
            else:
                info['fov_repeats'] = 1
        else:
            info['fold_lines'] = 0
            info['fov_repeats'] = 1
        # Determine number of frames in whole file
        # info['max_idx'] = int(
        #     os.path.getsize(self.file_name[:-4] + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 / (
        #                 2 - info['scanmode']) - 1)
        info['max_idx'] = int(
            os.path.getsize(self.file_name[:-4] + '.sbx')/info['recordsPerBuffer']/info['sz'][1]*factor/4 - 1)*int(
            info['fov_repeats'])
        # info['max_idx']=info['frame'][-1]

        info['frame_rate'] = info['resfreq']/info['config']['lines']*(2 - info['scanmode'])*info['fov_repeats']

        return info

    def sbxread(self, k=0, N=None):
        """
        Input: self.file_name should be full path excluding .sbx, starting index, batch size
        By default Loads whole file at once, make sure you have enough ram available to do this
        """
        # Check if contains .sbx and if so just truncate
        if '.sbx' in self.file_name:
            self.file_name = self.file_name[:-4]

        # Load info
        info = self.loadmat()
        # print info.keys()

        # Paramters
        # k = 0; #First frame
        max_idx = info['max_idx']
        if N is None:
            N = max_idx  # Last frame
        else:
            N = min([N, max_idx - k])

        nSamples = info['sz'][1]*info['recordsPerBuffer']/info['fov_repeats']*2*info['nChan']
        # print(nSamples, N)

        # Open File
        fo = open(self.file_name + '.sbx')

        # print(int(k) * int(nSamples))
        fo.seek(int(k)*int(nSamples), 0)
        x = np.fromfile(fo, dtype='uint16', count=int(nSamples/2*N))
        x = np.int16((np.int32(65535) - x).astype(np.int32)/np.int32(2))
        x = x.reshape((info['nChan'], info['sz'][1], int(info['recordsPerBuffer']/info['fov_repeats']), int(N)),
                      order='F')

        return x

    def scanbox_imaging_parameters(filepath):
        """Parse imaging parameters from Scanbox.
        Based off of the sbxRead Matlab implementation and
        https://scanbox.org/2016/09/02/reading-scanbox-files-in-python/
        """
        assert filepath.endswith('.mat')
        data_path = os.path.splitext(filepath)[0] + '.sbx'
        data = spio.loadmat(filepath)['info']
        info = check_keys(data)
        # Fix for old scanbox versions
        if 'sz' not in info:
            info['sz'] = np.array([512, 796])

        if 'scanmode' not in info:
            info['scanmode'] = 1  # unidirectional
        elif info['scanmode'] == 0:
            info['recordsPerBuffer'] *= 2  # bidirectional

        if info['channels'] == 1:
            # both PMT 0 and 1
            info['nchannels'] = 2
            # factor = 1
        elif info['channels'] == 2 or info['channels'] == 3:
            # PMT 0 or 1
            info['nchannels'] = 1
            # factor = 2

        # Bytes per frame (X * Y * C * bytes_per_pixel)
        info['nsamples'] = info['sz'][1]*info['recordsPerBuffer']* \
                           info['nchannels']*2

        # Divide 'max_idx' by the number of plane to get the number of time steps
        if info.get('scanbox_version', -1) >= 2:
            # last_idx = total_bytes / (Y * X * 4 / factor) - 1
            # info['max_idx'] = os.path.getsize(data_path) // \
            #     info['recordsPerBuffer'] // info['sz'][1] * factor // 4 - 1
            info['max_idx'] = os.path.getsize(data_path)//info['nsamples'] - 1
        else:
            if info['nchannels'] == 1:
                factor = 2
            elif info['nchannels'] == 2:
                factor = 1
            info['max_idx'] = os.path.getsize(data_path) \
                              //info['bytesPerBuffer']*factor - 1

        # Check optotune planes
        if ('volscan' in info and info['volscan'] > 0) or \
                ('volscan' not in info and len(info.get('otwave', []))):
            info['nplanes'] = len(info['otwave'])
        else:
            info['nplanes'] = 1

        return info

