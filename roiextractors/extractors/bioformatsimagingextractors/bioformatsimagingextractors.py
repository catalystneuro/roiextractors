from pathlib import Path
from .bioformatsimagingbaseextractor import BioformatsImagingExtractor

try:
    import javabridge
    import bioformats

    HAVE_BIOFORMATS = True
except:
    HAVE_BIOFORMATS = False


class FliImagingExtractor(BioformatsImagingExtractor):
    extractor_name = 'FliImaging'
    installed = HAVE_BIOFORMATS  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = "To use the FliImagingExtractor install python-bioformats: \n\n pip install python-bioformats\n\n"  # error message when not installed

    def __init__(self, file_path):
        assert HAVE_BIOFORMATS
        BioformatsImagingExtractor.__init__(self, file_path)

        self._kwargs = {'file_path': str(Path(file_path).absolute())}


    def _read_metadata(self):
        ## Read metadata from ONEXML
        metadata = bioformats.get_omexml_metadata(str(self.file_path))
        o = bioformats.OMEXML(metadata)
        dom = o.dom
        self._root = dom.getroot()

        for ch in self._root:
            tag = ch.tag[ch.tag.find('}') + 1:]
            if 'Image' in tag:
                attrib = ch.attrib
                if 'Primary' in attrib['Name']:
                    for ch1 in ch:
                        tag1 = ch1.tag[ch1.tag.find('}') + 1:]
                        attrib1 = ch1.attrib
                        if 'Pixels' in tag1:
                            pixels_info = attrib1

        self._num_channels = int(pixels_info['SizeC'])
        self._num_frames = int(pixels_info['SizeT'])
        self._size_x = int(pixels_info['SizeX'])
        self._size_y = int(pixels_info['SizeY'])
        self._size_z = int(pixels_info['SizeZ'])
        self._dtype = pixels_info['Type']
        self._pixelinfo = pixels_info

        # TODO retrieve channel names
        self._channel_names = [f'channel_{i}' for i in range(self._num_channels)]

        # TODO find better approach
        get_next = False
        for it in self._root.iter():
            if it.text is not None:
                if get_next:
                    self._sampling_frequency = float(it.text)
                    get_next = False
                if 'frame' in it.text:
                    get_next = True


class StkImagingExtractor(BioformatsImagingExtractor):
    extractor_name = 'StkImaging'
    installed = HAVE_BIOFORMATS  # check at class level if installed or not
    is_writable = False
    mode = 'file'
    installation_mesg = "To use the StkImagingExtractor install python-bioformats: \n\n pip install python-bioformats\n\n"  # error message when not installed

    def __init__(self, file_path, sampling_frequency):
        assert HAVE_BIOFORMATS
        self._sampling_frequency = sampling_frequency
        BioformatsImagingExtractor.__init__(self, file_path)

        self._kwargs = {'file_path': str(Path(file_path).absolute()),
                        'sampling_frequency': sampling_frequency}

    def _read_metadata(self):
        ## Read metadata from ONEXML
        metadata = bioformats.get_omexml_metadata(str(self.file_path))
        o = bioformats.OMEXML(metadata)
        dom = o.dom
        self._root = dom.getroot()

        for ch in self._root:
            tag = ch.tag[ch.tag.find('}') + 1:]
            if 'Image' in tag:
                for ch1 in ch:
                    tag1 = ch1.tag[ch1.tag.find('}') + 1:]
                    attrib1 = ch1.attrib
                    if 'Pixels' in tag1:
                        pixels_info = attrib1

        self._num_channels = int(pixels_info['SizeC'])
        self._num_frames = int(pixels_info['SizeT'])
        self._size_x = int(pixels_info['SizeX'])
        self._size_y = int(pixels_info['SizeY'])
        self._size_z = int(pixels_info['SizeZ'])
        self._dtype = pixels_info['Type']
        self._pixelinfo = pixels_info

        # TODO retrieve channel names
        self._channel_names = [f'channel_{i}' for i in range(self._num_channels)]

        # TODO find better approach
        get_next = False
        for it in self._root.iter():
            if it.text is not None:
                if get_next:
                    self._sampling_frequency = float(it.text)
                    get_next = False
                if 'frame' in it.text:
                    get_next = True
