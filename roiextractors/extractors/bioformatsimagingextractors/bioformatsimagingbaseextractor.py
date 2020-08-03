from pathlib import Path
import numpy as np
from roiextractors import ImagingExtractor

try:
    import javabridge
    import bioformats

    HAVE_BIOFORMATS = True
except:
    HAVE_BIOFORMATS = False


class BioformatsImagingExtractor(ImagingExtractor):
    def __init__(self, file_path):
        ImagingExtractor.__init__(self)
        self.file_path = Path(file_path)
        self._start_javabridge_vm()
        self._reader = bioformats.ImageReader(str(self.file_path))
        self._read_metadata()

    def __del__(self):
        self._reader.close()
        self._kill_javabridge_vm()

    # TODO: check that other formats have the same structure
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
            else:
                if 'StructuredAnnotations' in ch.tag:
                    for ch1 in ch:
                        tag1 = ch1.tag[ch1.tag.find('}') + 1:]
                        attrib1 = ch1.attrib
                        for ch2 in ch1:
                            tag2 = ch2.tag[ch2.tag.find('}') + 1:]
                        # todo get sampling frequency

        self.metadata = {}
        self.metadata['num_channels'] = int(pixels_info['SizeC'])
        self.metadata['num_frames'] = int(pixels_info['SizeT'])
        self.metadata['size_x'] = int(pixels_info['SizeX'])
        self.metadata['size_y'] = int(pixels_info['SizeY'])
        self.metadata['size_z'] = int(pixels_info['SizeZ'])
        self.metadata['dtype'] = pixels_info['Type']

        self._pixelinfo = pixels_info

        # TODO retrieve channel names
        self.metadata['channel_names'] = [f'channel_{i}' for i in range(self.metadata['num_channels'])]

        # TODO find better approach
        get_next = False
        for it in self._root.iter():
            if it.text is not None:
                if get_next:
                    self._sampling_frequency = float(it.text)
                    get_next = False
                if 'frame' in it.text:
                    get_next = True

    def _start_javabridge_vm(self):
        javabridge.start_vm(class_path=bioformats.JARS)

    def _kill_javabridge_vm(self):
        javabridge.kill_vm()

    def get_frame(self, frame_idx, channel=0):
        assert frame_idx < self.get_num_frames()
        plane = self._reader.read(t=frame_idx).T
        return plane

    def get_frames(self, frame_idxs, channel=0):
        assert np.all(np.array(frame_idxs) < self.get_num_frames())
        planes = np.zeros((len(frame_idxs), self.metadata['size_x'], self.metadata['size_y']))
        for i, frame_idx in enumerate(frame_idxs):
            plane = self._reader.read(t=frame_idx).T
            planes[i] = plane
        return planes

    def get_video(self, start_frame=None, end_frame=None, channel=0):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        end_frame = min(end_frame, self.get_num_frames())

        video = np.zeros((end_frame - start_frame, self.metadata['size_x'], self.metadata['size_y']))
        for i, frame_idx in enumerate(np.arange(start_frame, end_frame)):
            video[i] = self._reader.read(t=frame_idx).T

        return video

    def get_image_size(self):
        return np.array([self.metadata['size_x'], self.metadata['size_y']])

    def get_num_frames(self):
        return self.metadata['num_frames']

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_channel_names(self):
        return self.metadata['channel_names']

    def get_num_channels(self):
        return self.metadata['num_channels']
