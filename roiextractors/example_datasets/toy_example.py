import numpy as np
import spikeextractors as se

from ..extractors.numpyextractors import NumpyImagingExtractor, NumpySegmentationExtractor


def _gaussian(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi * sigma) * np.exp(- (x - mu) ** 2 / sigma)


def _generate_rois(num_units=10, size_x=100, size_y=100, roi_size=4, min_dist=5, mode='uniform'):
    image = np.zeros((size_x, size_y))
    max_iter = 1000

    count = 0
    it = 0
    means = []

    while count < num_units:
        mean_x = np.random.randint(0, size_x - 1)
        mean_y = np.random.randint(0, size_y - 1)

        mean_ = np.array([mean_x, mean_y])

        if len(means) == 0:
            means.append(mean_)
            count += 1
        else:
            dists = np.array([np.linalg.norm(mean_ - m) for m in means])

            if np.all(dists > min_dist):
                means.append(mean_)
                count += 1

        it += 1

        if it >= max_iter:
            raise Exception("Could not fit ROIs given 'min_dist'")

    roi_pixels = []

    for m, mean in enumerate(means):
        # print(f"ROI {m + 1}/{num_units}")
        pixels = []
        for i in np.arange(size_x):
            for j in np.arange(size_y):
                p = np.array([i, j])

                if np.linalg.norm(p - mean) < roi_size:
                    pixels.append(p)
                    if mode == 'uniform':
                        image[i, j] = 1
                    elif mode == 'gaussian':
                        image[i, j] = _gaussian(i, mean[0], roi_size) + _gaussian(j, mean[1], roi_size)
                    else:
                        raise Exception("'mode' can be 'uniform' or 'gaussian'")
        roi_pixels.append(np.array(pixels))

    return roi_pixels, image, means


def toy_example(duration=10, num_rois=10, size_x=100, size_y=100, roi_size=4, min_dist=5, mode='uniform',
                sampling_frequency=30, decay_time=0.5, noise_std=0.05):
    """
    Create a toy example of an ImagingExtractor and a SegmentationExtractor.

    Parameters
    ----------
    duration: float
        Duration in s
    num_rois: int
        Number of ROIs
    size_x: int
        Size of x dimension (pixels)
    size_y: int
        Size of y dimension (pixels)
    roi_size: int
        Siz of ROI in x and y dimension (pixels)
    min_dist: int
        Minimum distance between ROI centers (pixels)
    mode: str
        'uniform' or 'gaussian'.
        If 'uniform', ROI values are uniform and equal to 1.
        If 'gaussian', ROI values are gaussian modulated
    sampling_frequency: float
        The sampling rate
    decay_time: float
        Decay time of fluorescence reponse
    noise_std: float
        Standard deviation of added gaussian noise

    Returns
    -------
    imag: NumpyImagingExtractor
        The output imaging extractor
    seg: NumpySegmentationExtractor
        The output segmentation extractor

    """
    # generate ROIs
    num_rois = int(num_rois)
    roi_pixels, im, means = _generate_rois(num_units=num_rois, size_x=size_x, size_y=size_y,
                                           roi_size=roi_size, min_dist=min_dist, mode=mode)

    # generate spike trains
    rec, sort = se.example_datasets.toy_example(duration=duration, K=num_rois, num_channels=1,
                                                sampling_frequency=sampling_frequency)

    # create decaying response
    resp_samples = int(decay_time * rec.get_sampling_frequency())
    resp_tau = resp_samples / 5
    tresp = np.arange(resp_samples)
    resp = np.exp(-tresp / resp_tau)

    # convolve response with ROIs
    raw = np.zeros((len(sort.get_unit_ids()), rec.get_num_frames()))
    deconvolved = np.zeros((len(sort.get_unit_ids()), rec.get_num_frames()))
    neuropil = noise_std * np.random.randn(len(sort.get_unit_ids()), rec.get_num_frames())
    frames = rec.get_num_frames()
    for u_i, unit in enumerate(sort.get_unit_ids()):
        for s in sort.get_unit_spike_train(unit):
            if s < rec.get_num_frames():
                if s + len(resp) < frames:
                    raw[u_i, s:s + len(resp)] += resp
                else:
                    raw[u_i, s:] = resp[:frames - s]
                deconvolved[u_i, s] = 1

    # generate video
    video = np.zeros((frames, size_x, size_y))
    for (rp, t) in zip(roi_pixels, raw):
        for r in rp:
            video[:, r[0], r[1]] += t * im[r[0], r[1]]

    # normalize video
    video /= np.max(video)

    # add noise
    video += noise_std * np.abs(np.random.randn(*video.shape))

    # instantiate imaging and segmentation extractors
    imag = NumpyImagingExtractor(timeseries=video, sampling_frequency=30)

    # create image masks
    image_masks = np.zeros((size_x, size_y, num_rois))
    for rois_i, roi in enumerate(roi_pixels):
        for r in roi:
            image_masks[r[0], r[1], rois_i] += im[r[0], r[1]]

    seg = NumpySegmentationExtractor(image_masks=image_masks, raw=raw, deconvolved=deconvolved,
                                     neuropil=neuropil, sampling_frequency=sampling_frequency)

    return imag, seg
