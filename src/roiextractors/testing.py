"""Testing utilities for the roiextractors package."""

import warnings
from collections.abc import Iterable
from typing import Literal

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from numpy.typing import DTypeLike

from roiextractors import GaussianNoiseImagingExtractor, NumpySegmentationExtractor

from .imagingextractor import ImagingExtractor
from .segmentationextractor import SegmentationExtractor

NoneType = type(None)
floattype = (float, np.floating)
inttype = (int, np.integer)


def generate_dummy_video(
    size: tuple[int, int, int] | tuple[int, int, int, int], dtype: DTypeLike = "uint16", seed: int = 0
):
    """Generate a dummy video of a given size and dtype.

    .. deprecated::
        ``generate_dummy_video`` is deprecated and will be removed in or after September 2026.
        Use ``GaussianNoiseImagingExtractor`` or ``PoissonNoiseImagingExtractor`` instead.

    Parameters
    ----------
    size : tuple[int, int, int] or tuple[int, int, int, int]
        Size of the video to generate.
        For planar data: (num_frames, num_rows, num_columns)
        For volumetric data: (num_frames, num_rows, num_columns, num_planes)
    dtype : DTypeLike, optional
        Dtype of the video to generate, by default "uint16".
    seed : int, default 0
        seed for the random number generator, by default 0.

    Returns
    -------
    video : np.ndarray
        A dummy video of the given size and dtype.
    """
    warnings.warn(
        "generate_dummy_video is deprecated and will be removed in or after September 2026. "
        "Use GaussianNoiseImagingExtractor or PoissonNoiseImagingExtractor instead.",
        FutureWarning,
        stacklevel=2,
    )
    dtype = np.dtype(dtype)
    number_of_bytes = dtype.itemsize

    rng = np.random.default_rng(seed)

    low = 0 if "u" in dtype.name else 2 ** (number_of_bytes - 1) - 2**number_of_bytes
    high = 2**number_of_bytes - 1 if "u" in dtype.name else 2**number_of_bytes - 2 ** (number_of_bytes - 1) - 1
    video = rng.random(size=size) if "float" in dtype.name else rng.integers(low=low, high=high, size=size, dtype=dtype)

    return video


class MockImagingExtractor(GaussianNoiseImagingExtractor):
    """A mock imaging extractor for use in tests.

    Generates Gaussian noise data and supports optional synthetic timestamps and a
    configurable dtype, making it suitable for testing code paths that depend on
    those properties.

    Parameters
    ----------
    num_samples : int, default 30
        Number of samples in the video.
    num_rows : int, default 10
        Number of rows in each sample.
    num_columns : int, default 10
        Number of columns in each sample.
    num_planes : int or None, default None
        Number of depth planes. When not None the extractor is volumetric.
    sampling_frequency : float, default 30.0
        Sampling frequency in Hz.
    seed : int, default 0
        Random seed for reproducibility.
    noise_mean : float, default 0.0
        Mean of the Gaussian noise distribution.
    noise_std : float, default 1.0
        Standard deviation of the Gaussian noise distribution.
    native_timestamps : "evenly_spaced" | "unevenly_spaced" | None, default None
        Controls whether the extractor returns native timestamps.
        None: no native timestamps (returns None).
        "evenly_spaced": evenly spaced timestamps based on sampling_frequency.
        "unevenly_spaced": timestamps with small random jitter around the regular spacing.
    dtype : DTypeLike, default np.float32
        The dtype of the data returned by ``get_series``.
    """

    def __init__(
        self,
        *,
        num_samples: int = 30,
        num_rows: int = 10,
        num_columns: int = 10,
        num_planes: int | None = None,
        sampling_frequency: float = 30.0,
        seed: int = 0,
        noise_mean: float = 0.0,
        noise_std: float = 1.0,
        native_timestamps: Literal["evenly_spaced", "unevenly_spaced"] | None = None,
        dtype: DTypeLike = np.float32,
    ):
        super().__init__(
            num_samples=num_samples,
            num_rows=num_rows,
            num_columns=num_columns,
            num_planes=num_planes,
            sampling_frequency=sampling_frequency,
            seed=seed,
            noise_mean=noise_mean,
            noise_std=noise_std,
        )
        self._dtype = np.dtype(dtype)

        valid_types = (None, "evenly_spaced", "unevenly_spaced")
        if native_timestamps not in valid_types:
            raise ValueError(f"native_timestamps must be one of {valid_types}, got '{native_timestamps}'")

        num_samples = self.get_num_samples()
        sampling_frequency = self.get_sampling_frequency()

        if native_timestamps is None:
            self._native_timestamps = None
        elif native_timestamps == "evenly_spaced":
            self._native_timestamps = np.arange(num_samples) / sampling_frequency
        elif native_timestamps == "unevenly_spaced":
            rng = np.random.default_rng(self._seed)
            timestamps = np.arange(num_samples) / sampling_frequency
            jitter = rng.normal(loc=0.0, scale=0.1 / sampling_frequency, size=num_samples)
            self._native_timestamps = np.sort(timestamps + jitter)

    def get_dtype(self) -> np.dtype:
        return self._dtype

    def get_series(self, start_sample: int | None = None, end_sample: int | None = None) -> np.ndarray:
        return super().get_series(start_sample, end_sample).astype(self._dtype, copy=False)

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        if self._native_timestamps is None:
            return None
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self.get_num_samples()
        return self._native_timestamps[start_sample:end_sample]


def generate_dummy_imaging_extractor(
    *,
    num_rows: int = 10,
    num_columns: int = 10,
    sampling_frequency: float = 30.0,
    dtype: DTypeLike = np.float32,
    seed: int = 0,
    num_samples: int | None = 30,
    has_native_timestamps: bool = False,
    native_timestamps: Literal["evenly_spaced", "unevenly_spaced"] | None = None,
    num_planes: int | None = None,
):
    """Generate a dummy imaging extractor for testing.

    The imaging extractor is built using a `MockImagingExtractor` which generates
    Gaussian noise on-the-fly.

    Parameters
    ----------
    num_rows : int, optional
        number of rows in the video, by default 10.
    num_columns : int, optional
        number of columns in the video, by default 10.
    sampling_frequency : float, optional
        sampling frequency of the video, by default 30.
    dtype : DTypeLike, optional
        dtype of the returned data, by default np.float32.
    seed : int, default 0
        seed for the random number generator, by default 0.
    num_samples : int, default 30
        number of samples in the video, by default 30.
    has_native_timestamps : bool, default False
        Deprecated. Use ``native_timestamps="evenly_spaced"`` instead.
    native_timestamps : "evenly_spaced" | "unevenly_spaced" | None, default None
        Controls whether the extractor returns native timestamps.
        None: no native timestamps (returns None).
        "evenly_spaced": evenly spaced timestamps based on sampling_frequency.
        "unevenly_spaced": timestamps with small random jitter around the regular spacing.
    num_planes : int, optional
        number of depth planes for volumetric data. If None, creates 2D data.

    Returns
    -------
    ImagingExtractor
        An imaging extractor with random Gaussian noise data.
    """
    if has_native_timestamps:
        warnings.warn(
            "has_native_timestamps is deprecated and will be removed in or after September 2026. "
            'Use native_timestamps="evenly_spaced" instead.',
            FutureWarning,
            stacklevel=2,
        )
        if native_timestamps is None:
            native_timestamps = "evenly_spaced"

    return MockImagingExtractor(
        num_samples=num_samples,
        num_rows=num_rows,
        num_columns=num_columns,
        num_planes=num_planes,
        sampling_frequency=sampling_frequency,
        seed=seed,
        native_timestamps=native_timestamps,
        dtype=dtype,
    )


class _DummySegmentationExtractor(NumpySegmentationExtractor):
    """A private subclass of NumpySegmentationExtractor that optionally provides native timestamps.

    NumpySegmentationExtractor returns None for get_native_timestamps() because numpy arrays
    do not have native timestamps. This subclass allows the dummy generator to optionally
    produce timestamps without modifying the underlying class.
    """

    def __init__(self, *args, native_timestamps: np.ndarray | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._native_timestamps = native_timestamps

    def get_native_timestamps(
        self, start_sample: int | None = None, end_sample: int | None = None
    ) -> np.ndarray | None:
        if self._native_timestamps is None:
            return None
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = self.get_num_samples()
        return self._native_timestamps[start_sample:end_sample]


def generate_dummy_segmentation_extractor(
    *,
    num_rois: int = 10,
    num_rows: int = 25,
    num_columns: int = 25,
    sampling_frequency: float = 30.0,
    has_summary_images: bool = True,
    has_raw_signal: bool = True,
    has_dff_signal: bool = True,
    has_deconvolved_signal: bool = True,
    has_neuropil_signal: bool = True,
    rejected_list: list | None = None,
    seed: int = 0,
    num_samples: int | None = 30,
    mask_type: Literal["image", "pixel"] = "image",
    native_timestamps: Literal["evenly_spaced", "unevenly_spaced"] | None = None,
) -> SegmentationExtractor:
    """Generate a dummy segmentation extractor for testing.

    The segmentation extractor is built by feeding random data into the
    `NumpySegmentationExtractor`.

    Parameters
    ----------
    num_rois : int, optional
        number of regions of interest, by default 10.
    num_rows : int, optional
        number of rows in the hypothetical video from which the data was extracted, by default 25.
    num_columns : int, optional
        number of columns in the hypothetical video from which the data was extracted, by default 25.
    sampling_frequency : float, optional
        sampling frequency of the hypothetical video from which the data was extracted, by default 30.0.
    has_summary_images : bool, optional
        whether the dummy segmentation extractor has summary images or not (mean and correlation).
    has_raw_signal : bool, optional
        whether a raw fluorescence signal is desired in the object, by default True.
    has_dff_signal : bool, optional
        whether a relative (df/f) fluorescence signal is desired in the object, by default True.
    has_deconvolved_signal : bool, optional
        whether a deconvolved signal is desired in the object, by default True.
    has_neuropil_signal : bool, optional
        whether a neuropil signal is desired in the object, by default True.
    rejected_list: list, optional
        A list of rejected rois, None by default.
    seed : int, default 0
        seed for the random number generator, by default 0.
    num_samples : int, optional
        Number of samples in the recording, by default 30.
    mask_type : str, default "image"
        Type of mask to generate. One of "image" or "pixel".
        "image" generates dense masks of shape (num_rows, num_columns, num_rois).
        "pixel" generates sparse masks as a list of (n_pixels, 3) arrays with columns [y, x, weight].
    native_timestamps : "evenly_spaced" | "unevenly_spaced" | None, default None
        Controls whether the extractor returns native timestamps.
        None: no native timestamps (returns None).
        "evenly_spaced": evenly spaced timestamps based on sampling_frequency.
        "unevenly_spaced": timestamps with small random jitter around the regular spacing.

    Returns
    -------
    SegmentationExtractor
        A segmentation extractor with random data fed into `NumpySegmentationExtractor`

    Notes
    -----
    Note that this dummy example is meant to be a mock object with the right shape, structure and objects but does not
    contain meaningful content. That is, the image masks matrices are not plausible image mask for a roi, the raw signal
    is not a meaningful biological signal and is not related appropriately to the deconvolved signal , etc.
    """
    valid_mask_types = ("image", "pixel")
    if mask_type not in valid_mask_types:
        raise ValueError(f"mask_type must be one of {valid_mask_types}, got '{mask_type}'")

    rng = np.random.default_rng(seed)

    # Create dummy image masks (always needed for NumpySegmentationExtractor construction)
    image_masks = rng.random((num_rows, num_columns, num_rois))
    movie_dims = (num_rows, num_columns)

    # Create signals
    raw = rng.random((num_samples, num_rois)) if has_raw_signal else None
    dff = rng.random((num_samples, num_rois)) if has_dff_signal else None
    deconvolved = rng.random((num_samples, num_rois)) if has_deconvolved_signal else None
    neuropil = rng.random((num_samples, num_rois)) if has_neuropil_signal else None

    # Summary images
    mean_image = rng.random((num_rows, num_columns)) if has_summary_images else None
    correlation_image = rng.random((num_rows, num_columns)) if has_summary_images else None

    # Rois
    width = len(
        str(num_rois - 1)
    )  # e.g., width=2 for 10 ROIs (roi_00, roi_01, ..., roi_09), width=3 for 100 ROIs (roi_000, roi_001, ..., roi_099)
    roi_ids = [f"roi_{id:0{width}d}" for id in range(num_rois)]
    roi_locations_rows = rng.integers(low=0, high=num_rows, size=num_rois)
    roi_locations_columns = rng.integers(low=0, high=num_columns, size=num_rois)
    roi_locations = np.vstack((roi_locations_rows, roi_locations_columns))

    accepted_list = roi_ids
    if rejected_list is not None:
        accepted_list = list(set(accepted_list).difference(rejected_list))

    # Generate native timestamps if requested
    native_timestamps_array = None
    if native_timestamps is None:
        native_timestamps_array = None
    elif native_timestamps == "evenly_spaced":
        native_timestamps_array = np.arange(num_samples) / sampling_frequency
    elif native_timestamps == "unevenly_spaced":
        timestamps = np.arange(num_samples) / sampling_frequency
        jitter = rng.normal(loc=0.0, scale=0.1 / sampling_frequency, size=num_samples)
        native_timestamps_array = np.sort(timestamps + jitter)
    else:
        valid_types = (None, "evenly_spaced", "unevenly_spaced")
        raise ValueError(f"native_timestamps must be one of {valid_types}, got '{native_timestamps}'")

    dummy_segmentation_extractor = _DummySegmentationExtractor(
        sampling_frequency=sampling_frequency,
        image_masks=image_masks,
        raw=raw,
        dff=dff,
        deconvolved=deconvolved,
        neuropil=neuropil,
        mean_image=mean_image,
        correlation_image=correlation_image,
        roi_ids=roi_ids,
        roi_locations=roi_locations,
        accepted_list=accepted_list,
        rejected_list=rejected_list,
        movie_dims=movie_dims,
        channel_names=["channel_num_0"],
        native_timestamps=native_timestamps_array,
    )

    # Replace mask data with pixel masks if requested
    if mask_type == "pixel":
        from .segmentationextractor import _ROIMasks

        num_pixels_per_roi = 5
        pixel_masks = []
        for _ in range(num_rois):
            y_coords = rng.integers(low=0, high=num_rows, size=num_pixels_per_roi).astype(float)
            x_coords = rng.integers(low=0, high=num_columns, size=num_pixels_per_roi).astype(float)
            weights = rng.random(num_pixels_per_roi)
            pixel_masks.append(np.column_stack([y_coords, x_coords, weights]))

        roi_id_map = {roi_id: index for index, roi_id in enumerate(dummy_segmentation_extractor.get_roi_ids())}
        dummy_segmentation_extractor._roi_masks = _ROIMasks(
            data=pixel_masks,
            mask_tpe="nwb-pixel_mask",
            field_of_view_shape=(num_rows, num_columns),
            roi_id_map=roi_id_map,
        )

    return dummy_segmentation_extractor


def _assert_iterable_shape(iterable, shape):
    """Assert that the iterable has the given shape. If the iterable is a numpy array, the shape is checked directly."""
    ar = iterable if isinstance(iterable, np.ndarray) else np.array(iterable)
    for ar_shape, given_shape in zip(ar.shape, shape):
        if isinstance(given_shape, int):
            assert ar_shape == given_shape, f"Expected {given_shape}, received {ar_shape}!"


def _assert_iterable_shape_max(iterable, shape_max):
    """Assert that the iterable has a shape less than or equal to the given maximum shape."""
    ar = iterable if isinstance(iterable, np.ndarray) else np.array(iterable)
    for ar_shape, given_shape in zip(ar.shape, shape_max):
        if isinstance(given_shape, int):
            assert ar_shape <= given_shape


def _assert_iterable_element_dtypes(iterable, dtypes):
    """Assert that the iterable has elements of the given dtypes."""
    if isinstance(iterable, Iterable) and not isinstance(iterable, str):
        for iter in iterable:
            _assert_iterable_element_dtypes(iter, dtypes)
    else:
        assert isinstance(iterable, dtypes), f"array is none of the types {dtypes}"


def _assert_iterable_complete(iterable, dtypes=None, element_dtypes=None, shape=None, shape_max=None):
    """Assert that the iterable is complete, i.e. it is not None and has the given dtypes, element_dtypes, shape and shape_max."""
    assert isinstance(iterable, dtypes), f"iterable {type(iterable)} is none of the types {dtypes}"
    if not isinstance(iterable, NoneType):
        if shape is not None:
            _assert_iterable_shape(iterable, shape=shape)
        if shape_max is not None:
            _assert_iterable_shape_max(iterable, shape_max=shape_max)
        if element_dtypes is not None:
            _assert_iterable_element_dtypes(iterable, element_dtypes)


def check_segmentations_equal(
    segmentation_extractor1: SegmentationExtractor, segmentation_extractor2: SegmentationExtractor
):
    """Check that two segmentation extractors have equal fields."""
    check_segmentation_return_types(segmentation_extractor1)
    check_segmentation_return_types(segmentation_extractor2)
    # assert equality:
    assert segmentation_extractor1.get_num_rois() == segmentation_extractor2.get_num_rois()
    assert segmentation_extractor1.get_num_samples() == segmentation_extractor2.get_num_samples()
    assert np.isclose(
        segmentation_extractor1.get_sampling_frequency(), segmentation_extractor2.get_sampling_frequency()
    )
    assert_array_equal(segmentation_extractor1.get_frame_shape(), segmentation_extractor2.get_frame_shape())
    assert_array_equal(
        segmentation_extractor1.get_roi_image_masks(roi_ids=segmentation_extractor1.get_roi_ids()[:1]),
        segmentation_extractor2.get_roi_image_masks(roi_ids=segmentation_extractor2.get_roi_ids()[:1]),
    )
    assert set(
        segmentation_extractor1.get_roi_pixel_masks(roi_ids=segmentation_extractor1.get_roi_ids()[:1])[0].flatten()
    ) == set(
        segmentation_extractor2.get_roi_pixel_masks(roi_ids=segmentation_extractor1.get_roi_ids()[:1])[0].flatten()
    )

    check_segmentations_images(segmentation_extractor1, segmentation_extractor2)

    assert_array_equal(segmentation_extractor1.get_accepted_list(), segmentation_extractor2.get_accepted_list())
    assert_array_equal(segmentation_extractor1.get_rejected_list(), segmentation_extractor2.get_rejected_list())
    assert_array_equal(segmentation_extractor1.get_roi_ids(), segmentation_extractor2.get_roi_ids())
    assert_array_equal(segmentation_extractor1.get_traces(), segmentation_extractor2.get_traces())

    assert_array_equal(
        segmentation_extractor1.get_timestamps(),
        segmentation_extractor2.get_timestamps(),
    )


def check_segmentations_images(
    segmentation_extractor1: SegmentationExtractor,
    segmentation_extractor2: SegmentationExtractor,
):
    """Check that the segmentation images are equal for the given segmentation extractors."""
    images_in_extractor1 = segmentation_extractor1.get_images_dict()
    images_in_extractor2 = segmentation_extractor2.get_images_dict()

    assert len(images_in_extractor1) == len(images_in_extractor2)

    image_names_are_equal = all(image_name in images_in_extractor1.keys() for image_name in images_in_extractor2.keys())
    assert image_names_are_equal, "The names of segmentation images in the segmentation extractors are not the same."

    for image_name in images_in_extractor1.keys():
        assert_array_equal(
            images_in_extractor1[image_name],
            images_in_extractor2[image_name],
        ), f"The segmentation images for {image_name} are not equal."


def check_segmentation_return_types(seg: SegmentationExtractor):
    """Check that the return types of the segmentation extractor are correct."""
    assert isinstance(seg.get_num_rois(), int)
    assert isinstance(seg.get_num_samples(), int)
    assert isinstance(seg.get_num_channels(), int)
    assert isinstance(seg.get_sampling_frequency(), (NoneType, floattype))
    _assert_iterable_complete(
        seg.get_channel_names(),
        dtypes=list,
        element_dtypes=str,
        shape_max=(seg.get_num_channels(),),
    )
    _assert_iterable_complete(seg.get_image_size(), dtypes=Iterable, element_dtypes=inttype, shape=(2,))
    _assert_iterable_complete(
        seg.get_roi_image_masks(roi_ids=seg.get_roi_ids()[:1]),
        dtypes=(np.ndarray,),
        element_dtypes=floattype,
        shape=(*seg.get_image_size(), 1),
    )
    _assert_iterable_complete(
        seg.get_roi_ids(),
        dtypes=(list,),
        shape=(seg.get_num_rois(),),
    )
    assert isinstance(seg.get_roi_pixel_masks(roi_ids=seg.get_roi_ids()[:2]), list)
    _assert_iterable_complete(
        seg.get_roi_pixel_masks(roi_ids=seg.get_roi_ids()[:1])[0],
        dtypes=(np.ndarray,),
        element_dtypes=floattype,
        shape_max=(np.prod(seg.get_image_size()), 3),
    )
    for image_name in seg.get_images_dict():
        _assert_iterable_complete(
            seg.get_image(image_name),
            dtypes=(np.ndarray, NoneType),
            element_dtypes=floattype,
            shape_max=(*seg.get_image_size(),),
        )
    _assert_iterable_complete(
        seg.get_accepted_list(),
        dtypes=(list, NoneType),
        shape_max=(seg.get_num_rois(),),
    )
    _assert_iterable_complete(
        seg.get_rejected_list(),
        dtypes=(list, NoneType),
        shape_max=(seg.get_num_rois(),),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="get_roi_locations", category=FutureWarning)
        _assert_iterable_complete(
            seg.get_roi_locations(),
            dtypes=(np.ndarray,),
            shape=(2, seg.get_num_rois()),
            element_dtypes=inttype,
        )
    _assert_iterable_complete(
        seg.get_traces(),
        dtypes=(np.ndarray, NoneType),
        element_dtypes=floattype,
        shape=(np.prod(seg.get_num_rois()), None),
    )
    assert isinstance(seg.get_traces_dict(), dict)
    assert isinstance(seg.get_images_dict(), dict)
    assert {"raw", "dff", "neuropil", "deconvolved", "denoised"} == set(seg.get_traces_dict().keys())


def check_imaging_equal(imaging_extractor1: ImagingExtractor, imaging_extractor2: ImagingExtractor):
    """Check that two imaging extractors have equal fields."""
    # assert equality:
    assert imaging_extractor1.get_num_samples() == imaging_extractor2.get_num_samples()
    assert np.isclose(imaging_extractor1.get_sampling_frequency(), imaging_extractor2.get_sampling_frequency())
    assert_array_equal(imaging_extractor1.get_sample_shape(), imaging_extractor2.get_sample_shape())

    assert_array_equal(
        imaging_extractor1.get_series(start_sample=0, end_sample=1),
        imaging_extractor2.get_series(start_sample=0, end_sample=1),
    )

    assert_array_almost_equal(
        imaging_extractor1.get_timestamps(),
        imaging_extractor2.get_timestamps(),
    )


def check_imaging_return_types(img_ex: ImagingExtractor):
    """Check that the return types of the imaging extractor are correct."""
    assert isinstance(img_ex.get_num_samples(), inttype)
    assert isinstance(img_ex.get_sampling_frequency(), floattype)
    _assert_iterable_complete(
        iterable=img_ex.get_channel_names(),
        dtypes=(list, NoneType),
        element_dtypes=str,
        shape_max=(1,),
    )
    _assert_iterable_complete(iterable=img_ex.get_image_size(), dtypes=Iterable, element_dtypes=inttype, shape=(2,))

    # This needs a method for getting frame shape not image size. It only works for n_channel==1
    # two_first_frames = img_ex.get_frames(frame_idxs=[0, 1])
    # _assert_iterable_complete(
    #     iterable=two_first_frames,
    #     dtypes=(np.ndarray,),
    #     element_dtypes=inttype + floattype,
    #     shape=(2, *img_ex.get_image_size()),
    # )
