"""Deprecated module. Use :mod:`roiextractors.planarstackimagingextractor` instead."""

import warnings

from .planarstackimagingextractor import (
    DepthSlicePlanarStackImagingExtractor,
    PlanarStackImagingExtractor,
)


class VolumetricImagingExtractor(PlanarStackImagingExtractor):
    """Deprecated alias for :class:`PlanarStackImagingExtractor`.

    .. deprecated::
        ``VolumetricImagingExtractor`` is deprecated and will be removed in or after November 2026.
        Use :class:`PlanarStackImagingExtractor` instead. The old name suggested this was the
        canonical base class for volumetric imaging, but it is actually a composition utility that
        stacks per-plane extractors. Native volumetric extractors should inherit from
        ``ImagingExtractor`` directly and set ``is_volumetric = True``.
    """

    extractor_name = "VolumetricImaging"

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "VolumetricImagingExtractor is deprecated and will be removed in or after November 2026. "
            "Use PlanarStackImagingExtractor instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class DepthSliceVolumetricImagingExtractor(DepthSlicePlanarStackImagingExtractor):
    """Deprecated alias for :class:`DepthSlicePlanarStackImagingExtractor`.

    .. deprecated::
        ``DepthSliceVolumetricImagingExtractor`` is deprecated and will be removed in or after
        November 2026. Use :class:`DepthSlicePlanarStackImagingExtractor` instead.
    """

    extractor_name = "DepthSliceVolumetricImagingExtractor"

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DepthSliceVolumetricImagingExtractor is deprecated and will be removed in or after November 2026. "
            "Use DepthSlicePlanarStackImagingExtractor instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
