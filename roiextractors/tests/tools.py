import numpy as np
from collections.abc import Iterable

NoneType = type(None)


def assert_iterable_shape(iterable, shape):
    ar = iterable if isinstance(iterable, np.ndarray) else np.array(iterable)
    for ar_shape, given_shape in zip(ar.shape, shape):
        if isinstance(given_shape, int):
            assert ar_shape == given_shape


def assert_iterable_shape_max(iterable, shape_max):
    ar = iterable if isinstance(iterable, np.ndarray) else np.array(iterable)
    for ar_shape, given_shape in zip(ar.shape, shape_max):
        if isinstance(given_shape, int):
            assert ar_shape <= given_shape


def assert_iterable_element_dtypes(iterable, dtypes):
    for iter in iterable:
        if isinstance(iter, Iterable):
            assert_iterable_element_dtypes(iter, dtypes)
        else:
            assert any([isinstance(iter, dtype) for dtype in dtypes]), f'array is none of the types {dtypes}'


def assert_iterable_complete(iterable, dtypes=None, element_dtypes=None, shape=None, shape_max=None):
    assert any([isinstance(iterable, dtype) for dtype in dtypes]), f'iterable is none of the types {dtypes}'
    if not isinstance(iterable, NoneType):
        if shape is not None:
            assert_iterable_shape(iterable, shape=shape)
        if shape_max is not None:
            assert_iterable_shape_max(iterable, shape_max=shape_max)
        if element_dtypes is not None:
            assert_iterable_element_dtypes(iterable, element_dtypes)
