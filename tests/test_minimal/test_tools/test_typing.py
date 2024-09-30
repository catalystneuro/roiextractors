from roiextractors.tools.typing import (
    ArrayType,
    PathType,
    DtypeType,
    IntType,
    FloatType,
    NoneType,
)
from numpy.typing import ArrayLike, DTypeLike
import numpy as np
from typing import Union
from pathlib import Path


def test_ArrayType():
    assert ArrayType == ArrayLike


def test_PathType():
    assert PathType == Union[str, Path]


def test_DtypeType():
    assert DtypeType == DTypeLike


def test_IntType():
    assert IntType == Union[int, np.integer]


def test_FloatType():
    assert FloatType == Union[float, np.floating]


def test_NoneType():
    assert NoneType == type(None)
