from typing import Union
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

ArrayType = ArrayLike
PathType = Union[str, Path]
DtypeType = DTypeLike
IntType = Union[int, np.integer]
FloatType = Union[float, np.floating]
NoneType = type(None)
