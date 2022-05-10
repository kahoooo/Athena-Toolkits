from typing import Callable

import numpy as np
import numpy.typing as npt

__all__ = [
    'IntegerArray',
    'FloatArray',
    'ReduceFunction',
    'RestrictFunction',
    'RefineFunction'
]

IntegerArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float_]
ReduceFunction = Callable[[FloatArray, FloatArray], FloatArray]
RestrictFunction = Callable[[FloatArray], FloatArray]
RefineFunction = Callable[[FloatArray], FloatArray]
