import numpy as np

from .typing import FloatArray

__all__ = [
    'reduce_sum',
    'reduce_max',
    'restrict_mean',
    'restrict_max',
    'refine_donor'
]


def reduce_sum(a: FloatArray, b: FloatArray) -> FloatArray:
    return a + b


def reduce_max(a: FloatArray, b: FloatArray) -> FloatArray:
    return np.maximum(a, b)


def restrict_mean(a: FloatArray) -> FloatArray:
    b = a.reshape(*(x for s in a.shape for x in (s // 2, 2)))
    return b.mean(axis=tuple(2 * i + 1 for i in range(a.ndim)))


def restrict_max(a: FloatArray) -> FloatArray:
    b = a.reshape(*(x for s in a.shape for x in (s // 2, 2)))
    return b.max(axis=tuple(2 * i + 1 for i in range(a.ndim)))


def refine_donor(a: FloatArray) -> FloatArray:
    b = a.reshape(*(x for s in a.shape for x in (s, 1)))
    c = np.broadcast_to(b, tuple(x for s in a.shape for x in (s, 2)))
    return c.reshape(*(s * 2 for s in a.shape))
