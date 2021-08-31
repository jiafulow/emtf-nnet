"""Numpy-related utilities."""

import numpy as np


def div_no_nan(x, y):
  x, y = np.asarray(x), np.asarray(y)
  if y.ndim == 0:
    return (x - x) if y == 0 else x
  elif x.ndim == 0 or x.shape == y.shape:
    mask = (y == 0)
    return (x - (x * mask)) / (y + ((1 - y) * mask))
  else:
    raise ValueError('Inconsistent shapes: x.shape={0} y.shape={1}'.format(
        x.shape, y.shape))
