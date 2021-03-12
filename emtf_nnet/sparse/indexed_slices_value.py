# The following source code was originally obtained from:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/indexed_slices.py
# ==============================================================================

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Value for IndexedSlices."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np


class IndexedSlicesValue(object):
  """Represents the value of an `IndexedSlices`.

  See `tf.IndexedSlices` API for descriptions.

  Example:

  >>> IndexedSlicesValue(indices=np.array([0, 2, 4]),
  ...                    values=np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]]),
  ...                    dense_shape=np.array([6, 4]))
  IndexedSlicesValue(indices=array([0, 2, 4]), values=array([[ 1,  2,  3,  4],
         [-1, -2, -3, -4],
         [ 5,  6,  7,  8]]), dense_shape=array([6, 4]))
  """

  def __init__(self, indices, values, dense_shape):
    """Creates an `IndexedSlices`.

    Args:
      indices: A 1-D integer Tensor with shape [D0].
      values: A Tensor of any dtype with shape [D0, D1, ..., Dn].
      dense_shape: A 1-D int64 tensor of shape [ndims], e.g. [LARGE0, D1, .. , DN] where LARGE0 >> D0
    """
    if not (isinstance(indices, (np.ndarray, np.generic)) and
            indices.dtype in (np.int64, np.int32) and indices.ndim == 1):
      raise TypeError("indices must be a 1D int32 or int64 numpy array")
    if not (isinstance(values, (np.ndarray, np.generic)) and values.ndim >= 1):
      raise TypeError("values must be a n-D numpy array")
    if not (isinstance(dense_shape, (np.ndarray, np.generic)) and
            dense_shape.dtype in (np.int64, np.int32) and dense_shape.ndim == 1):
      raise TypeError("dense_shape must be a 1D int32 or int64 numpy array")
    self._indices = indices
    self._values = values
    self._dense_shape = dense_shape

  indices = property(
      lambda self: self._indices,
      doc="""The indices of the tensor slices.""")
  values = property(
      lambda self: self._values,
      doc="""The values of the tensor slices.""")
  dtype = property(
      lambda self: self._values.dtype,
      doc="""The numpy dtype of values in this tensor.""")
  dense_shape = property(
      lambda self: tuple(self._dense_shape),
      doc="""A tuple representing the shape of the dense tensor.""")
  shape = property(
      lambda self: tuple(self._dense_shape),
      doc="""A tuple representing the shape of the dense tensor.""")

  def __str__(self):
    return "IndexedSlicesValue(indices=%s, values=%s, dense_shape=%s)" % (
        self._indices, self._values, self._dense_shape)

  def __repr__(self):
    return "IndexedSlicesValue(indices=%r, values=%r, dense_shape=%r)" % (
        self._indices, self._values, self._dense_shape)

  def with_values(self, new_values):
    """Returns a copy of `self` with `values` replaced by `new_values`."""
    return IndexedSlicesValue(self._indices, new_values, self._dense_shape)


IndexedSlicesNamedTuple = collections.namedtuple(
    'IndexedSlicesNamedTuple', ['indices', 'values', 'dense_shape'])


def dense_to_indexed_slices(dense, indices):
  dense = np.asarray(dense)
  indices = np.asarray(indices)
  values = dense[indices]
  dense_shape = np.asarray(dense.shape)
  return IndexedSlicesValue(indices=indices, values=values, dense_shape=dense_shape)


def indexed_slices_to_dense(indexed):
  dense = np.zeros(indexed.dense_shape, dtype=indexed.dtype)
  dense[indexed.indices] = indexed.values
  return dense
