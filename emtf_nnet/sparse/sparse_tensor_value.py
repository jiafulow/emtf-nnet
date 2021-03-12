# The following source code was originally obtained from:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/sparse_tensor.py
# ==============================================================================

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Value for SparseTensor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np


class SparseTensorValue(object):
  """Represents the value of a `SparseTensor`.

  See [Sparse tensor](https://www.tensorflow.org/guide/sparse_tensor) for descriptions.

  Example:

  >>> SparseTensorValue(indices=np.array([[0, 0], [1, 2], [2, 3]]),
  ...                   values=np.array([1, 2, 3]),
  ...                   dense_shape=np.array([3, 4]))
  SparseTensorValue(indices=array([[0, 0],
         [1, 2],
         [2, 3]]), values=array([1, 2, 3]), dense_shape=array([3, 4]))
  """

  def __init__(self, indices, values, dense_shape):
    """Creates a `SparseTensor`.

    Args:
      indices: A 2-D int64 tensor of shape `[N, ndims]`.
      values: A 1-D tensor of any type and shape `[N]`.
      dense_shape: A 1-D int64 tensor of shape `[ndims]`.
    """
    if not (isinstance(indices, (np.ndarray, np.generic)) and
            indices.dtype in (np.int64, np.int32) and indices.ndim == 2):
      raise TypeError("indices must be a 2D int32 or int64 numpy array")
    if not (isinstance(values, (np.ndarray, np.generic)) and values.ndim >= 1):
      raise TypeError("values must be a n-D numpy array")
    if not (isinstance(dense_shape, (np.ndarray, np.generic)) and
            dense_shape.dtype in (np.int64, np.int32) and dense_shape.ndim == 1):
      raise TypeError("dense_shape must be a 1D int32 or int64 numpy array")
    if not (indices.shape[0] == values.shape[0]):
      raise TypeError("indices and values must have the same first dim")
    if not (indices.shape[1] + (values.ndim - 1) == dense_shape.shape[0]):
      raise TypeError("indices, values, and dense_shape must have consistent shapes")
    self._indices = indices
    self._values = values
    self._dense_shape = dense_shape

  indices = property(
      lambda self: self._indices,
      doc="""The indices of non-zero values in the represented dense tensor.""")
  values = property(
      lambda self: self._values,
      doc="""The non-zero values in the represented dense tensor.""")
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
    return "SparseTensorValue(indices=%s, values=%s, dense_shape=%s)" % (
        self._indices, self._values, self._dense_shape)

  def __repr__(self):
    return "SparseTensorValue(indices=%r, values=%r, dense_shape=%r)" % (
        self._indices, self._values, self._dense_shape)

  def with_values(self, new_values):
    """Returns a copy of `self` with `values` replaced by `new_values`."""
    return SparseTensorValue(self._indices, new_values, self._dense_shape)


SparseTensorNamedTuple = collections.namedtuple(
    'SparseTensorNamedTuple', ['indices', 'values', 'dense_shape'])


def dense_to_sparse(dense):
  dense = np.asarray(dense)
  indices = np.argwhere(dense)
  values = dense[dense.nonzero()]
  dense_shape = np.asarray(dense.shape)
  return SparseTensorValue(indices=indices, values=values, dense_shape=dense_shape)


def sparse_to_dense(sparse):
  dense = np.zeros(sparse.dense_shape, dtype=sparse.dtype)
  ndims = sparse.indices.shape[1]
  tup = tuple(sparse.indices[:, i] for i in range(ndims))
  dense[tup] = sparse.values
  return dense


def sparse_to_dense_n(sparse, n):
  dense_shape = (n,) + sparse.dense_shape[1:]
  dense = np.zeros(dense_shape, dtype=sparse.dtype)
  for i in range(len(sparse.indices)):
    if sparse.indices[i, 0] >= n:
      break
    tup = tuple(sparse.indices[i])
    dense[tup] = sparse.values[i]
  return dense
