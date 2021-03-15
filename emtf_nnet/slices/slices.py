# The following source code was originally obtained from:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/array_ops.py
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
"""Scatter-gather operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

#TODO: implement scatter_nd, gather, gather_nd


def scatter_nd(indices, updates, shape):
  raise NotImplementedError


def gather(params, indices, axis=None, batch_dims=0):
  raise NotImplementedError


def gather_nd(params, indices):
  raise NotImplementedError


def gather_indices_by_values(arr, max_value=None):
  """Gather indices by values.

  Instead of gathering values by indices, this function gathers indices by
  (non-negative integer) values in an array. The result is a nested list,
  which can be addressed by content from the original array.

  Negative values in the array are ignored.

  Example:

  >>> arr = [0, 1, 1, 2, 2, 2, 4]
  >>> gather_indices_by_values(arr)
  [[0], [1, 2], [3, 4, 5], [], [6]]
  """
  arr = np.asarray(arr)
  if not (arr.dtype in (np.int64, np.int32) and arr.ndim == 1):
    raise TypeError('arr must be a 1-D int32 or int64 numpy array')
  if max_value is None:
    max_value = np.max(arr)
  return [
      [i for (i, x_i) in enumerate(arr) if x_i == y_j]
      for y_j in range(max_value + 1)
  ]


def gather_inputs_by_outputs(arr, default_value=-99, padding_value=-99):
  """Gather inputs by outputs.

  Assume a 2-D array that is used to map multiple inputs to a smaller number
  of outputs. If two or more inputs are mapped to the same output, the one
  with the highest priority is used as the output.
  """
  arr = np.asarray(arr)
  if not (arr.dtype in (np.int64, np.int32) and arr.ndim == 2):
    raise TypeError('arr must be a 2-D int32 or int64 numpy array')
  max_value = np.max(arr)
  matrix_shape = (max_value + 1, len(arr))
  matrix = np.full(matrix_shape, default_value, dtype=np.int32)
  for i, x_i in enumerate(arr):
    for j, x_j in enumerate(x_i):
      if x_j != padding_value:
        priority = len(x_i) - 1 - j
        matrix[x_j, i] = priority
  return matrix
