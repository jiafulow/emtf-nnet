# The following source code was originally obtained from:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/ragged/ragged_tensor_value.py
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/ragged/ragged_getitem.py
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/ragged/ragged_factory_ops.py
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/ragged/ragged_math_ops.py
# ==============================================================================

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Value for RaggedTensor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np


class RaggedTensorValue(object):
  """Represents the value of a `RaggedTensor`.

  See [Ragged tensor](https://www.tensorflow.org/guide/ragged_tensor) for descriptions.
  It allows only ragged_rank = 1.

  Example:

  >>> RaggedTensorValue(values=np.array([3, 1, 4, 1, 5, 9, 2]), row_splits=np.array([0, 4, 4, 6, 7]))
  RaggedTensorValue(values=array([3, 1, 4, 1, 5, 9, 2]), row_splits=array([0, 4, 4, 6, 7]))
  """

  def __init__(self, values, row_splits):
    """Creates a `RaggedTensorValue`.

    Args:
      values: A numpy array of any type and shape; or a RaggedTensorValue.
      row_splits: A 1-D int32 or int64 numpy array.
    """
    if not (isinstance(row_splits, (np.ndarray, np.generic)) and
            row_splits.dtype in (np.int64, np.int32) and row_splits.ndim == 1):
      raise TypeError("row_splits must be a 1D int32 or int64 numpy array")
    if not isinstance(values, (np.ndarray, np.generic, RaggedTensorValue)):
      raise TypeError("values must be a numpy array or a RaggedTensorValue")
    # flake8: noqa:E129
    if (isinstance(values, RaggedTensorValue) and
        row_splits.dtype != values.row_splits.dtype):
      raise ValueError("row_splits and values.row_splits must have "
                       "the same dtype")
    self._values = values
    self._row_splits = row_splits

  row_splits = property(
      lambda self: self._row_splits,
      doc="""The split indices for the ragged tensor value.""")
  values = property(
      lambda self: self._values,
      doc="""The concatenated values for all rows in this tensor.""")
  dtype = property(
      lambda self: self._values.dtype,
      doc="""The numpy dtype of values in this tensor.""")
  row_lengths = property(
      lambda self: self._row_splits[1:] - self._row_splits[:-1],
      doc="""The lengths of the rows in this ragged tensor value.""")
  nrows = property(
      lambda self: self._row_splits.shape[0] - 1,
      doc="""The number of rows in this ragged tensor value.""")
  flat_values = property(
      lambda self: self.values,
      doc="""The innermost `values` array for this ragged tensor value.""")
  nested_row_splits = property(
      lambda self: tuple([self.row_splits]),
      doc="""The row_splits for all ragged dimensions in this ragged tensor value.""")
  ragged_rank = property(
      lambda self: 1,
      doc="""The number of ragged dimensions in this ragged tensor value.""")

  @property
  def shape(self):
    """A tuple indicating the shape of this RaggedTensorValue."""
    return (self._row_splits.shape[0] - 1,) + (None,) + self._values.shape[1:]

  def __str__(self):
    return "RaggedTensorValue(values=%s, row_splits=%s)" % (self._values,
                                                            self._row_splits)

  def __repr__(self):
    return "RaggedTensorValue(values=%r, row_splits=%r)" % (self._values,
                                                            self._row_splits)

  def __len__(self):
    return self.nrows

  def __getitem__(self, row_key):
    # Slicing a range of rows
    if isinstance(row_key, slice):
      # Use row_key to slice the starts & limits.
      new_starts = self.row_splits[:-1][row_key]
      new_limits = self.row_splits[1:][row_key]
      zero_pad = np.arange(1, dtype=self.row_splits.dtype)

      # If there's no slice step, then we can just select a single continuous
      # span of `ragged.values(rt_input)`.
      if row_key.step is None or row_key.step == 1:
        # Construct the new splits.  If new_starts and new_limits are empty,
        # then this reduces to [0].  Otherwise, this reduces to:
        #   concat([[new_starts[0]], new_limits])
        new_splits = np.concatenate(
            [zero_pad[new_starts.size:], new_starts[:1], new_limits],
            axis=0)
        values_start = new_splits[0]
        values_limit = new_splits[-1]
        return RaggedTensorValue(
            self.values[values_start:values_limit], new_splits - values_start)
      else:
        raise ValueError("slicing with slice step is not supported")

    # Indexing with an index array
    elif (isinstance(row_key, (np.ndarray, np.generic)) and
          row_key.dtype in (np.int64, np.int32) and row_key.ndim == 1):
      # Use row_key to slice the starts & limits.
      new_starts = self.row_splits[:-1][row_key]
      new_limits = self.row_splits[1:][row_key]
      value_ranges = ragged_range(new_starts, new_limits)
      return RaggedTensorValue(
          self.values[value_ranges.values], value_ranges.row_splits)

    # Indexing a single row
    starts = self.row_splits[:-1]
    limits = self.row_splits[1:]
    row = self.values[starts[row_key]:limits[row_key]]
    return row

  def __iter__(self):
    for i in range(len(self)):
      yield self[i]

  def to_list(self):
    """Returns this ragged tensor value as a nested Python list."""
    if isinstance(self._values, RaggedTensorValue):
      values_as_list = self._values.to_list()
    else:
      values_as_list = self._values.tolist()
    return [
        values_as_list[self._row_splits[i]:self._row_splits[i + 1]]
        for i in range(self.nrows)
    ]

  def to_array(self):
    """Returns this ragged tensor value as a nested Numpy array."""
    arr = np.empty((self.nrows,), dtype=np.object)
    for i in range(self.nrows):
      arr[i] = self._values[self._row_splits[i]:self._row_splits[i + 1]]
    return arr

  def with_values(self, new_values):
    """Returns a copy of `self` with `values` replaced by `new_value`."""
    return RaggedTensorValue(new_values, self._row_splits)

  def with_flat_values(self, new_values):
    """Returns a copy of `self` with `flat_values` replaced by `new_value`."""
    return self.with_values(new_values)


RaggedTensorNamedTuple = collections.namedtuple(
    'RaggedTensorNamedTuple', ['values', 'row_splits'])


def create_ragged_array(pylist):
  """Construct a constant RaggedTensorValue from a nested list."""

  # Ragged rank for returned value
  ragged_rank = 1

  # Build the splits for each ragged rank, and concatenate the inner values
  # into a single list.
  nested_splits = []
  values = pylist
  for dim in range(ragged_rank):
    nested_splits.append([0])
    concatenated_values = []
    for row in values:
      nested_splits[dim].append(nested_splits[dim][-1] + len(row))
      concatenated_values.extend(row)
    values = concatenated_values

  values = np.asarray(values)
  for row_splits in reversed(nested_splits):
    row_splits = np.asarray(row_splits, dtype=np.int32)
    values = RaggedTensorValue(values, row_splits)
  return values


def ragged_stack(tup):
  if not np.all([x.values.ndim > 1 for x in tup]):
    raise TypeError("ragged values must be at least 2D")

  tup_values = [x.values for x in tup]
  tup_row_splits = [x.row_splits for x in tup]

  new_values = np.vstack(tup_values)
  new_row_splits = [0]
  for row_splits in tup_row_splits:
    # Ignore the first entry in row_splits, as the first entry is always zero.
    # Increment all the entries in row_splits by the last value in new_row_splits.
    new_row_splits.extend(new_row_splits[-1] + row_splits[1:])

  new_values = np.asarray(new_values)
  new_row_splits = np.asarray(new_row_splits, dtype=np.int32)
  return RaggedTensorValue(new_values, new_row_splits)


def ragged_boolean_mask(ragged, mask):
  if not (isinstance(mask, (np.ndarray, np.generic)) and
          mask.dtype in (np.bool,) and mask.ndim == 1):
    raise TypeError("mask must be a 1D bool numpy array")
  if not isinstance(ragged, (RaggedTensorValue,)):
    raise TypeError("ragged must be a RaggedTensorValue")
  if not (ragged.values.shape[0] == mask.shape[0]):
    raise ValueError("The length of ragged.values must be equal to the length of mask")

  data = ragged.values
  new_values = data[mask]

  new_row_lengths = np.zeros((ragged.nrows,), dtype=ragged.row_splits.dtype)
  for i in range(ragged.nrows):
    new_row_lengths[i] = np.count_nonzero(mask[ragged.row_splits[i]:ragged.row_splits[i + 1]])
  new_row_splits = np.append(0, np.cumsum(new_row_lengths))

  new_values = np.asarray(new_values)
  new_row_splits = np.asarray(new_row_splits, dtype=ragged.row_splits.dtype)
  return RaggedTensorValue(new_values, new_row_splits)


def ragged_row_boolean_mask(ragged, row_mask):
  if not (isinstance(row_mask, (np.ndarray, np.generic)) and
          row_mask.dtype in (np.bool,) and row_mask.ndim == 1):
    raise TypeError("row_mask must be a 1D bool numpy array")
  if not isinstance(ragged, (RaggedTensorValue,)):
    raise TypeError("ragged must be a RaggedTensorValue")
  if not (ragged.nrows == row_mask.shape[0]):
    raise ValueError("The number of rows in ragged must be equal to the length of row_mask")

  data = ragged.values
  mask = np.zeros((ragged.values.shape[0],), dtype=np.bool)
  for i in range(ragged.nrows):
    mask[ragged.row_splits[i]:ragged.row_splits[i + 1]] = row_mask[i]
  new_values = data[mask]

  new_row_lengths = ragged.row_lengths[row_mask]
  new_row_splits = np.append(0, np.cumsum(new_row_lengths))

  new_values = np.asarray(new_values)
  new_row_splits = np.asarray(new_row_splits, dtype=ragged.row_splits.dtype)
  return RaggedTensorValue(new_values, new_row_splits)


def ragged_row_splits_to_segment_ids(row_splits):
  if not (isinstance(row_splits, (np.ndarray, np.generic)) and
          row_splits.dtype in (np.int64, np.int32) and row_splits.ndim == 1):
    raise TypeError("row_splits must be a 1D int32 or int64 numpy array")

  row_lengths = row_splits[1:] - row_splits[:-1]
  nrows = row_splits.shape[0] - 1
  indices = np.arange(nrows, dtype=row_splits.dtype)
  segment_ids = np.repeat(indices, repeats=row_lengths)
  return segment_ids


def ragged_segment_ids_to_row_splits(segment_ids, num_segments=None):
  if not (isinstance(segment_ids, (np.ndarray, np.generic)) and
          segment_ids.dtype in (np.int64, np.int32) and segment_ids.ndim == 1):
    raise TypeError("segment_ids must be a 1D int32 or int64 numpy array")
  if num_segments is not None:
    if not (isinstance(num_segments, (np.ndarray, np.generic)) and
            num_segments.dtype in (np.int64, np.int32) and num_segments.ndim == 0):
      raise TypeError("num_segment must be a 0D int32 or int64 numpy array")

  row_lengths = np.bincount(segment_ids, minlength=num_segments)
  row_splits = np.append(0, np.cumsum(row_lengths))
  row_splits = np.asarray(row_splits, dtype=segment_ids.dtype)
  return row_splits


def ragged_range(starts, limits=None, deltas=1, dtype=np.int32):
  if limits is None:
    starts = np.asarray(starts)
    starts, limits = np.zeros_like(starts, dtype=starts.dtype), starts
  else:
    starts, limits = np.asarray(starts), np.asarray(limits)

  deltas = np.asarray(deltas)
  if deltas.ndim == 0:
    nested_range = [np.arange(start, limit, deltas, dtype=dtype)
                    for (start, limit) in zip(starts, limits)]
  else:
    nested_range = [np.arange(start, limit, delta, dtype=dtype)
                    for (start, limit, delta) in zip(starts, limits, deltas)]

  values = np.concatenate(nested_range, axis=0)
  row_lengths = np.array([len(x) for x in nested_range])
  row_splits = np.append(0, np.cumsum(row_lengths))
  row_splits = np.asarray(row_splits, dtype=np.int32)
  return RaggedTensorValue(values, row_splits)
