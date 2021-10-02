# The following source code was originally obtained from:
# https://github.com/keras-team/keras/blob/r2.6/keras/utils/data_utils.py
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
"""Utilities for data loading."""

import itertools
import numpy as np

from keras.utils import data_utils


class DataGenerator(data_utils.Sequence):
  """Data generator that implements the base `keras.utils.Sequence`.

  It implements the `__getitem__` and the `__len__` abstract methods.
  """
  def __init__(self, x, batch_size=None, steps=None, shuffle=False):
    super().__init__()
    self.x = x
    self.num_samples = int(x.shape[0])
    if not batch_size:
      batch_size = int(np.ceil(self.num_samples / float(steps))) if steps else 32
    self.batch_size = batch_size
    self.num_batches = int(np.ceil(self.num_samples / float(batch_size)))
    self.shuffle = shuffle
    self.index_array = np.arange(self.num_samples)
    if self.shuffle:
      np.random.shuffle(self.index_array)

  def __len__(self):
    """Number of batch in the Sequence."""
    return self.num_batches

  def __getitem__(self, index):
    """Gets batch at position `index`."""
    start, stop = (index * self.batch_size, min(self.num_samples,
                                                (index + 1) * self.batch_size))
    return self.x[self.index_array[start:stop]]

  def on_epoch_end(self):
    """Method called at the end of every epoch."""
    if self.shuffle:
      np.random.shuffle(self.index_array)


class TransformedDataGenerator(DataGenerator):
  """Data generator that applies data transformation.

  It applies a transformation to each batch of samples while being iterated.
  """
  def __init__(self, x, transform_fn=None, **kwargs):
    super().__init__(x, **kwargs)
    self.transform_fn = transform_fn

  def __getitem__(self, index):
    """Gets a batch of transformed samples."""
    start, stop = (index * self.batch_size, min(self.num_samples,
                                                (index + 1) * self.batch_size))
    if self.transform_fn is None:
      return self.x[self.index_array[start:stop]]
    else:
      return self.transform_fn(self.x[self.index_array[start:stop]])


def train_test_split(*arrays, test_size=0.25, batch_size=32, shuffle=True):
  """Split arrays into train and test subsets."""

  if not len(arrays) >= 2:
    raise ValueError('Expect more than 2 array-like objects.')
  num_samples = arrays[0].shape[0]
  num_train_samples = int(np.ceil(num_samples * (1. - test_size)))
  num_train_samples = int(np.ceil(num_train_samples / float(batch_size)) * batch_size)
  index_array = np.arange(num_samples)
  if shuffle:
    np.random.shuffle(index_array)
  index_array_train = index_array[:num_train_samples]
  index_array_test = index_array[num_train_samples:]
  train_test_pairs = (
      (arr[index_array_train], arr[index_array_test])
      for arr in arrays)
  return tuple(itertools.chain.from_iterable(train_test_pairs))
