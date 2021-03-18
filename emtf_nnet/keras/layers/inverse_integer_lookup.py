# The following source code was originally obtained from:
# https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/layers/preprocessing/integer_lookup.py
# https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/layers/preprocessing/index_lookup.py
# ==============================================================================

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Integer lookup preprocessing layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.engine.base_layer import Layer


class InverseIntegerLookup(Layer):
  """Maps integer indices to integer vocabulary items."""

  def __init__(self,
               vocabulary,
               max_values=None,
               num_oov_indices=0,
               mask_value=None,
               oov_value=-1,
               invert=True,
               **kwargs):
    allowed_dtypes = [dtypes.int32]

    if 'dtype' in kwargs and kwargs['dtype'] not in allowed_dtypes:
      raise ValueError('The value of the dtype argument for InverseIntegerLookup may '
                       'only be one of %s.' % (allowed_dtypes,))

    if 'dtype' not in kwargs:
      kwargs['dtype'] = dtypes.int32

    # If max_values is set, the value must be greater than 1 - otherwise we
    # are creating a 0-element vocab, which doesn't make sense.
    if max_values is not None and max_values <= 1:
      raise ValueError('If set, max_values must be greater than 1. '
                       'You passed %s' % (max_values,))

    if num_oov_indices < 0:
      raise ValueError(
          'num_oov_indices must be greater than or equal to 0. You passed %s' %
          (num_oov_indices,))

    super(InverseIntegerLookup, self).__init__(**kwargs)
    self.vocabulary = vocabulary
    self.max_values = max_values  # unused
    self.num_oov_indices = num_oov_indices  # unused
    self.mask_value = mask_value  # unused
    self.oov_value = oov_value
    self.invert = invert  # unused
    self._key_dtype = dtypes.as_dtype(self.dtype)
    self._value_dtype = dtypes.as_dtype(self.dtype)

  def build(self, input_shape):
    tokens = np.array(self.vocabulary, dtype=self._value_dtype.as_numpy_dtype)
    indices = np.arange(len(tokens), dtype=self._key_dtype.as_numpy_dtype)
    keys = ops.convert_to_tensor_v2_with_dispatch(indices)
    values = ops.convert_to_tensor_v2_with_dispatch(tokens)
    if values.shape.ndims != 1:
      raise ValueError('`values` must be 1-dimensional, got an input with '
                       ' %s dimensions.' % values.shape.ndims)
    self._table = lookup_ops.StaticHashTable(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_value=self.oov_value)
    self.built = True

  def call(self, inputs):
    inputs = ops.convert_to_tensor_v2_with_dispatch(inputs)
    if inputs.dtype != self._key_dtype:
      inputs = math_ops.cast(inputs, self._key_dtype)
    outputs = self._table.lookup(inputs)
    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
      'vocabulary': self.vocabulary,
      'max_values': self.max_values,
      'num_oov_indices': self.num_oov_indices,
      'mask_value': self.mask_value,
      'oov_value': self.oov_value,
      'invert': self.invert,
    }
    base_config = super(InverseIntegerLookup, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
