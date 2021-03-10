# The following source code is obtained from:
# https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/layers/core.py#L77-L140
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
"""Core Keras layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.layers.core import Layer


class Normalisa(Layer):
  def __init__(self, axis=-1, **kwargs):
    super(Normalisa, self).__init__(**kwargs)
    self.supports_masking = True
    self._compute_output_and_mask_jointly = True

    # Standardize `axis` to a tuple.
    if axis is None:
      axis = ()
    elif isinstance(axis, int):
      axis = (axis,)
    else:
      axis = tuple(axis)

    if 0 in axis:
      raise ValueError('The argument \'axis\' may not be 0.')

    self.axis = axis

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    weight_shape = tuple(input_shape[d] for d in self.axis)
    self.kernel = self.add_weight(
        'kernel',
        shape=weight_shape,
        initializer=init_ops.ones_initializer,
        dtype=self.dtype,
        trainable=False)
    self.bias = self.add_weight(
        'bias',
        shape=weight_shape,
        initializer=init_ops.zeros_initializer,
        dtype=self.dtype,
        trainable=False)
    self.built = True

  def compute_mask(self, inputs, mask=None):
    return math_ops.is_finite(inputs)

  def call(self, inputs):
    mask = math_ops.is_finite(inputs)
    outputs = math_ops.multiply_no_nan(inputs, math_ops.cast(mask, inputs.dtype))
    outputs = outputs * math_ops.cast(self.kernel, inputs.dtype) + math_ops.cast(self.bias, inputs.dtype)

    # Compute the mask and outputs simultaneously.
    outputs._keras_mask = mask
    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'axis': self.axis}
    base_config = super(Normalisa, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
