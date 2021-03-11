# The following source code was originally obtained from:
# https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/layers/preprocessing/normalization.py#L51-L227
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
"""Normalization preprocessing layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.engine.base_layer import Layer


class FeatureNormalization(Layer):
  """Feature-wise normalization of the data."""

  def __init__(self, axis=-1, **kwargs):
    super(FeatureNormalization, self).__init__(**kwargs)
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
    self.scale = self.add_weight(
        'scale',
        shape=weight_shape,
        dtype=self.dtype,
        initializer=init_ops.ones_initializer,
        trainable=False)
    self.offset = self.add_weight(
        'offset',
        shape=weight_shape,
        dtype=self.dtype,
        initializer=init_ops.zeros_initializer,
        trainable=False)
    self.built = True

  def compute_mask(self, inputs, mask=None):
    return math_ops.is_finite(inputs)

  def call(self, inputs):
    inputs = ops.convert_to_tensor_v2_with_dispatch(inputs)
    if inputs.shape.rank == 1:
      inputs = array_ops.expand_dims_v2(inputs, 1)
    # If the inputs are not floats, cast them to floats. This avoids issues
    # with int-float multiplication and division below.
    if inputs.dtype != K.floatx():
      inputs = math_ops.cast(inputs, K.floatx())

    mask = math_ops.is_finite(inputs)
    outputs = math_ops.multiply_no_nan(inputs, math_ops.cast(mask, inputs.dtype))
    outputs = (outputs * math_ops.cast(self.scale, inputs.dtype) +
               math_ops.cast(self.offset, inputs.dtype))

    # Compute the mask and outputs simultaneously.
    outputs._keras_mask = mask
    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'axis': self.axis}
    base_config = super(FeatureNormalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
