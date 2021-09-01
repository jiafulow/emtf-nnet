# The following source code was originally obtained from:
# https://github.com/keras-team/keras/blob/r2.6/keras/layers/preprocessing/normalization.py#L27-L282
# https://github.com/keras-team/keras/blob/r2.6/keras/layers/core.py#L55-L119
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

import tensorflow.compat.v2 as tf

from keras.engine.base_layer import Layer


class FeatureNormalization(Layer):
  """Feature-wise normalization of the data."""

  def __init__(self,
               axis=-1,
               **kwargs):
    super().__init__(**kwargs)
    self.supports_masking = True
    self._compute_output_and_mask_jointly = True

    # Standardize `axis` to a tuple.
    if axis is None:
      axis = ()
    elif isinstance(axis, int):
      axis = (axis,)
    else:
      axis = tuple(axis)
    self.axis = axis

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    weight_shape = tuple(input_shape[d] for d in self.axis)
    self.scale = self.add_weight(
        name='scale',
        shape=weight_shape,
        dtype=self.dtype,
        initializer='ones',
        trainable=False)
    self.offset = self.add_weight(
        name='offset',
        shape=weight_shape,
        dtype=self.dtype,
        initializer='zeros',
        trainable=False)
    self.built = True

  def compute_mask(self, inputs, mask=None):
    return tf.math.is_finite(inputs)

  def call(self, inputs):
    inputs = tf.convert_to_tensor(inputs)
    if inputs.dtype != self.dtype:
      inputs = tf.cast(inputs, self.dtype)

    mask = tf.math.is_finite(inputs)
    outputs = tf.math.multiply_no_nan(inputs, tf.cast(mask, inputs.dtype))
    outputs = (outputs * tf.cast(self.scale, inputs.dtype) +
               tf.cast(self.offset, inputs.dtype))

    # Compute the mask and outputs simultaneously.
    outputs._keras_mask = mask
    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def compute_output_signature(self, input_spec):
    return input_spec

  def get_config(self):
    config = super().get_config()
    config.update({
        'axis': self.axis,
    })
    return config
