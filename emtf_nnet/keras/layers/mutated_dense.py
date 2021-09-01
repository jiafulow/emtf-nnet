# The following source code was originally obtained from:
# https://github.com/keras-team/keras/blob/r2.6/keras/layers/core.py#L1066-L1270
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
"""Keras dense layers."""

import tensorflow.compat.v2 as tf

from keras.layers.core import Dense


class MutatedDense(Dense):
  """Dense layer with correction to the gradient."""

  def __init__(self,
               units,
               **kwargs):
    super().__init__(units=units, **kwargs)
    self.supports_masking = True
    self._compute_output_and_mask_jointly = True

  def _dense(self, inputs, corr, kernel, bias=None, activation=None, dtype=None):
    if dtype:
      if inputs.dtype.base_dtype != dtype.base_dtype:
        inputs = tf.cast(inputs, dtype)
      if corr.dtype.base_dtype != dtype.base_dtype:
        corr = tf.cast(corr, dtype)

    rank = inputs.shape.rank
    if rank == 2:
      # Apply correction to the gradient while keeping the same outputs.
      # f(x) = x * stop[gx] + stop[fx - x * gx]
      #      = stop[fx] + ((x - stop[x]) * stop[gx])
      #      = stop[fx] + 0
      # g(x) = stop[gx] + grad[stop[fx - x * gx]]
      #      = stop[gx] + 0
      outputs = tf.raw_ops.AddV2(
          x=tf.raw_ops.MatMul(a=tf.raw_ops.Mul(x=inputs, y=tf.stop_gradient(corr)), b=kernel),
          y=-tf.stop_gradient(tf.raw_ops.MatMul(a=tf.raw_ops.Mul(x=inputs, y=corr), b=kernel)))
      outputs = tf.raw_ops.AddV2(
          x=outputs,
          y=tf.stop_gradient(tf.raw_ops.MatMul(a=inputs, b=kernel)))
    else:
      raise ValueError('inputs must be rank 2.')

    if bias is not None:
      outputs = tf.nn.bias_add(outputs, bias)

    if activation is not None:
      outputs = activation(outputs)
    return outputs

  def call(self, inputs, training=None, mask=None):
    # Returns Dense(x) with a correction to the gradient
    if mask is None:
      mask = tf.math.is_finite(inputs)
    mask = tf.cast(mask, inputs.dtype)
    mean = tf.math.reduce_mean(mask, axis=0)  # reduce along the batch dimension
    corr = tf.math.reciprocal_no_nan(mean)  # corr = 1/mean
    outputs = self._dense(
        inputs * mask,
        corr,
        self.kernel,
        bias=self.bias,
        activation=self.activation,
        dtype=self._compute_dtype_object)

    # Compute the mask and outputs simultaneously.
    outputs._keras_mask = tf.math.is_finite(outputs)
    return outputs
