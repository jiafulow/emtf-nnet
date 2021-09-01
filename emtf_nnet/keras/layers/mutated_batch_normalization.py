# The following source code was originally obtained from:
# https://github.com/keras-team/keras/blob/r2.6/keras/layers/normalization/batch_normalization.py#L30-L947
# https://github.com/keras-team/keras/blob/r2.6/keras/layers/normalization/batch_normalization.py#L1112-L1253
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
"""Keras normalization layers."""

import tensorflow.compat.v2 as tf

from keras.utils import control_flow_util
from keras.layers.normalization.batch_normalization import BatchNormalization


class MutatedBatchNormalization(BatchNormalization):
  """Batch normalization layer with simplified call()."""

  def __init__(self,
               axis=-1,
               **kwargs):
    super().__init__(axis=axis, **kwargs)
    assert self._USE_V2_BEHAVIOR
    assert not self.renorm
    assert self.virtual_batch_size is None
    assert self.adjustment is None
    assert self.fused is None

  def build(self, input_shape):
    super().build(input_shape)
    assert self.built
    assert not self.fused

  def call(self, inputs, training=None):
    training = self._get_training_value(training)

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.shape
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.axis]

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = control_flow_util.constant_value(training)
    if training_value is not None and bool(training_value) is False:
      mean, variance = self.moving_mean, self.moving_variance
    else:
      keep_dims = False
      mean, variance = tf.nn.moments(
          tf.cast(inputs, self._param_dtype),
          reduction_axes,
          keepdims=keep_dims)
      mean = control_flow_util.smart_cond(
          training,
          lambda: mean,
          lambda: tf.convert_to_tensor(self.moving_mean))
      variance = control_flow_util.smart_cond(
          training,
          lambda: variance,
          lambda: tf.convert_to_tensor(self.moving_variance))

      def _do_update(var, value):
        input_batch_size = None
        return self._assign_moving_average(var, value, self.momentum,
                                           input_batch_size)

      def _fake_update(var):
        return tf.identity(var)

      def mean_update():
        return control_flow_util.smart_cond(
            training,
            lambda: _do_update(self.moving_mean, mean),
            lambda: _fake_update(self.moving_mean))

      def variance_update():
        return control_flow_util.smart_cond(
            training,
            lambda: _do_update(self.moving_variance, variance),
            lambda: _fake_update(self.moving_variance))

      self.add_update(mean_update)
      self.add_update(variance_update)

    # Get gamma and beta
    scale, offset = self.gamma, self.beta

    mean = tf.cast(mean, inputs.dtype)
    variance = tf.cast(variance, inputs.dtype)
    if offset is not None:
      offset = tf.cast(offset, inputs.dtype)
    if scale is not None:
      scale = tf.cast(scale, inputs.dtype)
    outputs = tf.nn.batch_normalization(inputs, mean, variance, offset, scale,
                                        self.epsilon)
    return outputs
