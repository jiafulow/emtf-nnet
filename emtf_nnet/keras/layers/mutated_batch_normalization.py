# The following source code was originally obtained from:
# https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/layers/normalization.py
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
"""Normalization layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization


class MutatedBatchNormalization(BatchNormalization):
  """Batch normalization layer."""

  def __init__(self, axis=-1, **kwargs):
    super(MutatedBatchNormalization, self).__init__(axis=axis, **kwargs)
    assert self._USE_V2_BEHAVIOR
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
      mean, variance = nn.moments(inputs, reduction_axes, keep_dims=False)
      mean = control_flow_util.smart_cond(
          training,
          lambda: mean,
          lambda: ops.convert_to_tensor_v2_with_dispatch(self.moving_mean))
      variance = control_flow_util.smart_cond(
          training,
          lambda: variance,
          lambda: ops.convert_to_tensor_v2_with_dispatch(self.moving_variance))

      def _do_update(var, value):
        input_batch_size = None
        return self._assign_moving_average(
            var, value, self.momentum, input_batch_size)

      def _fake_update(var):
        return array_ops.identity(var)

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

    mean = math_ops.cast(mean, inputs.dtype)
    variance = math_ops.cast(variance, inputs.dtype)
    if offset is not None:
      offset = math_ops.cast(offset, inputs.dtype)
    if scale is not None:
      scale = math_ops.cast(scale, inputs.dtype)
    outputs = nn.batch_normalization(inputs, mean, variance, offset, scale,
                                     self.epsilon)
    return outputs
