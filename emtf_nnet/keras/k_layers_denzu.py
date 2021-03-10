# The following source code is obtained from:
# https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/layers/core.py#L1081-L1247
# https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/layers/ops/core.py
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

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.layers.core import Dense


class Denzu(Dense):
  def __init__(self, units, **kwargs):
    super(Denzu, self).__init__(units=units, **kwargs)
    self.supports_masking = True
    self._compute_output_and_mask_jointly = True

  def _dense(self, inputs, corr, one_minus_corr, kernel, bias=None, activation=None, dtype=None):
    rank = inputs.shape.rank
    if rank == 2:
      outputs = gen_math_ops.add_v2(
          gen_math_ops.mat_mul(inputs * corr, kernel),
          array_ops.stop_gradient(gen_math_ops.mat_mul(inputs * one_minus_corr, kernel)))
    else:
      raise ValueError('inputs must be rank 2.')

    if bias is not None:
      outputs = nn_ops.bias_add(outputs, bias)

    if activation is not None:
      outputs = activation(outputs)

    # Compute the mask and outputs simultaneously.
    outputs._keras_mask = math_ops.is_finite(outputs)
    return outputs

  def call(self, inputs, training=None, mask=None):
    # Calculate (x * mask) / mean + (x * mask) * (1 - (1 / mean))
    if mask is None:
      mask = math_ops.is_finite(inputs)
    mask = math_ops.cast(mask, inputs.dtype)
    mean = math_ops.reduce_mean(mask, axis=0)  # reduce along the batch dimension
    corr = math_ops.reciprocal(math_ops.maximum(mean, 1e-4))  # include epsilon to avoid division by 0
    one_minus_corr = constant_op.constant(1., dtype=inputs.dtype) - corr
    return self._dense(
        inputs * mask,
        corr,
        one_minus_corr,
        self.kernel,
        self.bias,
        self.activation,
        dtype=self._compute_dtype_object)
