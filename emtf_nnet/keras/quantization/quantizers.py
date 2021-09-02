# The following source code was originally obtained from:
# https://github.com/tensorflow/model-optimization/blob/v0.7.0/tensorflow_model_optimization/python/core/quantization/keras/quantizers.py
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/fake_quant_ops.cc
# ==============================================================================

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Quantizer classes which implement quantization using TF Ops on a tensor."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quant_ops
from tensorflow_model_optimization.python.core.quantization.keras import quantizers


class FixedRangeQuantizer(quantizers.Quantizer):
  """Quantize tensor in a fixed range."""

  def __init__(self,
               num_bits,
               num_int_bits,
               per_axis=False,
               symmetric=False,
               narrow_range=False):
    self.num_bits = num_bits
    self.num_int_bits = num_int_bits
    self.per_axis = per_axis
    self.symmetric = symmetric  # unused
    self.narrow_range = narrow_range

  def build(self, tensor_shape, name, layer):
    quant_min = 1 if self.narrow_range else 0
    quant_max = (1 << self.num_bits) - 1
    zero_point = (quant_max - quant_min + 1) // 2
    zero_point_from_min = quant_min + zero_point
    range_min = quant_min - zero_point_from_min
    range_max = quant_max - zero_point_from_min
    range_min /= (1 << (self.num_bits - self.num_int_bits))
    range_max /= (1 << (self.num_bits - self.num_int_bits))
    min_weight = layer.add_weight(
        name + '_min',
        initializer=tf.keras.initializers.Constant(range_min),
        trainable=False)
    max_weight = layer.add_weight(
        name + '_max',
        initializer=tf.keras.initializers.Constant(range_max),
        trainable=False)
    return {'min_var': min_weight, 'max_var': max_weight}

  def _quantize(self,
                inputs,
                min_var,
                max_var,
                per_channel=False,
                name_prefix='FixedRangeQuantize',
                is_training=True,
                num_bits=8,
                narrow_range=False,
                symmetric=False):
    with tf.name_scope(name_prefix):
      return quant_ops._FakeQuantWithMinMaxVars(
          inputs,
          min_var,
          max_var,
          per_channel=per_channel,
          num_bits=num_bits,
          narrow_range=narrow_range)

  def __call__(self, inputs, training, weights, **kwargs):
    return self._quantize(
        inputs,
        weights['min_var'],
        weights['max_var'],
        is_training=training,
        num_bits=self.num_bits,
        per_channel=self.per_axis,
        symmetric=self.symmetric,
        narrow_range=self.narrow_range)

  def get_config(self):
    config = {
        'num_bits': self.num_bits,
        'num_int_bits': self.num_int_bits,
        'per_axis': self.per_axis,
        'symmetric': self.symmetric,
        'narrow_range': self.narrow_range,
    }
    return config

  def __eq__(self, other):
    if not isinstance(other, FixedRangeQuantizer):
      return False

    return (self.num_bits == other.num_bits and
            self.num_int_bits == other.num_int_bits and
            self.per_axis == other.per_axis and
            self.symmetric == other.symmetric and
            self.narrow_range == other.narrow_range)

  def __ne__(self, other):
    return not self.__eq__(other)
