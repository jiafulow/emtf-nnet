# The following source code was originally obtained from:
# https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/layers/core.py#L407-L448
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
"""Layers that act as activation functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.keras.engine.base_layer import Layer


class HardTanhActivation(Layer):
  """An implementation of piecewise linear tanh activation composed of ReLU."""

  def __init__(self, **kwargs):
    super(HardTanhActivation, self).__init__(**kwargs)
    self.supports_masking = True

  def call(self, inputs):
    x = inputs
    pieces = [
      x,
      nn.relu(x - constant_op.constant(0.5, dtype=x.dtype)) * constant_op.constant(-0.5, dtype=x.dtype),
      nn.relu(x - constant_op.constant(1.0, dtype=x.dtype)) * constant_op.constant(-0.25, dtype=x.dtype),
      nn.relu(x - constant_op.constant(1.5, dtype=x.dtype)) * constant_op.constant(-0.125, dtype=x.dtype),
      nn.relu(x - constant_op.constant(2.0, dtype=x.dtype)) * constant_op.constant(-0.0625, dtype=x.dtype),
      nn.relu(x - constant_op.constant(2.5, dtype=x.dtype)) * constant_op.constant(-0.03125, dtype=x.dtype),
      #nn.relu(x - constant_op.constant(3.0, dtype=x.dtype)) * constant_op.constant(-0.015625, dtype=x.dtype),
      #nn.relu(x - constant_op.constant(3.5, dtype=x.dtype)) * constant_op.constant(-0.0078125, dtype=x.dtype),
      #nn.relu(x - constant_op.constant(4.0, dtype=x.dtype)) * constant_op.constant(-0.00390625, dtype=x.dtype),
      nn.relu(-x - constant_op.constant(0.5, dtype=x.dtype)) * constant_op.constant(0.5, dtype=x.dtype),
      nn.relu(-x - constant_op.constant(1.0, dtype=x.dtype)) * constant_op.constant(0.25, dtype=x.dtype),
      nn.relu(-x - constant_op.constant(1.5, dtype=x.dtype)) * constant_op.constant(0.125, dtype=x.dtype),
      nn.relu(-x - constant_op.constant(2.0, dtype=x.dtype)) * constant_op.constant(0.0625, dtype=x.dtype),
      nn.relu(-x - constant_op.constant(2.5, dtype=x.dtype)) * constant_op.constant(0.03125, dtype=x.dtype),
      #nn.relu(-x - constant_op.constant(3.0, dtype=x.dtype)) * constant_op.constant(0.015625, dtype=x.dtype),
      #nn.relu(-x - constant_op.constant(3.5, dtype=x.dtype)) * constant_op.constant(0.0078125, dtype=x.dtype),
      #nn.relu(-x - constant_op.constant(4.0, dtype=x.dtype)) * constant_op.constant(0.00390625, dtype=x.dtype),
    ]
    return math_ops.add_n(pieces)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {}
    base_config = super(HardTanhActivation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
