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
from tensorflow.python.keras.engine.base_layer import Layer


class TanhActivation(Layer):
  """An implementation of scaled tanh activation."""

  def __init__(self, alpha=1., beta=1., **kwargs):
    super(TanhActivation, self).__init__(**kwargs)
    self.supports_masking = True
    self.alpha = alpha
    self.beta = beta

  def call(self, inputs):
    x = inputs
    if self.alpha is not None:
      alpha = constant_op.constant(self.alpha, dtype=x.dtype)
    else:
      alpha = constant_op.constant(1., dtype=x.dtype)
    if self.beta is not None:
      beta = constant_op.constant(self.beta, dtype=x.dtype)
    else:
      beta = constant_op.constant(1., dtype=x.dtype)
    return math_ops.tanh(x * alpha) * beta

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'alpha': self.alpha, 'beta': self.beta}
    base_config = super(TanhActivation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
