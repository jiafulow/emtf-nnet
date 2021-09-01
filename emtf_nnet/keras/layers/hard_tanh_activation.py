# The following source code was originally obtained from:
# https://github.com/keras-team/keras/blob/r2.6/keras/layers/core.py#L388-L430
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

import tensorflow.compat.v2 as tf

from keras.engine.base_layer import Layer


class HardTanhActivation(Layer):
  """An implementation of piecewise linear tanh activation composed of ReLU."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.supports_masking = True

  def call(self, inputs):
    x = inputs
    pieces = [
      x,
      tf.nn.relu(x - tf.cast(0.5, x.dtype)) * tf.cast(-0.5, x.dtype),
      tf.nn.relu(x - tf.cast(1.0, x.dtype)) * tf.cast(-0.25, x.dtype),
      tf.nn.relu(x - tf.cast(1.5, x.dtype)) * tf.cast(-0.125, x.dtype),
      tf.nn.relu(x - tf.cast(2.0, x.dtype)) * tf.cast(-0.0625, x.dtype),
      tf.nn.relu(x - tf.cast(2.5, x.dtype)) * tf.cast(-0.03125, x.dtype),
      #tf.nn.relu(x - tf.cast(3.0, x.dtype)) * tf.cast(-0.015625, x.dtype),
      #tf.nn.relu(x - tf.cast(3.5, x.dtype)) * tf.cast(-0.0078125, x.dtype),
      #tf.nn.relu(x - tf.cast(4.0, x.dtype)) * tf.cast(-0.00390625, x.dtype),
      tf.nn.relu(-x - tf.cast(0.5, x.dtype)) * tf.cast(0.5, x.dtype),
      tf.nn.relu(-x - tf.cast(1.0, x.dtype)) * tf.cast(0.25, x.dtype),
      tf.nn.relu(-x - tf.cast(1.5, x.dtype)) * tf.cast(0.125, x.dtype),
      tf.nn.relu(-x - tf.cast(2.0, x.dtype)) * tf.cast(0.0625, x.dtype),
      tf.nn.relu(-x - tf.cast(2.5, x.dtype)) * tf.cast(0.03125, x.dtype),
      #tf.nn.relu(-x - tf.cast(3.0, x.dtype)) * tf.cast(0.015625, x.dtype),
      #tf.nn.relu(-x - tf.cast(3.5, x.dtype)) * tf.cast(0.0078125, x.dtype),
      #tf.nn.relu(-x - tf.cast(4.0, x.dtype)) * tf.cast(0.00390625, x.dtype),
    ]
    return tf.math.add_n(pieces)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = super().get_config()
    return config
