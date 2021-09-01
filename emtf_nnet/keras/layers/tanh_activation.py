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


class TanhActivation(Layer):
  """An implementation of scaled tanh activation."""

  def __init__(self, alpha=None, beta=None, **kwargs):
    super().__init__(**kwargs)
    self.supports_masking = True
    self.alpha = alpha
    self.beta = beta

  def call(self, inputs):
    x = inputs
    if self.alpha is not None:
      alpha = tf.cast(self.alpha, x.dtype)
    else:
      alpha = tf.cast(1., x.dtype)
    if self.beta is not None:
      beta = tf.cast(self.beta, x.dtype)
    else:
      beta = tf.cast(1., x.dtype)
    return tf.math.tanh(x * alpha) * beta

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = super().get_config()
    config.update({
        'alpha': self.alpha,
        'beta': self.beta,
    })
    return config
