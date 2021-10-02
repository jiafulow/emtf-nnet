# The following source code was originally obtained from:
# https://github.com/keras-team/keras/blob/r2.6/keras/layers/core.py#L388-L430
# https://github.com/keras-team/keras/blob/r2.6/keras/layers/preprocessing/image_preprocessing.py#L314-L361
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


class ScaleActivation(Layer):
  """Applies scale activation function to an output."""

  def __init__(self, scale, offset=0., **kwargs):
    self.scale = scale
    self.offset = offset
    super().__init__(**kwargs)
    self.supports_masking = True

  def call(self, inputs):
    dtype = self._compute_dtype
    scale = tf.cast(self.scale, dtype)
    offset = tf.cast(self.offset, dtype)
    return tf.cast(inputs, dtype) * scale + offset

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = super().get_config()
    config.update({
        'scale': self.scale,
        'offset': self.offset,
    })
    return config
