# The following source code was originally obtained from:
# https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/quantize_layer.py
# ==============================================================================

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Keras Layer which quantizes tensors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class QuantizableLayer(tf.keras.layers.Layer):
  """Placeholder layer for quantization of tensors passed through the layer.

  Quantization occurs when it is given a QuantizeConfig and wrapped by QuantizeWrapper.
  """

  def __init__(self, **kwargs):
    super(QuantizableLayer, self).__init__(**kwargs)
    self.supports_masking = True

  def call(self, inputs):
    return inputs

  def compute_output_shape(self, input_shape):
    return input_shape
