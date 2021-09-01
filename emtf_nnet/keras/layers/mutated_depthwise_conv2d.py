# The following source code was originally obtained from:
# https://github.com/keras-team/keras/blob/r2.6/keras/layers/convolutional.py#L2083-L2254
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
"""Keras convolution layers."""

import tensorflow.compat.v2 as tf

from keras.utils import conv_utils
from keras.layers.convolutional import SeparableConv2D


class MutatedDepthwiseConv2D(SeparableConv2D):
  """Depthwise 2D convolution layer that inherits from SeparableConv2D."""

  def __init__(self,
               kernel_size,
               use_bias=False,
               **kwargs):
    super().__init__(filters=1, kernel_size=kernel_size, use_bias=use_bias, **kwargs)

  def call(self, inputs):
    # Apply the actual ops.
    if self.data_format == 'channels_last':
      strides = (1,) + self.strides + (1,)
    else:
      strides = (1, 1) + self.strides
    outputs = tf.nn.depthwise_conv2d(
        inputs,
        self.depthwise_kernel,
        strides=strides,
        padding=self.padding.upper(),
        data_format=conv_utils.convert_data_format(self.data_format, ndim=4),
        dilations=self.dilation_rate)

    if self.use_bias:
      outputs = tf.nn.bias_add(
          outputs,
          self.bias,
          data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    if self.activation is not None:
      return self.activation(outputs)
    return outputs
