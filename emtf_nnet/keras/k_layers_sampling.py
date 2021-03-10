# The following source code is obtained from:
# https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/layers/core.py
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

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.keras.layers.core import Layer


class Sampling(Layer):
  """Applies a normal sampling function to an output.
  """
  def __init__(self, scale_w=1.0, scale_b=0.0, epsilon=1e-7, seed=None, kl_weight=None, **kwargs):
    super(Sampling, self).__init__(**kwargs)
    self.scale_w = scale_w
    self.scale_b = scale_b
    self.epsilon = epsilon
    self.seed = seed
    self.kl_weight = kl_weight

  def build(self, input_shape):
    self._scale_w = self.add_weight(shape=(1,),
                                    initializer=initializers.Constant(self.scale_w),
                                    trainable=False)
    self._scale_b = self.add_weight(shape=(1,),
                                    initializer=initializers.Constant(self.scale_b),
                                    trainable=False)

  # From:
  #   https://github.com/tensorflow/probability/blob/v0.12.1/tensorflow_probability/python/distributions/normal.py#L266-L284
  #   https://github.com/tensorflow/probability/blob/v0.12.1/tensorflow_probability/python/layers/dense_variational_v2.py#L165-L179
  # Reduces to 0.5 * (- 1 - log(a_scale^2) + a_loc^2 + a_scale^2) if b_loc = 0 and b_scale = 1
  def _kl_divergence_fn(self, a_loc, a_scale, b_loc, b_scale):
    """Calculate the batched KL divergence KL(a || b) with a and b Normal."""
    diff_log_scale = math_ops.log(a_scale) - math_ops.log(b_scale)
    kl = (
        0.5 * math_ops.squared_difference(a_loc / b_scale, b_loc / b_scale) +
        0.5 * math_ops.expm1(2. * diff_log_scale) -
        diff_log_scale)
    if self.kl_weight is not None:
      kl = math_ops.cast(self.kl_weight, dtype=kl.dtype) * kl
    return K.mean(K.sum(kl, axis=-1))

  def call(self, inputs, training=None):
    #if training is None:
    #  training = K.learning_phase()
    loc = inputs[..., :1]
    scale = inputs[..., 1:]
    scale = math_ops.maximum(nn_ops.softplus((self._scale_w * scale) + self._scale_b), self.epsilon)

    # Add KL divergence regularization loss.
    loc_prior = array_ops.zeros_like(loc, dtype=loc.dtype)
    scale_prior = array_ops.ones_like(scale, dtype=scale.dtype)
    self.add_loss(self._kl_divergence_fn(loc, scale, loc_prior, scale_prior))

    sampled = random_ops.random_normal(shape=array_ops.shape(loc), seed=self.seed, dtype=inputs.dtype)
    return K.in_train_phase(sampled * scale + loc, loc, training=training)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'scale_w': self.scale_w, 'scale_b': self.scale_b, 'epsilon': self.epsilon, 'seed': self.seed, 'kl_weight': self.kl_weight}
    base_config = super(Sampling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
