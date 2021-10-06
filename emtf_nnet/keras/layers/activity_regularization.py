# The following source code was originally obtained from:
# https://github.com/keras-team/keras/blob/r2.6/keras/layers/core.py#L1274-L1303
# https://github.com/keras-team/keras/blob/r2.6/keras/engine/base_layer.py#L2458-L2471
# https://github.com/tensorflow/probability/blob/r0.14/tensorflow_probability/python/math/numeric.py#L68-L100
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
"""Layers that add regularization losses."""

import itertools
import numpy as np

import tensorflow.compat.v2 as tf

from keras.engine.base_layer import Layer


def clip_by_value_preserve_gradient(t, clip_value_min, clip_value_max,
                                    name=None):
  """Clips values to a specified min and max while leaving gradient unaltered.
  """
  clip_t = tf.clip_by_value(t, clip_value_min, clip_value_max, name=name)
  return t + tf.stop_gradient(clip_t - t)


class ActivityRegularizationLoss(object):
  """Computes the regularization loss function."""

  def __init__(self):
    self._model = None
    self._data = None
    self._enter_dunder_call = False

  def set_model(self, model):
    assert isinstance(model, tf.keras.Model)
    self._model = model

  def set_data(self, data, batch_size=32):
    assert isinstance(data, tf.Tensor)
    self._data = data
    # Implements a simple data iterator that yields a slice object.
    # Skips the last batch which could be a partial batch.
    num_samples = data.shape[0]
    num_batches = int(np.ceil(num_samples / float(batch_size)))
    self._iterator = itertools.cycle([
        slice(i * batch_size, min(num_samples, (i + 1) * batch_size))
        for i in range(num_batches - 1)
    ])

  def __call__(self, x, l1=0.01, bias=2.0):
    # This is needed to avoid recursion
    self._enter_dunder_call = True

    l1 = tf.cast(l1, dtype=x.dtype)
    bias = tf.cast(bias, dtype=x.dtype)
    zero = tf.constant(0., dtype=x.dtype)
    clip_value_min_1 = tf.constant(1e-3, dtype=x.dtype)
    clip_value_max_1 = tf.constant(1e3, dtype=x.dtype)
    clip_value_min_2 = tf.constant(0., dtype=x.dtype)
    clip_value_max_2 = tf.constant(256., dtype=x.dtype)

    loss = tf.constant(0., dtype=x.dtype)
    if self._model is not None and self._data is not None:
      x = self._data[next(self._iterator)]
      x = tf.convert_to_tensor(x)
      y = self._model(x, training=False)
      # Take absolute value and clip
      abs_y = tf.where(y < zero, -y, y)
      new_abs_y = clip_by_value_preserve_gradient(abs_y,
                                                  clip_value_min_1,
                                                  clip_value_max_1)
      # Find reciprocal
      y_as_pt = tf.math.reciprocal_no_nan(new_abs_y)
      # Shift and clip
      new_y_as_pt = clip_by_value_preserve_gradient(y_as_pt - bias,
                                                    clip_value_min_2,
                                                    clip_value_max_2)
      # Compute mean
      mean_activity_loss = l1 * tf.math.reduce_mean(new_y_as_pt)
      loss += mean_activity_loss

    self._enter_dunder_call = False
    return loss


class ActivityRegularization(Layer):
  """Layer that applies an update to the cost function based on activity."""

  def __init__(self, l1=0.01, bias=2.0, **kwargs):
    super().__init__(**kwargs)
    self.supports_masking = True
    self.l1 = l1
    self.bias = bias
    self.loss_fn = ActivityRegularizationLoss()

  def set_model(self, model):
    self.loss_fn.set_model(model)

  def set_data(self, data, batch_size=32):
    self.loss_fn.set_data(data, batch_size=batch_size)

  def call(self, inputs):
    # This is needed to avoid recursion
    if not self.loss_fn._enter_dunder_call:
      loss = self.loss_fn(inputs, l1=self.l1, bias=self.bias)
      self.add_loss(loss)
    return inputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = super().get_config()
    config.update({
        'l1': self.l1,
        'bias': self.bias,
    })
    return config
