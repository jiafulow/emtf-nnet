# The following source code was originally obtained from:
# https://github.com/keras-team/keras/blob/r2.6/keras/optimizer_v2/optimizer_v2.py#L452-L455
# https://github.com/keras-team/keras/blob/r2.6/keras/optimizer_v2/adam.py#L23-L243
# https://github.com/keras-team/keras/blob/r2.6/keras/optimizer_v2/learning_rate_schedule.py#L90-L192
# https://github.com/keras-team/keras/blob/r2.6/keras/optimizer_v2/learning_rate_schedule.py#L547-L638
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
"""Optimizers and learning rate decay functions."""

import tensorflow.compat.v2 as tf

import numpy as np

from keras import backend
from keras.optimizer_v2 import adam
from keras.optimizer_v2 import learning_rate_schedule


class Adamu(adam.Adam):
  """Optimizer that implements the Adam algorithm."""

  def __init__(self, **kwargs):
    super(Adamu, self).__init__(**kwargs)
    self._set_hyper('gradient_maxnorm', 0.)

  def _get_gradients(self, tape, loss, var_list, grad_loss=None):
    """Called in `minimize` to compute gradients from loss."""
    grads = tape.gradient(loss, var_list, grad_loss)

    def l2norm_fn(t):
      # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
      l2sum = tf.math.reduce_sum(t * t, axis=None, keepdims=True)
      pred = l2sum > 0
      # Two-tap tf.where trick to bypass NaN gradients
      l2sum_safe = tf.where(pred, l2sum, tf.ones_like(l2sum))
      l2norm = tf.where(pred, tf.math.sqrt(l2sum_safe), l2sum)
      return l2norm

    # Find the max L2-norm
    gradient_maxnorm = tf.math.reduce_max(
        tf.stack([tf.math.reduce_max(tf.stack(l2norm_fn(g))) for g in grads]))
    hyper = self._get_hyper('gradient_maxnorm')
    hyper.assign(gradient_maxnorm)

    # Return grads_and_vars
    return list(zip(grads, var_list))


class WarmupCosineDecay(learning_rate_schedule.CosineDecay):
  """A LearningRateSchedule that uses a cosine decay schedule."""

  def __init__(self,
               initial_learning_rate,
               warmup_steps,
               decay_steps,
               alpha=0.0,
               name=None):
    super(WarmupCosineDecay, self).__init__(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=alpha,
        name=name)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    with tf.name_scope(self.name or "WarmupCosineDecay") as name:
      initial_learning_rate = tf.convert_to_tensor(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      warmup_steps = tf.cast(self.warmup_steps, dtype)
      decay_steps = tf.cast(self.decay_steps, dtype)
      alpha = tf.cast(self.alpha, dtype)

      global_step_recomp = tf.cast(step, dtype)
      fraction = tf.math.divide_no_nan(global_step_recomp, warmup_steps)
      fraction = tf.math.maximum(fraction, backend.epsilon())
      warmup_learning_rate = tf.math.multiply(
          initial_learning_rate, fraction)

      global_step_recomp = tf.cast(step - self.warmup_steps, dtype)
      global_step_recomp = tf.math.minimum(global_step_recomp, decay_steps)
      fraction = tf.math.divide_no_nan(global_step_recomp, decay_steps)
      cosine_decayed = 0.5 * (1.0 + tf.math.cos(
          tf.constant(np.pi, dtype=dtype) * fraction))

      decayed = (1.0 - alpha) * cosine_decayed + alpha
      learning_rate = tf.math.multiply(initial_learning_rate, decayed)
      return tf.cond(
          step < self.warmup_steps,
          lambda: warmup_learning_rate,
          lambda: learning_rate,
          name=name)

  def get_config(self):
    config = super(WarmupCosineDecay, self).get_config()
    config.update({'warmup_steps': self.warmup_steps})
    return config


class WarmupExponentialDecay(learning_rate_schedule.ExponentialDecay):
  """A LearningRateSchedule that uses an exponential decay schedule."""

  def __init__(self,
               initial_learning_rate,
               warmup_steps,
               decay_steps,
               decay_rate,
               staircase=False,
               name=None):
    super(WarmupExponentialDecay, self).__init__(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase,
        name=name)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    with tf.name_scope(self.name or "WarmupExponentialDecay") as name:
      initial_learning_rate = tf.convert_to_tensor(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      warmup_steps = tf.cast(self.warmup_steps, dtype)
      decay_steps = tf.cast(self.decay_steps, dtype)
      decay_rate = tf.cast(self.decay_rate, dtype)

      global_step_recomp = tf.cast(step, dtype)
      fraction = tf.math.divide_no_nan(global_step_recomp, warmup_steps)
      fraction = tf.math.maximum(fraction, backend.epsilon())
      warmup_learning_rate = tf.math.multiply(
          initial_learning_rate, fraction)

      global_step_recomp = tf.cast(step - self.warmup_steps, dtype)
      p = tf.math.divide_no_nan(global_step_recomp, decay_steps)
      if self.staircase:
        p = tf.math.floor(p)
      learning_rate = tf.math.multiply(
          initial_learning_rate, tf.math.pow(decay_rate, p))
      learning_rate = tf.math.maximum(learning_rate, 1e-5)  # bounded below at 1e-5
      return tf.cond(
          step < self.warmup_steps,
          lambda: warmup_learning_rate,
          lambda: learning_rate,
          name=name)

  def get_config(self):
    config = super(WarmupExponentialDecay, self).get_config()
    config.update({'warmup_steps': self.warmup_steps})
    return config
