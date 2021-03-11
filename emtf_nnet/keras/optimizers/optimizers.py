# The following source code was originally obtained from:
# https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/optimizer_v2/optimizer_v2.py#L439-L442
# https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/optimizer_v2/adam.py#L34-L253
# https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/optimizer_v2/learning_rate_schedule.py#L65-L166
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import CosineDecay
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay


class AdamOptim(Adam):
  """Optimizer that implements the Adam algorithm."""

  def __init__(self, **kwargs):
    super(AdamOptim, self).__init__(**kwargs)
    self._set_hyper('grads_maxnorm', 0.)

  def _get_gradients(self, tape, loss, var_list, grad_loss=None):
    """Called in `minimize` to compute gradients from loss."""
    grads = tape.gradient(loss, var_list, grad_loss)

    def l2norm_fn(t):
      # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
      l2sum = math_ops.reduce_sum(t * t, axis=None, keepdims=True)
      pred = l2sum > 0
      # Two-tap tf.where trick to bypass NaN gradients
      l2sum_safe = array_ops.where(pred, l2sum, array_ops.ones_like(l2sum))
      l2norm = array_ops.where(pred, math_ops.sqrt(l2sum_safe), l2sum)
      return l2norm

    # Find the max L2-norm
    grads_maxnorm = math_ops.reduce_max(array_ops.stack([
        math_ops.reduce_max(array_ops.stack(l2norm_fn(g))) for g in grads]))
    self._get_hyper('grads_maxnorm').assign(grads_maxnorm)

    # Return grads_and_vars
    return list(zip(grads, var_list))


class WarmupCosineDecay(CosineDecay):
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
    with ops.name_scope_v2(self.name or "WarmupCosineDecay") as name:
      initial_learning_rate = ops.convert_to_tensor_v2_with_dispatch(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      warmup_steps = math_ops.cast(self.warmup_steps, dtype)
      decay_steps = math_ops.cast(self.decay_steps, dtype)
      alpha = math_ops.cast(self.alpha, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      p = math_ops.div_no_nan(global_step_recomp, warmup_steps)
      p = math_ops.maximum(p, K.epsilon())
      warmup_learning_rate = math_ops.multiply(
          initial_learning_rate, p)

      global_step_recomp = math_ops.cast(step - self.warmup_steps, dtype)
      global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)
      completed_fraction = math_ops.div_no_nan(global_step_recomp, decay_steps)
      cosine_decayed = 0.5 * (1.0 + math_ops.cos(
          constant_op.constant(np.pi, dtype=dtype) * completed_fraction))

      decayed = (1.0 - alpha) * cosine_decayed + alpha
      learning_rate = math_ops.multiply(initial_learning_rate, decayed)
      return control_flow_ops.cond(
          step < self.warmup_steps,
          lambda: warmup_learning_rate,
          lambda: learning_rate,
          name=name)

  def get_config(self):
    config = {'warmup_steps': self.warmup_steps}
    base_config = super(WarmupCosineDecay, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class WarmupExponentialDecay(ExponentialDecay):
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
    with ops.name_scope_v2(self.name or "WarmupExponentialDecay") as name:
      initial_learning_rate = ops.convert_to_tensor_v2_with_dispatch(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      warmup_steps = math_ops.cast(self.warmup_steps, dtype)
      decay_steps = math_ops.cast(self.decay_steps, dtype)
      decay_rate = math_ops.cast(self.decay_rate, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      p = math_ops.div_no_nan(global_step_recomp, warmup_steps)
      p = math_ops.maximum(p, K.epsilon())
      warmup_learning_rate = math_ops.multiply(
          initial_learning_rate, p)

      global_step_recomp = math_ops.cast(step - self.warmup_steps, dtype)
      p = math_ops.div_no_nan(global_step_recomp, decay_steps)
      if self.staircase:
        p = math_ops.floor(p)
      learning_rate = math_ops.multiply(
          initial_learning_rate, math_ops.pow(decay_rate, p))
      learning_rate = math_ops.maximum(learning_rate, 1e-5)  # bounded below at 1e-5
      return control_flow_ops.cond(
          step < self.warmup_steps,
          lambda: warmup_learning_rate,
          lambda: learning_rate,
          name=name)

  def get_config(self):
    config = {'warmup_steps': self.warmup_steps}
    base_config = super(WarmupExponentialDecay, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
