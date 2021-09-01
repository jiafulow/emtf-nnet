# The following source code was originally obtained from:
# https://github.com/keras-team/keras/blob/r2.6/keras/losses.py#L1107-L1171
# https://github.com/keras-team/keras/blob/r2.6/keras/losses.py#L1546-L1577
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
"""Built-in loss functions."""

import tensorflow.compat.v2 as tf

from keras import backend
from keras.utils import losses_utils
from keras.losses import LossFunctionWrapper


def huber(y_true, y_pred, delta=1.345):
  """Computes Huber loss value.

  For each value x in `error = y_true - y_pred`:

  ```
  loss = 0.5 * x^2                  if |x| <= d
  loss = d * |x| - 0.5 * d^2        if |x| > d
  ```
  where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

  Args:
    y_true: tensor of true targets.
    y_pred: tensor of predicted targets.
    delta: A float, the point where the Huber loss function changes from a
      quadratic to linear.

  Returns:
    Tensor with one scalar loss entry per sample.
  """
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)
  delta = tf.cast(delta, y_pred.dtype)
  half = tf.cast(0.5, y_pred.dtype)
  error = tf.math.subtract(y_pred, y_true)
  squared_loss = half * tf.math.square(error)
  absolute_loss = delta * tf.math.abs(error) - half * tf.math.square(delta)
  return backend.mean(
      tf.where(tf.math.abs(error) <= delta, squared_loss, absolute_loss),
      axis=-1)


class Huber(LossFunctionWrapper):
  """Computes the Huber loss between `y_true` and `y_pred`.

  For each value x in `error = y_true - y_pred`:

  ```
  loss = 0.5 * x^2                  if |x| <= d
  loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
  ```
  where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

  Standalone usage:

  >>> y_true = [[0, 1], [0, 0]]
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> h = tf.keras.losses.Huber()
  >>> h(y_true, y_pred).numpy()
  0.155

  >>> # Calling with 'sample_weight'.
  >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
  0.09

  >>> # Using 'sum' reduction type.
  >>> h = tf.keras.losses.Huber(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> h(y_true, y_pred).numpy()
  0.31

  >>> # Using 'none' reduction type.
  >>> h = tf.keras.losses.Huber(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> h(y_true, y_pred).numpy()
  array([0.18, 0.13], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.Huber())
  ```
  """

  def __init__(self,
               delta=1.345,
               reduction=losses_utils.ReductionV2.AUTO,
               name='huber_loss'):
    """Initializes `Huber` instance.

    Args:
      delta: A float, the point where the Huber loss function changes from a
        quadratic to linear.
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'huber_loss'.
    """
    super().__init__(huber, name=name, reduction=reduction, delta=delta)
