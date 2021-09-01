# The following source code was originally obtained from:
# https://github.com/keras-team/keras/blob/r2.6/keras/losses.py#L990-L1044
# https://github.com/keras-team/keras/blob/r2.6/keras/losses.py#L1580-L1617
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

import numpy as np

from keras import backend
from keras.utils import losses_utils
from keras.losses import LossFunctionWrapper


def log_cosh(y_true, y_pred):
  """Logarithm of the hyperbolic cosine of the prediction error.

  `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
  to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
  like the mean squared error, but will not be so strongly affected by the
  occasional wildly incorrect prediction.

  Standalone usage:

  >>> y_true = np.random.random(size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.logcosh(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> x = y_pred - y_true
  >>> assert np.allclose(
  ...     loss.numpy(),
  ...     np.mean(x + np.log(np.exp(-2. * x) + 1.) - math_ops.log(2.), axis=-1),
  ...     atol=1e-5)

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Logcosh error values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)
  zero = tf.cast(0., y_pred.dtype)
  double = tf.cast(2., y_pred.dtype)
  log_two = tf.cast(np.log(2.), y_pred.dtype)
  error = tf.subtract(y_pred, y_true)
  positive_branch = tf.math.softplus(-double * error) + error - log_two
  negative_branch = tf.math.softplus(double * error) - error - log_two
  return backend.mean(
      tf.where(error < zero, negative_branch, positive_branch),
      axis=-1)


class LogCosh(LossFunctionWrapper):
  """Computes the logarithm of the hyperbolic cosine of the prediction error.

  `logcosh = log((exp(x) + exp(-x))/2)`,
  where x is the error `y_pred - y_true`.

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [0., 0.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> l = tf.keras.losses.LogCosh()
  >>> l(y_true, y_pred).numpy()
  0.108

  >>> # Calling with 'sample_weight'.
  >>> l(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
  0.087

  >>> # Using 'sum' reduction type.
  >>> l = tf.keras.losses.LogCosh(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> l(y_true, y_pred).numpy()
  0.217

  >>> # Using 'none' reduction type.
  >>> l = tf.keras.losses.LogCosh(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> l(y_true, y_pred).numpy()
  array([0.217, 0.], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.LogCosh())
  ```
  """

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='log_cosh'):
    """Initializes `LogCosh` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'log_cosh'.
    """
    super().__init__(log_cosh, name=name, reduction=reduction)
