# The following source code is obtained from:
# https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/losses.py#L980-L1033
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.losses import LossFunctionWrapper


def regularization_fn(y_true, y_pred):
  """Add regularization loss.

  Add penalty for 2-4 GeV muons with large pt.
  """
  factor = constant_op.constant(0.002, dtype=y_pred.dtype)  # max of the softplus term should be roughly 5
  one_over_four = constant_op.constant(0.25, dtype=y_pred.dtype)
  zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
  softplus_scale = constant_op.constant(10., dtype=y_pred.dtype)
  softplus_offset = constant_op.constant(np.log(np.expm1(1.0)), dtype=y_pred.dtype)
  softplus_arg = (array_ops.where_v2(y_true < zeros, y_pred, -y_pred) * softplus_scale) + softplus_offset
  condition = math_ops.cast(math_ops.abs(y_true) >= one_over_four, y_pred.dtype)  # less than 4 GeV
  #regularization = factor * math_ops.reduce_sum(nn.softplus(softplus_arg) * condition)
  regularization = factor * math_ops.reduce_mean(nn.softplus(softplus_arg) * condition)
  return regularization


def log_cosh(y_true, y_pred):
  """Logarithm of the hyperbolic cosine of the prediction error.

  `logcosh = log(cosh(x)) = log((exp(x) + exp(-x))/2)`
  """
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  double = constant_op.constant(2.0, dtype=y_pred.dtype)
  log_two = constant_op.constant(np.log(2.0), dtype=y_pred.dtype)
  zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
  error = math_ops.subtract(y_pred, y_true)
  positive_branch = nn.softplus(-double * error) + error - log_two
  negative_branch = nn.softplus(double * error) - error - log_two

  # Add regularization loss
  use_regularization = True
  regularization = constant_op.constant(0., dtype=y_pred.dtype)
  if use_regularization:
    regularization += regularization_fn(y_true, y_pred)

  return K.mean(array_ops.where_v2(error < zeros, negative_branch, positive_branch), axis=-1) + regularization


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
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to 'log_cosh'.
    """
    super(LogCosh, self).__init__(log_cosh, name=name, reduction=reduction)
