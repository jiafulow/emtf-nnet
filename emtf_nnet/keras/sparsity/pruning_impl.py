# The following source code was originally obtained from:
# https://github.com/tensorflow/model-optimization/blob/v0.7.0/tensorflow_model_optimization/python/core/sparsity/keras/pruning_impl.py
# ==============================================================================

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Helper functions to add support for magnitude-based model pruning."""

import types

import tensorflow as tf

from tensorflow_model_optimization.python.core.keras import compat as tf_compat

from .pruning_utils import (
    expand_tensor, factorized_pool, weights_rearrange,
    m_by_n_sparsity_mask_prepare, generate_m_by_n_mask)

pruning_utils = types.ModuleType('pruning_utils')
pruning_utils.expand_tensor = expand_tensor
pruning_utils.factorized_pool = factorized_pool
pruning_utils.weights_rearrange = weights_rearrange
pruning_utils.m_by_n_sparsity_mask_prepare = m_by_n_sparsity_mask_prepare
pruning_utils.generate_m_by_n_mask = generate_m_by_n_mask


class Pruning(object):
  """Implementation of magnitude-based weight pruning."""

  def __init__(self,
               training_step_fn,
               pruning_vars,
               pruning_schedule,
               block_size,
               block_pooling_type,
               sparsity_m_by_n=None):
    """The logic for magnitude-based pruning weight tensors.

    Args:
      training_step_fn: A callable that returns the training step.
      pruning_vars: A list of (weight, mask, threshold) tuples
      pruning_schedule: A `PruningSchedule` object that controls pruning rate
        throughout training.
      block_size: The dimensions (height, weight) for the block sparse pattern
        in rank-2 weight tensors.
      block_pooling_type: (optional) The function to use to pool weights in the
        block. Must be 'AVG' or 'MAX'.
      sparsity_m_by_n: default None, otherwise a tuple of 2 integers, indicates
        pruning with m_by_n sparsity, e.g., (2, 4): 2 zeros out of 4 consecutive
        elements. It check whether we can do pruning with m_by_n sparsity.
    """
    self._pruning_vars = pruning_vars
    self._pruning_schedule = pruning_schedule
    self._block_size = list(block_size)
    self._block_pooling_type = block_pooling_type
    self._sparsity_m_by_n = sparsity_m_by_n

    # Training step
    self._step_fn = training_step_fn

    self._validate_block()

  def _validate_block(self):
    if self._block_size != [1, 1]:
      for weight, _, _ in self._pruning_vars:
        if weight.get_shape().ndims != 2:
          raise ValueError('Block Sparsity can only be used for layers which '
                           'have 2-dimensional weights.')

  def _update_mask(self, weights):
    """Updates the mask for a given weight tensor.

    This functions first estimates the threshold value such that
    a given fraction of weights have magnitude less than
    the threshold.

    Args:
      weights: The weight tensor that needs to be masked.

    Returns:
      new_threshold: The new value of the threshold based on weights, and
        sparsity at the current global_step
      new_mask: A numpy array of the same size and shape as weights containing
        0 or 1 to indicate which of the values in weights falls below
        the threshold

    Raises:
      ValueError: if sparsity is not defined
    """
    sparsity = self._pruning_schedule(self._step_fn())[1]
    with tf.name_scope('pruning_ops'):
      abs_weights = tf.math.abs(weights)
      k = tf.dtypes.cast(
          tf.math.maximum(
              tf.math.round(
                  tf.dtypes.cast(tf.size(abs_weights), tf.float32) *
                  (1 - sparsity)),
              1),
          tf.int32)
      # Sort the entire array
      values, _ = tf.math.top_k(
          tf.reshape(abs_weights, [-1]), k=tf.size(abs_weights))
      # Grab the (k-1)th value

      current_threshold = tf.gather(values, k - 1)
      new_mask = tf.dtypes.cast(
          tf.math.greater_equal(abs_weights, current_threshold), weights.dtype)
    return current_threshold, new_mask

  def _update_mask_sparsity_m_by_n(self, weights, m_by_n=(2, 4)):
    """Updates the m by n sparsity mask for a given weight tensor.

    This function creates a mask for the given weight tensor so
    that n elements with the lowest absolute values in the block
    of m elements are set to be zero. We don't return any threshold.

    If coverage ratio provided with the pruning schedule is less than 1.0,
    a partially sparsity mask will be calculated and add up to m_by_n
    sparsity mask, so that coverage ratio of m_by_n sparsity pattern
    on mask is (coverage_ratio * 100%).

    Args:
      weights: The weight tensor that needs to be masked.
      m_by_n: tuple of two integers, indicating m zeros out of n consecutive
        elements, default as 2 by 4 sparsity.

    Returns:
      new_mask: A numpy array of the same size and shape as weights containing
      0 or 1 to indicate which of the values in weights should be set to zero.
      It throws an error if the requested mask cannot be created.
    """
    coverage_ratio = self._pruning_schedule(self._step_fn())[1]
    with tf.name_scope('m_by_n_sparsity_pruning_ops'):
      prepared_weights = pruning_utils.weights_rearrange(weights)

      mask = pruning_utils.generate_m_by_n_mask(prepared_weights, m_by_n)
      new_mask = pruning_utils.m_by_n_sparsity_mask_prepare(mask, weights.shape)

      def update_mask_sparsity_m_by_n_with_coverage_ratio():
        partial_covered_mask = pruning_utils.generate_partial_sparsity_mask(
            prepared_weights, m_by_n[1], coverage_ratio)
        new_partial_covered_mask = pruning_utils.m_by_n_sparsity_mask_prepare(
            partial_covered_mask, weights.shape)

        m_by_n_mask = tf.clip_by_value(
            new_mask + new_partial_covered_mask,
            clip_value_min=0.0,
            clip_value_max=1.0
        )

        return m_by_n_mask

      m_by_n_mask = tf.cond(
          tf.math.less(coverage_ratio, 1.0),
          update_mask_sparsity_m_by_n_with_coverage_ratio,
          lambda: new_mask,
      )

    return m_by_n_mask

  def _maybe_update_block_mask(self, weights):
    """Performs block-granular masking of the weights.

    If sparsity_m_by_n is selected, then we return the relevant pruning mask,
    that nullify m out of n consecutive elements in the block.

    Block pruning occurs only if the block_height or block_width is > 1 and
    if the weight tensor, when squeezed, has ndims = 2. Otherwise, elementwise
    pruning occurs.
    Args:
      weights: The weight tensor that needs to be masked.

    Returns:
      new_threshold: The new value of the threshold based on weights, and
        sparsity at the current global_step. In case of sparsity m_by_n,
        the returned threshold is an arbitrary number.
      new_mask: A numpy array of the same size and shape as weights containing
        0 or 1 to indicate which of the values in weights falls below
        the threshold

    Raises:
      ValueError: if block pooling function is not AVG or MAX
    """
    if self._sparsity_m_by_n:
      mask = self._update_mask_sparsity_m_by_n(weights, self._sparsity_m_by_n)
      # We need to return some numbers for threshold.
      return (tf.constant(999.0, dtype=tf.float32), mask)

    if self._block_size == [1, 1]:
      return self._update_mask(weights)

    # TODO(pulkitb): Check if squeeze operations should now be removed since
    # we are only accepting 2-D weights.

    squeezed_weights = tf.squeeze(weights)
    abs_weights = tf.math.abs(squeezed_weights)
    pooled_weights = pruning_utils.factorized_pool(
        abs_weights,
        window_shape=self._block_size,
        pooling_type=self._block_pooling_type,
        strides=self._block_size,
        padding='SAME')

    if pooled_weights.get_shape().ndims != 2:
      pooled_weights = tf.squeeze(pooled_weights)

    new_threshold, new_mask = self._update_mask(pooled_weights)

    updated_mask = pruning_utils.expand_tensor(new_mask, self._block_size)
    sliced_mask = tf.slice(
        updated_mask, [0, 0],
        [squeezed_weights.get_shape()[0],
         squeezed_weights.get_shape()[1]])
    return new_threshold, tf.reshape(sliced_mask, tf.shape(weights))

  def weight_mask_op(self):
    """Returns an op to assign weights<=weights*mask."""

    assign_objs = []

    for weight, mask, _ in self._pruning_vars:
      masked_weight = tf.math.multiply(weight, mask)
      assign_objs.append(tf_compat.assign(weight, masked_weight))

    return tf.group(assign_objs)

  def conditional_mask_update(self):
    """Returns an op to updates masks as per the pruning schedule."""

    should_update = self._pruning_schedule(self._step_fn())[0]

    def no_update():
      return tf.no_op()

    def update():
      assign_objs = []

      for weight, mask, threshold in self._pruning_vars:
        new_threshold, new_mask = self._maybe_update_block_mask(weight)
        assign_objs.append(tf_compat.assign(threshold, new_threshold))
        assign_objs.append(tf_compat.assign(mask, new_mask))

      return tf.group(assign_objs)

    return tf.cond(should_update, update, no_update)
