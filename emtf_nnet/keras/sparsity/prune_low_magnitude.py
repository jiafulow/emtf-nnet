# The following source code was originally obtained from:
# https://github.com/tensorflow/model-optimization/blob/v0.7.0/tensorflow_model_optimization/python/core/sparsity/keras/prune.py
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
# pylint: disable=protected-access,missing-docstring,unused-argument
"""Entry point for pruning models during training."""

import types

import tensorflow as tf

from .pruning_schedule import (
    ConstantSparsity, PolynomialDecay, ConstantMbyNSparsity, PolynomialDecayMbyNSparsity)
from .pruning_wrapper import PruneLowMagnitude

pruning_sched = types.ModuleType('pruning_sched')
pruning_sched.ConstantSparsity = ConstantSparsity
pruning_sched.PolynomialDecay = PolynomialDecay
pruning_sched.ConstantMbyNSparsity = ConstantMbyNSparsity
pruning_sched.PolynomialDecayMbyNSparsity = PolynomialDecayMbyNSparsity

pruning_wrapper = types.ModuleType('pruning_wrapper')
pruning_wrapper.PruneLowMagnitude = PruneLowMagnitude


def _add_pruning_wrapper(layer):
  pruning_params = {
    'pruning_schedule': pruning_sched.ConstantSparsity(0.5, 0)
  }
  if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
    return layer
  if isinstance(layer, tf.keras.Model):
    raise ValueError('Pruning a tf.keras Model inside another tf.keras Model '
                     'is not supported.')
  return pruning_wrapper.PruneLowMagnitude(layer, **pruning_params)


def _strip_pruning_wrapper(layer):
  if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
    # The _batch_input_shape attribute in the first layer makes a Sequential
    # model to be built. This makes sure that when we remove the wrapper from
    # the first layer the model's built state preserves.
    if (not hasattr(layer.layer, '_batch_input_shape') and
        hasattr(layer, '_batch_input_shape')):
      layer.layer._batch_input_shape = layer._batch_input_shape
    return layer.layer
  return layer


def prune_scope():
  """Provides a scope in which Pruned layers and models can be deserialized.

  For TF 2.X: this is not needed for SavedModel or TF checkpoints, which are
  the recommended serialization formats.

  For TF 1.X: if a tf.keras h5 model or layer has been pruned, it needs to be
  within this
  scope to be successfully deserialized. This is not needed for loading just
  keras weights.

  Returns:
      Object of type `CustomObjectScope` with pruning objects included.

  Example:

  ```python
  pruned_model = prune_low_magnitude(model, **self.params)
  keras.models.save_model(pruned_model, keras_file)

  with prune_scope():
    loaded_model = keras.models.load_model(keras_file)
  ```
  """
  return tf.keras.utils.custom_object_scope(
      {'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude})


def prune_low_magnitude(to_prune, annotate_fn=_add_pruning_wrapper):
  """Modify a tf.keras layer or model to be pruned during training.

  This function wraps a tf.keras model or layer with pruning functionality which
  sparsifies the layer's weights during training. For example, using this with
  50% sparsity will ensure that 50% of the layer's weights are zero.

  The function accepts either a single keras layer
  (subclass of `tf.keras.layers.Layer`), list of keras layers or a Sequential
  or Functional tf.keras model and handles them appropriately.

  If it encounters a layer it does not know how to handle, it will throw an
  error. While pruning an entire model, even a single unknown layer would lead
  to an error.

  Prune a model:

  ```python
  pruning_params = {
      'pruning_schedule': ConstantSparsity(0.5, 0),
      'block_size': (1, 1),
      'block_pooling_type': 'AVG'
  }

  model = prune_low_magnitude(
      keras.Sequential([
          layers.Dense(10, activation='relu', input_shape=(100,)),
          layers.Dense(2, activation='sigmoid')
      ]), **pruning_params)
  ```

  Prune a layer:

  ```python
  pruning_params = {
      'pruning_schedule': PolynomialDecay(initial_sparsity=0.2,
          final_sparsity=0.8, begin_step=1000, end_step=2000),
      'block_size': (2, 3),
      'block_pooling_type': 'MAX'
  }

  model = keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      prune_low_magnitude(layers.Dense(2, activation='tanh'), **pruning_params)
  ])
  ```

  Pretrained models: you must first load the weights and then apply the
  prune API:

  ```python
  model.load_weights(...)
  model = prune_low_magnitude(model)
  ```

  Optimizer: this function removes the optimizer. The user is expected to
  compile the model
  again. It's easiest to rely on the default (step starts at 0) and then
  use that to determine the desired begin_step for the pruning_schedules.

  Checkpointing: checkpointing should include the optimizer, not just the
  weights. Pruning supports
  checkpointing though
  upon inspection, the weights of checkpoints are not sparse
  (https://github.com/tensorflow/model-optimization/issues/206).

  Arguments:
      to_prune: A single keras layer, list of keras layers, or a
        `tf.keras.Model` instance.
      pruning_schedule: A `PruningSchedule` object that controls pruning rate
        throughout training.
      block_size: (optional) The dimensions (height, weight) for the block
        sparse pattern in rank-2 weight tensors.
      block_pooling_type: (optional) The function to use to pool weights in the
        block. Must be 'AVG' or 'MAX'.
      pruning_policy: (optional) The object that controls to which layers
        `PruneLowMagnitude` wrapper will be applied. This API is experimental
        and is subject to change.
      sparsity_m_by_n: default None, otherwise a tuple of 2 integers, indicates
        pruning with m_by_n sparsity, e.g., (2, 4): 2 zeros out of 4 consecutive
        elements. It check whether we can do pruning with m_by_n sparsity.
        If this type of sparsity is not applicable, then an error is thrown.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
        Ignored when to_prune is not a keras layer.

  Returns:
    Layer or model modified with pruning wrappers. Optimizer is removed.

  Raises:
    ValueError: if the keras layer is unsupported, or the keras model contains
    an unsupported layer.
  """

  is_sequential_or_functional = (
      isinstance(to_prune, tf.keras.Model) and
      (isinstance(to_prune, tf.keras.Sequential) or to_prune._is_graph_network))

  if not is_sequential_or_functional:
    raise ValueError(
        '`prune_low_magnitude` can only prune an object of the following '
        'types: tf.keras.Sequential, tf.keras.Model. You passed '
        'an object of type: {input}.'.format(input=to_prune.__class__.__name__))

  if (not to_prune._is_graph_network and
      not isinstance(to_prune, tf.keras.Sequential)):
    raise ValueError('Subclassed models are not supported currently.')

  return tf.keras.models.clone_model(
      to_prune, input_tensors=None, clone_function=annotate_fn)


def strip_pruning(model, annotate_fn=_strip_pruning_wrapper):
  """Strip pruning wrappers from the model.

  Once a model has been pruned to required sparsity, this method can be used
  to restore the original model with the sparse weights.

  Only sequential and functional models are supported for now.

  Arguments:
      model: A `tf.keras.Model` instance with pruned layers.

  Returns:
    A keras model with pruning wrappers removed.

  Raises:
    ValueError: if the model is not a `tf.keras.Model` instance.
    NotImplementedError: if the model is a subclass model.

  Usage:

  ```python
  orig_model = tf.keras.Model(inputs, outputs)
  pruned_model = prune_low_magnitude(orig_model)
  exported_model = strip_pruning(pruned_model)
  ```
  The exported_model and the orig_model share the same structure.
  """

  if not isinstance(model, tf.keras.Model):
    raise ValueError(
        'Expected model to be a `tf.keras.Model` instance but got: ', model)

  return tf.keras.models.clone_model(
      model, input_tensors=None, clone_function=annotate_fn)
