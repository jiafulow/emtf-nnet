# The following source code was originally obtained from:
# https://github.com/tensorflow/model-optimization/blob/v0.7.0/tensorflow_model_optimization/python/core/sparsity/keras/pruning_callbacks.py
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
"""Keras callbacks for pruning."""

import types

import numpy as np

import tensorflow as tf

from .pruning_wrapper import PruneLowMagnitude, collect_prunable_layers

pruning_wrapper = types.ModuleType('pruning_wrapper')
pruning_wrapper.PruneLowMagnitude = PruneLowMagnitude
pruning_wrapper.collect_prunable_layers = collect_prunable_layers


class UpdatePruningStep(tf.keras.callbacks.Callback):
  """Keras callback which updates pruning wrappers with the optimizer step.

  This callback must be used when training a model which needs to be pruned. Not
  doing so will throw an error.

  Example:

  ```python
  model.fit(x, y,
      callbacks=[UpdatePruningStep()])
  ```
  """

  def __init__(self):
    super(UpdatePruningStep, self).__init__()
    self.prunable_layers = []

  def on_train_begin(self, logs=None):
    # Collect all the prunable layers in the model.
    self.prunable_layers = pruning_wrapper.collect_prunable_layers(self.model)
    if not self.prunable_layers:
      return
    # If the model is newly created/initialized, set the 'pruning_step' to 0.
    # If the model is saved and then restored, do nothing.
    if tf.keras.backend.get_value(self.prunable_layers[0].pruning_step) == -1:
      tuples = []
      for layer in self.prunable_layers:
        tuples.append((layer.pruning_step, 0))
      tf.keras.backend.batch_set_value(tuples)

  def on_epoch_end(self, epoch, logs=None):
    # At the end of every epoch, remask the weights. This ensures that when
    # the model is saved after completion, the weights represent mask*weights.
    weight_mask_ops = []

    for layer in self.prunable_layers:
      if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
        if tf.executing_eagerly():
          layer.pruning_obj.weight_mask_op()
        else:
          weight_mask_ops.append(layer.pruning_obj.weight_mask_op())

    tf.keras.backend.batch_get_value(weight_mask_ops)

    sparsities = []
    for layer in self.prunable_layers:
      if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
        if layer.pruning_vars:
          weight, _, _ = layer.pruning_vars[0]
          weight = tf.keras.backend.get_value(weight)
          sparsities.append(weight.size - np.count_nonzero(weight))
    logs['sparsities'] = sparsities
