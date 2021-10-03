# The following source code was originally obtained from:
# https://github.com/tensorflow/model-optimization/blob/v0.7.0/tensorflow_model_optimization/python/core/sparsity/keras/prune_registry.py
# ==============================================================================

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Registry responsible for built-in keras classes."""

import tensorflow as tf

from emtf_nnet.keras.layers import (
    FeatureNormalization, LinearActivation, MutatedBatchNormalization, MutatedDense,
    MutatedDenseFold, ScaleActivation, TanhActivation)


class PruneRegistry(object):
  """Registry responsible for built-in keras layers."""

  # The keys represent built-in keras layers and the values represent the
  # the variables within the layers which hold the kernel weights. This
  # allows the wrapper to access and modify the weights.
  _LAYERS_WEIGHTS_MAP = {
      #tf.keras.layers.Activation: [],
      #tf.keras.layers.BatchNormalization: [],
      #tf.keras.layers.Dense: ['kernel'],
      #tf.keras.layers.Rescaling: [],
      FeatureNormalization: [],
      LinearActivation: [],
      MutatedBatchNormalization: [],
      MutatedDense: ['kernel'],
      ScaleActivation: [],
      TanhActivation: [],
  }

  @classmethod
  def supports(cls, layer):
    """Returns whether the registry supports this layer type.

    Args:
      layer: The layer to check for support.

    Returns:
      True/False whether the layer type is supported.

    """
    if layer.__class__ in cls._LAYERS_WEIGHTS_MAP:
      return True
    return False

  @classmethod
  def _weight_names(cls, layer):
    return cls._LAYERS_WEIGHTS_MAP[layer.__class__]

  @classmethod
  def make_prunable(cls, layer):
    """Modifies a built-in layer object to support pruning.

    Args:
      layer: layer to modify for support.

    Returns:
      The modified layer object.

    """
    if not cls.supports(layer):
      raise ValueError('Layer ' + str(layer.__class__) + ' is not supported.')

    def get_prunable_weights():
      return [getattr(layer, weight) for weight in cls._weight_names(layer)]

    layer.get_prunable_weights = get_prunable_weights
    return layer
