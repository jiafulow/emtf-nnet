# The following source code is obtained from:
# https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_scheme.py
# https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_layout_transform.py
# https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_registry.py
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
"""Quantization scheme which specifies how quantization should be applied."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras import quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras import quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras import quantize_scheme
from tensorflow_model_optimization.python.core.quantization.keras import quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer

from k_quantization_quantize_configs import (
    FixedRangeQuantizer, DefaultDenseQuantizeConfig, DefaultInputQuantizeConfig, DefaultOutputQuantizeConfig,
    NoOpQuantizeConfig, SuperDenseQuantizeConfig)
from k_quantization_quantize_transforms import (
    QuantizeLayer, InputLayerQuantize, DenzuFolding, TanhjoReplace)

from k_layers_batchnoru import BatchNoru
from k_layers_denzu import Denzu
from k_layers_denzufold import DenzuFold
from k_layers_normalisa import Normalisa
from k_layers_tanhjo import Tanhjo
from k_layers_tanhlu import Tanhlu


class DefaultQuantizeLayoutTransform(quantize_layout_transform.QuantizeLayoutTransform):
  """Default quantization layout transformations."""

  def __init__(self):
    self._transforms = [
        #InputLayerQuantize(),
        DenzuFolding(),
        TanhjoReplace(),
    ]

  def apply(self, model, layer_quantize_map):
    """Implement default 8-bit transforms.
    Currently this means the following.
      1. Pull activations into layers, and apply fuse activations. (TODO)
      2. Modify range in incoming layers for Concat. (TODO)
      3. Fuse Conv2D/DepthwiseConv2D + BN into single layer.
    Args:
      model: Keras model to be quantized.
      layer_quantize_map: Map with keys as layer names, and values as dicts
        containing custom `QuantizeConfig`s which may have been passed with
        layers.
    Returns:
      (Transformed Keras model to better match TensorFlow Lite backend, updated
      layer quantize map.)
    """
    return model_transformer.ModelTransformer(
        model, self._transforms,
        set(layer_quantize_map.keys()), layer_quantize_map).transform()


class DefaultQuantizeRegistry(quantize_registry.QuantizeRegistry):
  """Default quantization registry."""

  def __init__(self):
    self._layer_quantize_map = {}

    #self._layer_quantize_map[tf.keras.layers.Dense] = DefaultDenseQuantizeConfig()
    #self._layer_quantize_map[tf.keras.layers.Activation] = DefaultOutputQuantizeConfig()
    self._layer_quantize_map[tf.keras.layers.Lambda] = NoOpQuantizeConfig()

    #self._layer_quantize_map[QuantizeLayer] = DefaultInputQuantizeConfig()
    #self._layer_quantize_map[BatchNoru] = DefaultOutputQuantizeConfig()
    self._layer_quantize_map[Denzu] = DefaultDenseQuantizeConfig()
    self._layer_quantize_map[DenzuFold] = SuperDenseQuantizeConfig()
    self._layer_quantize_map[Normalisa] = DefaultOutputQuantizeConfig()
    #self._layer_quantize_map[Tanhjo] = DefaultOutputQuantizeConfig()
    self._layer_quantize_map[Tanhlu] = DefaultOutputQuantizeConfig()

  def _is_supported_layer(self, layer_class):
    return layer_class in self._layer_quantize_map

  def _get_quantize_config(self, layer_class):
    return self._layer_quantize_map[layer_class]

  def supports(self, layer):
    """Returns whether the registry supports this layer type.
    # TODO(pulkitb): Consider pushing this function up to the registry.
    Args:
      layer: The layer to check for support.
    Returns:
      True/False whether the layer type is supported.
    """
    if self._is_supported_layer(layer.__class__):
      return True

    return False

  def get_quantize_config(self, layer):
    """Returns the quantization config for the given layer.
    Args:
      layer: input layer to return quantize config for.
    Returns:
      Returns the QuantizeConfig for the given layer.
    """
    if not self.supports(layer):
      raise ValueError(
          '`get_quantize_config()` called on an unsupported layer {}. Check '
          'if layer is supported by calling `supports()`. Alternatively, you '
          'can use `QuantizeConfig` to specify a behavior for your layer.'
          .format(layer.__class__))

    if self._is_supported_layer(layer.__class__):
      return self._get_quantize_config(layer.__class__)

    # Should never come here.
    raise ValueError('Invalid Layer type {}'.format(layer.__class__))


class DefaultQuantizeScheme(quantize_scheme.QuantizeScheme):
  """Quantization scheme which specifies how quantization should be applied."""

  _QUANTIZATION_OBJECTS = {
    'BatchNoru': BatchNoru,
    'Denzu': Denzu,
    'DenzuFold': DenzuFold,
    'Normalisa': Normalisa,
    'Tanhjo': Tanhjo,
    'Tanhlu': Tanhlu,
    'FixedRangeQuantizer': FixedRangeQuantizer,
    'DefaultDenseQuantizeConfig': DefaultDenseQuantizeConfig,
    'DefaultInputQuantizeConfig': DefaultInputQuantizeConfig,
    'DefaultOutputQuantizeConfig': DefaultOutputQuantizeConfig,
    'NoOpQuantizeConfig': NoOpQuantizeConfig,
    'SuperDenseQuantizeConfig': SuperDenseQuantizeConfig,
    'QuantizeAwareActivation': quantize_aware_activation.QuantizeAwareActivation,
    'QuantizeWrapper': quantize_wrapper.QuantizeWrapper,
    'AllValuesQuantizer': quantizers.AllValuesQuantizer,
    'LastValueQuantizer': quantizers.LastValueQuantizer,
    'MovingAverageQuantizer': quantizers.MovingAverageQuantizer,
  }

  def get_layout_transformer(self):
    return DefaultQuantizeLayoutTransform()

  def get_quantize_registry(self):
    return DefaultQuantizeRegistry()


def _test():
  """Quick test"""
  input_shape = (40,)
  custom_objects = dict(BatchNoru=BatchNoru, Denzu=Denzu, Normalisa=Normalisa, Tanhjo=Tanhjo)
  tf.keras.utils.get_custom_objects().update(custom_objects)

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
  model.add(Normalisa(axis=-1, name='preprocessing'))
  model.add(Denzu(30, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense'))
  model.add(BatchNoru(momentum=0.99, epsilon=1e-4, name='batch_normalization'))
  model.add(Tanhjo(name='activation'))
  model.add(Denzu(20, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense_1'))
  model.add(BatchNoru(momentum=0.99, epsilon=1e-4, name='batch_normalization_1'))
  model.add(Tanhjo(name='activation_1'))
  model.add(Denzu(10, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense_2'))
  model.add(BatchNoru(momentum=0.99, epsilon=1e-4, name='batch_normalization_2'))
  model.add(Tanhjo(name='activation_2'))
  model.add(tf.keras.layers.Lambda(lambda x: x / 32, name='lambda_normalization'))
  model.add(Denzu(1, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense_final'))
  model.summary()

  layout_transformer = DefaultQuantizeLayoutTransform()
  transforms = layout_transformer._transforms
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms).transform()
  transformed_model.summary()

  quantize_registry = DefaultQuantizeRegistry()
  for layer in transformed_model.layers:
    assert quantize_registry.supports(layer)
    assert quantize_registry.get_quantize_config(layer) is not None
