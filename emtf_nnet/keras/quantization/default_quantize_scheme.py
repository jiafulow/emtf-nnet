# The following source code was originally obtained from:
# https://github.com/tensorflow/model-optimization/blob/v0.7.0/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_scheme.py
# https://github.com/tensorflow/model-optimization/blob/v0.7.0/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_layout_transform.py
# https://github.com/tensorflow/model-optimization/blob/v0.7.0/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_registry.py
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

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation
from tensorflow_model_optimization.python.core.quantization.keras import quantize_layout_transform
from tensorflow_model_optimization.python.core.quantization.keras import quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras import quantize_scheme
from tensorflow_model_optimization.python.core.quantization.keras import quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras import quantizers
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import model_transformer

from emtf_nnet.keras.layers import (
    ActivityRegularization, FeatureNormalization, LinearActivation,
    MutatedBatchNormalization, MutatedDense, MutatedDenseFold, ScaleActivation,
    TanhActivation)

from .default_quantize_configs import (
    DefaultDenseQuantizeConfig, DefaultDenseFoldQuantizeConfig,
    DefaultInputQuantizeConfig, DefaultOutputQuantizeConfig, NoOpQuantizeConfig)

from .default_transforms import InputLayerQuantize, MutatedDenseFolding
from .quantizers import FixedRangeQuantizer


class DefaultQuantizeLayoutTransform(quantize_layout_transform.QuantizeLayoutTransform):
  """Default quantization layout transformations."""

  _TRANSFORMS = [
    #InputLayerQuantize(),
    MutatedDenseFolding(),
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
        model,
        self._TRANSFORMS,
        candidate_layers=set(layer_quantize_map.keys()),
        layer_metadata=layer_quantize_map).transform()


class DefaultQuantizeRegistry(quantize_registry.QuantizeRegistry):
  """Default quantization registry."""

  def __init__(self, disable_per_axis=False):
    self._layer_quantize_map = {}

    #self._layer_quantize_map[tf.keras.layers.Activation] = DefaultOutputQuantizeConfig()
    #self._layer_quantize_map[tf.keras.layers.BatchNormalization] = DefaultOutputQuantizeConfig()
    #self._layer_quantize_map[tf.keras.layers.Dense] = DefaultDenseQuantizeConfig()
    #self._layer_quantize_map[tf.keras.layers.Rescaling] = NoOpQuantizeConfig()

    self._layer_quantize_map[ActivityRegularization] = NoOpQuantizeConfig()
    self._layer_quantize_map[LinearActivation] = DefaultOutputQuantizeConfig()
    #self._layer_quantize_map[MutatedBatchNormalization] = DefaultOutputQuantizeConfig()
    self._layer_quantize_map[MutatedDense] = DefaultDenseQuantizeConfig()
    self._layer_quantize_map[MutatedDenseFold] = DefaultDenseFoldQuantizeConfig()
    self._layer_quantize_map[FeatureNormalization] = DefaultOutputQuantizeConfig()
    self._layer_quantize_map[ScaleActivation] = NoOpQuantizeConfig()
    self._layer_quantize_map[TanhActivation] = DefaultOutputQuantizeConfig()

    self._disable_per_axis = disable_per_axis  # unused

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
    return self._get_quantize_config(layer.__class__)


class DefaultQuantizeScheme(quantize_scheme.QuantizeScheme):
  """Quantization scheme which specifies how quantization should be applied."""

  _QUANTIZATION_OBJECTS = {
    'ActivityRegularization': ActivityRegularization,
    'FeatureNormalization': FeatureNormalization,
    'LinearActivation': LinearActivation,
    'MutatedBatchNormalization': MutatedBatchNormalization,
    'MutatedDense': MutatedDense,
    'MutatedDenseFold': MutatedDenseFold,
    'ScaleActivation': ScaleActivation,
    'TanhActivation': TanhActivation,
    'FixedRangeQuantizer': FixedRangeQuantizer,
    'DefaultDenseQuantizeConfig': DefaultDenseQuantizeConfig,
    'DefaultInputQuantizeConfig': DefaultInputQuantizeConfig,
    'DefaultOutputQuantizeConfig': DefaultOutputQuantizeConfig,
    'NoOpQuantizeConfig': NoOpQuantizeConfig,
    'DefaultDenseFoldQuantizeConfig': DefaultDenseFoldQuantizeConfig,
    # from tensorflow_model_optimization
    'QuantizeAwareActivation': quantize_aware_activation.QuantizeAwareActivation,
    'QuantizeWrapper': quantize_wrapper.QuantizeWrapper,
    'QuantizeWrapperV2': quantize_wrapper.QuantizeWrapperV2,
    'AllValuesQuantizer': quantizers.AllValuesQuantizer,
    'LastValueQuantizer': quantizers.LastValueQuantizer,
    'MovingAverageQuantizer': quantizers.MovingAverageQuantizer,
  }

  def __init__(self, disable_per_axis=False):
    self._disable_per_axis = disable_per_axis

  def get_layout_transformer(self):
    return DefaultQuantizeLayoutTransform()

  def get_quantize_registry(self):
    return DefaultQuantizeRegistry(
        disable_per_axis=self._disable_per_axis)
