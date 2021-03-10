# The following source code is obtained from:
# https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/quantize_layer.py
# https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_transforms.py
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
"""Default Transforms."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms

from k_layers_denzufold import DenzuFold
from k_layers_tanhlu import Tanhlu

deserialize_keras_object = tf.keras.utils.deserialize_keras_object
serialize_keras_object = tf.keras.utils.serialize_keras_object


class QuantizeLayer(tf.keras.layers.Layer):
  """Emulates quantization of tensors passed through the layer.

  Quantization occurs when it is wrapped by QuantizeWrapper.
  """

  def __init__(self, **kwargs):
    super(QuantizeLayer, self).__init__(**kwargs)
    self.supports_masking = True

  def call(self, inputs):
    return inputs


class InputLayerQuantize(transforms.Transform):
  """Quantizes InputLayer, by adding QuantizeLayer after it.

  InputLayer => InputLayer -> QuantizeLayer
  """

  def pattern(self):
    return transforms.LayerPattern('InputLayer')

  def replacement(self, match_layer):
    layer = QuantizeLayer()
    layer_config = serialize_keras_object(layer)
    layer_config['name'] = layer.name

    layer_metadata = {'quantize_config': None}
    return transforms.LayerNode(
        layer_config, weights=None, input_layers=[match_layer], metadata=layer_metadata)

  def custom_objects(self):
    return {'QuantizeLayer': QuantizeLayer}


class DenzuFolding(transforms.Transform):
  """Folds BatchNoru into Denzu."""

  def pattern(self):
    return transforms.LayerPattern(
        'BatchNoru', {}, [transforms.LayerPattern('Denzu', {}, [])])

  def replacement(self, match_layer):
    batchnorm_layer = match_layer.layer
    dense_layer = match_layer.input_layers[0].layer
    batchnorm_weights = match_layer.weights
    dense_weights = match_layer.input_layers[0].weights

    dense_config = dense_layer['config']
    batchnorm_config = batchnorm_layer['config']
    batchnorm_config.pop('name')
    #batchnorm_config['trainable'] = False  # set 'layer.trainable = False' to freeze the BN layer
    layer_config = dict(list(dense_config.items()) + list(batchnorm_config.items()))

    layer = DenzuFold(**layer_config)
    layer_config = serialize_keras_object(layer)
    layer_config['name'] = layer.name

    layer_weights = collections.OrderedDict(
        list(dense_weights.items()) + list(batchnorm_weights.items()))
    layer_metadata = {'quantize_config': None}
    return transforms.LayerNode(
        layer_config, weights=layer_weights, input_layers=None, metadata=layer_metadata)

  def custom_objects(self):
    return {'DenzuFold': DenzuFold}


class TanhjoReplace(transforms.Transform):
  """Replaces Tanhjo by Tanhlu."""

  def pattern(self):
    return transforms.LayerPattern('Tanhjo')

  def replacement(self, match_layer):
    layer_config = match_layer.layer['config']
    layer_config.pop('alpha')
    layer_config.pop('beta')

    layer = Tanhlu(**layer_config)
    layer_config = serialize_keras_object(layer)
    layer_config['name'] = layer.name

    layer_metadata = {'quantize_config': None}
    return transforms.LayerNode(
        layer_config, weights=None, input_layers=None, metadata=layer_metadata)

  def custom_objects(self):
    return {'Tanhlu': Tanhlu}
