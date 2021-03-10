# The following source code is obtained from:
# https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/quantize.py#L80-L210
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
"""Quantization API functions for tf.keras models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras import quantize_annotate as quantize_annotate_module
from tensorflow_model_optimization.python.core.quantization.keras import quantize as quantize_module

from k_quantization_quantize_scheme import DefaultQuantizeScheme


def _add_quant_wrapper(layer):
  """Add annotation wrapper."""
  # Already annotated layer. No need to wrap.
  if isinstance(layer, quantize_annotate_module.QuantizeAnnotate):
    return layer
  if isinstance(layer, tf.keras.Model):
    raise ValueError(
        'Quantizing a tf.keras Model inside another tf.keras Model is not supported.')
  return quantize_annotate_module.QuantizeAnnotate(layer)


def quantize_scope(*args):
  """Scope which can be used to deserialize quantized Keras models and layers.

  Under `quantize_scope`, Keras methods such as `tf.keras.load_model` and
  `tf.keras.models.model_from_config` will be able to deserialize Keras models
  and layers which contain quantization classes such as `QuantizeConfig`
  and `Quantizer`.

  Example:

  ```python
  tf.keras.models.save_model(quantized_model, keras_file)

  with quantize_scope():
    loaded_model = tf.keras.models.load_model(keras_file)

  # If your quantized model uses custom objects such as a specific `Quantizer`,
  # you can pass them to quantize_scope to deserialize your model.
  with quantize_scope({'FixedRangeQuantizer', FixedRangeQuantizer}
    loaded_model = tf.keras.models.load_model(keras_file)
  ```

  For further understanding, see `tf.keras.utils.custom_object_scope`.

  Args:
    *args: Variable length list of dictionaries of `{name, class}` pairs to add
      to the scope created by this method.

  Returns:
    Object of type `CustomObjectScope` with quantization objects included.
  """
  quantization_objects = DefaultQuantizeScheme._QUANTIZATION_OBJECTS.copy()
  return tf.keras.utils.custom_object_scope(*(args + (quantization_objects,)))


def quantize_model(to_quantize, annotate_fn=_add_quant_wrapper):
  """Quantize a `tf.keras` model with the default quantization implementation.

  Quantization constructs a model which emulates quantization during training.
  This allows the model to learn parameters robust to quantization loss, and
  also model the accuracy of a quantized model.

  For more information, see
  https://www.tensorflow.org/model_optimization/guide/quantization/training

  Quantize a model:

  ```python
  # Quantize sequential model
  model = quantize_model(
      tf.keras.Sequential([
          layers.Dense(10, activation='relu', input_shape=(100,)),
          layers.Dense(2, activation='sigmoid')
      ]))

  # Quantize functional model
  in = tf.keras.Input((3,))
  out = tf.keras.Dense(2)(in)
  model = tf.keras.Model(in, out)

  quantized_model = quantize_model(model)
  ```

  Note that this function removes the optimizer from the original model.

  The returned model copies over weights from the original model. So while
  it preserves the original weights, training it will not modify the weights
  of the original model.

  Args:
    to_quantize: tf.keras model to be quantized. It can have pre-trained
      weights.

  Returns:
    Returns a new `tf.keras` model prepared for quantization.
  """
  if to_quantize is None:
    raise ValueError('`to_quantize` cannot be None')

  if not isinstance(to_quantize, tf.keras.Model):
    raise ValueError(
        '`to_quantize` can only be a `tf.keras.Model` instance. Use '
        'the `quantize_annotate_layer` API to handle individual layers.'
        'You passed an instance of type: {input}.'.format(
            input=to_quantize.__class__.__name__))

  if not isinstance(to_quantize, tf.keras.Sequential) \
      and not to_quantize._is_graph_network:  # pylint: disable=protected-access
    raise ValueError(
        '`to_quantize` can only either be a tf.keras Sequential or '
        'Functional model.')

  annotated_model = quantize_annotate_model(to_quantize, annotate_fn)
  return quantize_module.quantize_apply(annotated_model, DefaultQuantizeScheme())


def quantize_annotate_model(to_annotate, annotate_fn=_add_quant_wrapper):
  """Annotate a `tf.keras` model to be quantized.

  This function does not actually quantize the model. It merely specifies
  that the model needs to be quantized. `quantize_apply` can then be used
  to quantize the model.

  This function is intended to be used in conjunction with the
  `quantize_annotate_layer` API. Otherwise, it is simpler to use
  `quantize_model`.

  Annotate a model while overriding the default behavior for a layer:

  ```python
  quantize_config = MyDenseQuantizeConfig()

  model = quantize_annotate_model(
    tf.keras.Sequential([
      layers.Dense(10, activation='relu', input_shape=(100,)),
      quantize_annotate_layer(
          layers.Dense(2, activation='sigmoid'),
          quantize_config=quantize_config)
    ]))

  # The first Dense layer gets quantized with the default behavior,
  # but the second layer uses `MyDenseQuantizeConfig` for quantization.
  quantized_model = quantize_apply(model)
  ```

  Note that this function removes the optimizer from the original model.

  Args:
    to_annotate: `tf.keras` model which needs to be quantized.

  Returns:
    New tf.keras model with each layer in the model wrapped with
    `QuantizeAnnotate`. The new model preserves weights from the original
    model.
  """
  if to_annotate is None:
    raise ValueError('`to_annotate` cannot be None')

  if not isinstance(to_annotate, tf.keras.Model):
    raise ValueError(
        '`to_annotate` can only be a `tf.keras.Model` instance. Use '
        'the `quantize_annotate_layer` API to handle individual layers. '
        'You passed an instance of type: {input}.'.format(
            input=to_annotate.__class__.__name__))

  if not isinstance(to_annotate, tf.keras.Sequential) \
      and not to_annotate._is_graph_network:  # pylint: disable=protected-access
    raise ValueError(
        '`to_annotate` can only either be a tf.keras Sequential or '
        'Functional model.')

  return tf.keras.models.clone_model(
      to_annotate, input_tensors=None, clone_function=annotate_fn)
