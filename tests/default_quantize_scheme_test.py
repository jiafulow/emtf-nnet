"""Testing DefaultQuantizeLayoutTransform and DefaultQuantizeRegistry."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from emtf_nnet.keras.layers import (
    FeatureNormalization, MutatedBatchNormalization, MutatedDense, TanhActivation)

from emtf_nnet.keras.quantization.default_quantize_scheme import (
    DefaultQuantizeLayoutTransform, DefaultQuantizeRegistry, model_transformer)


def test_me():
  input_shape = (40,)
  nodes0, nodes1, nodes2, nodes_out = (24, 24, 16, 1)

  custom_objects = dict(
      FeatureNormalization=FeatureNormalization,
      MutatedBatchNormalization=MutatedBatchNormalization,
      MutatedDense=MutatedDense,
      TanhActivation=TanhActivation)
  tf.keras.utils.get_custom_objects().update(custom_objects)

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name='inputs'))
  model.add(FeatureNormalization(axis=-1, name='preprocessing'))
  model.add(MutatedDense(nodes0, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense'))
  model.add(MutatedBatchNormalization(momentum=0.99, epsilon=1e-4, name='batch_normalization'))
  model.add(TanhActivation(name='activation'))
  model.add(MutatedDense(nodes1, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense_1'))
  model.add(MutatedBatchNormalization(momentum=0.99, epsilon=1e-4, name='batch_normalization_1'))
  model.add(TanhActivation(name='activation_1'))
  model.add(MutatedDense(nodes2, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense_2'))
  model.add(MutatedBatchNormalization(momentum=0.99, epsilon=1e-4, name='batch_normalization_2'))
  model.add(TanhActivation(name='activation_2'))
  model.add(tf.keras.layers.Lambda(lambda x: x / 64, name='lambda_normalization'))
  model.add(MutatedDense(nodes_out, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense_final'))
  model.summary()
  assert model

  layout_transform = DefaultQuantizeLayoutTransform()
  transforms = layout_transform._transforms
  transformed_model, _ = model_transformer.ModelTransformer(
      model, transforms).transform()
  transformed_model.summary()
  assert transformed_model

  quantize_registry = DefaultQuantizeRegistry()
  for layer in transformed_model.layers:
    assert quantize_registry.supports(layer)
    assert quantize_registry.get_quantize_config(layer) is not None
