"""Architecture NN models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

import emtf_nnet

from emtf_nnet.keras.layers import (
    FeatureNormalization, MutatedBatchNormalization, MutatedDense, TanhActivation)
from emtf_nnet.keras.losses import LogCosh
from emtf_nnet.keras.optimizers import AdamOptim, WarmupCosineDecay
from emtf_nnet.keras.quantization.quantize_model import quantize_model, quantize_scope


def get_x_y_data(features, truths, batch_size=32, mask_value=999999):
  assert isinstance(features, np.ndarray) and (features.dtype == np.int32)
  assert isinstance(truths, np.ndarray) and (truths.dtype == np.float32)
  assert features.shape[0] == truths.shape[0]

  # Split into train/test
  x_train, x_test, y_train, y_test = emtf_nnet.keras.utils.train_test_split(
      features, truths, batch_size=batch_size)

  # Cast x_train, x_test to float
  def _ismasked(t):
    return (t == mask_value)
  x_train_mask = _ismasked(x_train)
  x_test_mask = _ismasked(x_test)
  x_train = x_train.astype(np.float32)
  x_test = x_test.astype(np.float32)
  x_train[x_train_mask] = np.nan
  x_test[x_test_mask] = np.nan

  # Select parameter of interest
  y_train = y_train[:, :1]
  y_test = y_test[:, :1]
  return (x_train, x_test, y_train, y_test)


def create_preprocessing_layer(x_train, axis=-1):
  # Find mean and variance for each channel
  reduction_axes = np.arange(x_train.ndim)
  reduction_axes = tuple(d for d in reduction_axes if d != reduction_axes[axis])
  # Randomize sign so that mean is zero
  sign_randomization = ((np.random.random_sample(x_train.shape) < 0.5) * 2.) - 1.
  x_adapt = x_train * sign_randomization

  # First pass: find mean & var without NaN
  _, var = (np.nanmean(x_adapt, axis=reduction_axes), np.nanvar(x_adapt, axis=reduction_axes))
  # Second pass: normalize, apply nonlinearity, find mean & var again
  # flake8: noqa:E731
  div_no_nan = emtf_nnet.keras.utils.div_no_nan
  normalize = lambda x: x * div_no_nan(1., np.sqrt(var))  # ignore mean
  nonlinearity = lambda x: np.tanh(x * np.arctanh(3. / 6.) / 3.) * 6.  # linear up to +/-3, saturate at +/-6
  x_adapt = nonlinearity(normalize(x_adapt))
  _, var1 = (np.nanmean(x_adapt, axis=reduction_axes), np.nanvar(x_adapt, axis=reduction_axes))

  # Fake-quantize the weights
  quantize_fn = lambda x: np.clip(np.round(x * 1024.), 0., 1023.) / 1024.  # assume 10 fractional bits
  weights = [div_no_nan(1., np.sqrt(var * var1)), (var1 * 0.)]  # kernel=1/sqrt(var), bias=0
  weights = list(map(quantize_fn, weights))

  # Create the layer
  preprocessing_layer = FeatureNormalization(axis=-1, name='preprocessing')
  preprocessing_layer.build(x_train.shape)
  preprocessing_layer.set_weights(weights)  # non-trainable weights
  return preprocessing_layer


def create_lr_schedule(num_train_samples,
                       epochs=100,
                       warmup_epochs=30,
                       batch_size=32,
                       learning_rate=0.001,
                       final_learning_rate=0.00001):
  # Create learning rate schedule with warmup and cosine decay
  steps_per_epoch = int(np.ceil(num_train_samples / float(batch_size)))
  warmup_steps = steps_per_epoch * warmup_epochs
  total_steps = steps_per_epoch * epochs
  cosine_decay_alpha = final_learning_rate / learning_rate
  assert warmup_steps <= total_steps
  return WarmupCosineDecay(
      initial_learning_rate=learning_rate, warmup_steps=warmup_steps,
      decay_steps=(total_steps - warmup_steps), alpha=cosine_decay_alpha)


def create_optimizer(lr_schedule,
                     gradient_clipnorm=10000):
  # Create optimizer with lr_schedule and clipnorm
  optimizer = AdamOptim(learning_rate=lr_schedule, clipnorm=gradient_clipnorm)
  return optimizer


def create_model(nodes0=24,
                 nodes1=24,
                 nodes2=16,
                 nodes_in=40,
                 nodes_out=1,
                 preprocessing_layer=None,
                 optimizer=None,
                 name='nnet_model'):
  if preprocessing_layer is None:
    raise ValueError('preprocessing_layer cannot be None.')
  if optimizer is None:
    raise ValueError('optimizer cannot be None.')

  model = tf.keras.Sequential(name=name)
  model.add(tf.keras.layers.InputLayer(input_shape=(nodes_in,), name='inputs'))

  # Preprocessing
  model.add(preprocessing_layer)
  # Hidden layer 0
  model.add(MutatedDense(nodes0, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense'))
  model.add(MutatedBatchNormalization(momentum=0.99, epsilon=1e-4, name='batch_normalization'))
  model.add(TanhActivation(name='activation'))
  # Hidden layer 1
  model.add(MutatedDense(nodes1, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense_1'))
  model.add(MutatedBatchNormalization(momentum=0.99, epsilon=1e-4, name='batch_normalization_1'))
  model.add(TanhActivation(name='activation_1'))
  # Hidden layer 2
  model.add(MutatedDense(nodes2, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense_2'))
  model.add(MutatedBatchNormalization(momentum=0.99, epsilon=1e-4, name='batch_normalization_2'))
  model.add(TanhActivation(name='activation_2'))
  # Output layer
  model.add(tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./64, name='rescaling'))
  model.add(MutatedDense(nodes_out, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense_final'))

  # Loss function & optimizer
  logcosh_loss = LogCosh()
  logcosh_loss_w = 100
  model.compile(optimizer=optimizer, loss=logcosh_loss, loss_weights=logcosh_loss_w)
  return model


def create_quant_model(base_model,
                       optimizer=None,
                       name='quant_nnet_model'):
  if optimizer is None:
    raise ValueError('optimizer cannot be None.')

  with quantize_scope():
    model = quantize_model(base_model)
    model._name = name

  # Loss function & optimizer
  model.compile(optimizer=optimizer, loss=base_model.compiled_loss._user_losses,
                loss_weights=base_model.compiled_loss._user_loss_weights)
  return model


__all__ = [
  'get_x_y_data',
  'create_preprocessing_layer',
  'create_lr_schedule',
  'create_optimizer',
  'create_model',
  'create_quant_model',
]
