"""Architecture NN models."""

import itertools
import numpy as np

import tensorflow as tf

import emtf_nnet

from emtf_nnet.keras.layers import (
    ActivityRegularization, FeatureNormalization, MutatedBatchNormalization,
    MutatedDense, ScaleActivation, TanhActivation)
from emtf_nnet.keras.losses import LogCosh
from emtf_nnet.keras.optimizers import Adamu, WarmupCosineDecay
from emtf_nnet.keras.quantization import quantize_model, quantize_scope
from emtf_nnet.keras.quantization import quantize_annotate  # this is a module
from emtf_nnet.keras.sparsity import prune_low_magnitude, prune_scope
from emtf_nnet.keras.sparsity import pruning_schedule as pruning_sched  # this is a module
from emtf_nnet.keras.sparsity import pruning_wrapper  # this is a module


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


def create_regularization_layer(noises, l1=1e-5, bias=14.0, batch_size=32, mask_value=999999):
  assert isinstance(noises, np.ndarray) and (noises.dtype == np.int32)

  # Cast noises to float
  def _ismasked(t):
    return (t == mask_value)
  noises_mask = _ismasked(noises)
  noises = noises.astype(np.float32)
  noises[noises_mask] = np.nan

  # Convert to tensor
  noises = tf.convert_to_tensor(noises, name='noises')

  # Create the layer
  regularization_layer = ActivityRegularization(l1=l1, bias=bias, name='regularization')
  regularization_layer.set_data(noises, batch_size=batch_size)
  return regularization_layer


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
  optimizer = Adamu(learning_rate=lr_schedule, clipnorm=gradient_clipnorm)
  return optimizer


def create_simple_model(preprocessing_layer=None,
                        regularization_layer=None,
                        optimizer=None,
                        nodes0=28,
                        nodes1=24,
                        nodes2=16,
                        nodes_in=40,
                        nodes_out=1,
                        name='simple_nnet_model'):
  if preprocessing_layer is None:
    preprocessing_layer = tf.keras.layers.Activation('linear', name='preprocessing')
  if regularization_layer is None:
    regularization_layer = tf.keras.layers.Activation('linear', name='regularization')
  if optimizer is None:
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

  # Sequential
  model = tf.keras.Sequential(name=name)
  # Input layer
  model.add(tf.keras.layers.InputLayer(input_shape=(nodes_in,), name='inputs'))
  # Preprocessing
  model.add(preprocessing_layer)
  # Hidden layer 0
  model.add(tf.keras.layers.Dense(nodes0, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense'))
  model.add(tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-4, name='batch_normalization'))
  model.add(tf.keras.layers.Activation('tanh', name='activation'))
  # Hidden layer 1
  model.add(tf.keras.layers.Dense(nodes1, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense_1'))
  model.add(tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-4, name='batch_normalization_1'))
  model.add(tf.keras.layers.Activation('tanh', name='activation_1'))
  # Hidden layer 2
  model.add(tf.keras.layers.Dense(nodes2, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense_2'))
  model.add(tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-4, name='batch_normalization_2'))
  model.add(tf.keras.layers.Activation('tanh', name='activation_2'))
  # Output layer
  model.add(tf.keras.layers.Rescaling(scale=1./64, name='rescaling'))
  model.add(tf.keras.layers.Dense(nodes_out, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense_final'))
  model.add(regularization_layer)
  # Loss function & optimizer
  model.compile(optimizer=optimizer, loss='mse', loss_weights=100)
  return model


def create_model(preprocessing_layer,
                 regularization_layer,
                 optimizer,
                 nodes0=28,
                 nodes1=24,
                 nodes2=16,
                 nodes_in=40,
                 nodes_out=1,
                 name='nnet_model'):
  # Sequential
  model = tf.keras.Sequential(name=name)
  regularization_layer.set_model(model)
  # Input layer
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
  model.add(ScaleActivation(scale=1./64, name='rescaling'))
  model.add(MutatedDense(nodes_out, kernel_initializer='glorot_uniform', use_bias=False, activation=None, name='dense_final'))
  model.add(regularization_layer)
  # Loss function & optimizer
  logcosh_loss = LogCosh()
  logcosh_loss_w = 100
  model.compile(optimizer=optimizer, loss=logcosh_loss, loss_weights=logcosh_loss_w)
  return model


def create_quant_model(base_model,
                       optimizer,
                       name='quant_nnet_model'):
  layers_to_skip = {'regularization'}

  def _add_quant_wrapper(layer):
    if layer.name in layers_to_skip:
      return layer
    # Already annotated layer. No need to wrap.
    if isinstance(layer, quantize_annotate.QuantizeAnnotate):
      return layer
    if isinstance(layer, tf.keras.Model):
      raise ValueError('Quantizing a tf.keras Model inside another tf.keras Model '
                       'is not supported.')
    return quantize_annotate.QuantizeAnnotate(layer)

  with quantize_scope():
    model = quantize_model(base_model, annotate_fn=_add_quant_wrapper)
    model._name = name

  # Set up regularization layer
  model.get_layer('regularization').set_data(
      base_model.get_layer('regularization').loss_fn._data)
  model.get_layer('regularization').set_model(model)

  # Loss function & optimizer
  compile_args = base_model._get_compile_args()
  compile_args['optimizer'] = optimizer
  model.compile(**compile_args)
  return model


def create_sparsity_m_by_n_list(m, n):
    return list(zip(range(m + 1), itertools.repeat(n)))


def create_pruning_schedule(num_train_samples,
                            epochs=100,
                            batch_size=32):
  steps_per_epoch = int(np.ceil(num_train_samples / float(batch_size)))
  total_steps = steps_per_epoch * epochs
  frequency = total_steps // 100
  return pruning_sched.PolynomialDecayMbyNSparsity(
      initial_coverage_ratio=0.0, begin_step=0, end_step=total_steps,
      power=1.0, frequency=frequency)


def create_pruned_model(base_model,
                        optimizer,
                        layers_to_prune,
                        layers_to_preserve,
                        pruning_schedule=pruning_sched.ConstantMbyNSparsity(),  # noqa: B008
                        sparsity_m_by_n=(2, 4),
                        name='pruned_nnet_model'):
  def _add_pruning_wrapper(layer):
    pruning_params = {
      'pruning_schedule': pruning_schedule,
      'sparsity_m_by_n': sparsity_m_by_n,
    }
    dummy_pruning_params = {
      'pruning_schedule': pruning_sched.ConstantMbyNSparsity(),
      'sparsity_m_by_n': (0, 4),
    }
    if isinstance(layer, tf.keras.Model):
      raise ValueError('Pruning a tf.keras Model inside another tf.keras Model '
                       'is not supported.')
    if layer.name in layers_to_prune:
      return pruning_wrapper.PruneLowMagnitude(layer, **pruning_params)
    elif layer.name in layers_to_preserve:
      # Preserve previous pruning
      return pruning_wrapper.PruneLowMagnitude(layer, **dummy_pruning_params)
    return layer

  with prune_scope():
    model = prune_low_magnitude(base_model, annotate_fn=_add_pruning_wrapper)
    model._name = name

  # Loss function & optimizer
  compile_args = base_model._get_compile_args()
  compile_args['optimizer'] = optimizer
  model.compile(**compile_args)
  return model


__all__ = [
  'get_x_y_data',
  'create_preprocessing_layer',
  'create_regularization_layer',
  'create_lr_schedule',
  'create_optimizer',
  'create_simple_model',
  'create_model',
  'create_quant_model',
  'create_sparsity_m_by_n_list',
  'create_pruning_schedule',
  'create_pruned_model',
]
