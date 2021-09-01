# The following source code was originally obtained from:
# https://github.com/tensorflow/model-optimization/blob/c69b0f/tensorflow_model_optimization/python/core/quantization/keras/layers/conv_batchnorm.py
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
"""Dense with folded batch normalization layer."""

import tensorflow.compat.v2 as tf

from keras import backend
from keras.utils import control_flow_util

from .mutated_batch_normalization import MutatedBatchNormalization
from .mutated_dense import MutatedDense


class MutatedDenseFold(MutatedDense):
  """Experimental dense layer with folded batch normalization layer.

  It also applies quantization (if used).
  """

  def __init__(self,
               # Dense params
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               name=None,
               # BatchNormalization params
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               renorm=False,
               renorm_clipping=None,
               renorm_momentum=0.99,
               fused=None,
               trainable=True,
               virtual_batch_size=None,
               adjustment=None,
               **kwargs):
    if activation is not None and activation != 'linear':
      raise ValueError('Nonlinear activation is not allowed.')

    super().__init__(
        units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        name=name,
        **kwargs)

    self.batchnorm = MutatedBatchNormalization(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        fused=fused,
        trainable=trainable,
        virtual_batch_size=virtual_batch_size,
        adjustment=adjustment,
        name=self.name + '_batchnorm')

  def build(self, input_shape):
    # responsible for trainable self.kernel weights
    super().build(input_shape)

    # resposible for trainable gamma and beta weights
    self.batchnorm.build(self.compute_output_shape(input_shape))
    self.built = True

  def _make_quantizer_fn(self, quantizer, x, training, quantizer_vars):
    """Use currying to return True/False specialized fns to the cond."""

    def quantizer_fn():
      return quantizer(x, training, weights=quantizer_vars)
    return quantizer_fn

  def _apply_quantizer(self, quantizer, x, training, quantizer_vars):
    if training is None:
      training = backend.learning_phase()
    return control_flow_util.smart_cond(
        training,
        self._make_quantizer_fn(quantizer, x, True, quantizer_vars),
        self._make_quantizer_fn(quantizer, x, False, quantizer_vars))

  def call(self, inputs, training=None, mask=None):
    outputs = super().call(inputs, training=training, mask=mask)
    _ = self.batchnorm.call(outputs, training=training)

    # Fold the batchnorm weights into kernel, add bias
    folded_kernel_multiplier = self.batchnorm.gamma * tf.math.rsqrt(
        self.batchnorm.moving_variance + self.batchnorm.epsilon,
        name='folded_kernel_multiplier')
    folded_kernel = tf.math.multiply(
        folded_kernel_multiplier, self.kernel, name='folded_kernel')
    folded_bias = tf.math.subtract(
        self.batchnorm.beta,
        self.batchnorm.moving_mean * folded_kernel_multiplier,
        name='folded_bias')

    # Quantize the weights (if quantized)
    if getattr(self, '_quantize_weight_vars', None):
      (unquantized_weight, quantizer, quantizer_vars) = self._quantize_weight_vars[0]
      folded_kernel = self._apply_quantizer(quantizer, folded_kernel, training, quantizer_vars)
      folded_bias = self._apply_quantizer(quantizer, folded_bias, training, quantizer_vars)

    # Swap the weights
    original_kernel = self.kernel
    original_bias = self.bias
    self.kernel = folded_kernel
    self.bias = folded_bias

    # Actual call
    outputs = super().call(inputs, training=training, mask=mask)

    # Swap back the original weights
    self.folded_kernel = self.kernel
    self.folded_bias = self.bias
    self.kernel = original_kernel
    self.bias = original_bias

    # Quantize the output after activation (if quantized)
    if getattr(self, '_quantize_activation_vars', None):
      (activation, quantizer, quantizer_vars) = self._quantize_activation_vars[0]
      outputs = self._apply_quantizer(quantizer, outputs, training, quantizer_vars)
    return outputs

  def get_config(self):
    batchnorm_config = self.batchnorm.get_config()
    batchnorm_config.pop('name')
    config = super().get_config()
    config.update(batchnorm_config)
    return config
