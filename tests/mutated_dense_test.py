"""Testing MutatedDense."""

import numpy as np

import tensorflow as tf

from emtf_nnet.keras.layers import FeatureNormalization, MutatedDense


def _replace_nan(x, replacement=0):
  np.copyto(x, replacement, where=np.isnan(x))
  return x


def test_me():
  #input_shape = (2,)
  units = 4
  weights = [
    np.array([[0.5, 1.0, 1.5, 2.0], [-0.5, -1.0, -1.5, -2.0]], dtype=np.float32),
    np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32),
  ]
  input_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
  input_data_w_nan = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan]], dtype=np.float32)

  # Expectations
  layer = tf.keras.layers.Dense(
      units,
      kernel_initializer=tf.keras.initializers.Constant(weights[0]),
      bias_initializer=tf.keras.initializers.Constant(weights[1]))

  x = tf.convert_to_tensor(input_data)
  model = tf.keras.Sequential()
  model.add(layer)
  _ = model(x)  # get the model built

  # Find expected output and gradients
  with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
    tape.watch(x)
    tape.watch(layer.kernel)
    tape.watch(layer.bias)
    y = model(x)

  expected_output = y
  expected_gradients = [tape.gradient(y, x), tape.gradient(y, layer.kernel), tape.gradient(y, layer.bias)]
  #print(expected_output, expected_gradients)

  # Find expected output and gradients (inputs contain NaN)
  x = tf.convert_to_tensor(_replace_nan(input_data_w_nan.copy()))

  with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
    tape.watch(x)
    tape.watch(layer.kernel)
    tape.watch(layer.bias)
    y = model(x)

  expected_output_w_nan = y
  expected_gradients_w_nan = [tape.gradient(y, x), tape.gradient(y, layer.kernel), tape.gradient(y, layer.bias)]
  expected_gradients_w_nan[0] *= (3. / 2)  # only 2 out of 3 elements are valid
  expected_gradients_w_nan[0] *= np.array([[1.0], [1.0], [0.0]], dtype=np.float32)  # valid, valid, invalid
  expected_gradients_w_nan[1] *= (3. / 2)
  #print(expected_output_w_nan, expected_gradients_w_nan)

  # Actual results
  preprocessing_layer = FeatureNormalization()

  layer = MutatedDense(
      units,
      kernel_initializer=tf.keras.initializers.Constant(weights[0]),
      bias_initializer=tf.keras.initializers.Constant(weights[1]))

  x = tf.convert_to_tensor(input_data)
  model = tf.keras.Sequential()
  model.add(preprocessing_layer)
  model.add(layer)
  _ = model(x)  # get the model built

  # Find actual output and gradients
  with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
    tape.watch(x)
    tape.watch(layer.kernel)
    tape.watch(layer.bias)
    y = model(x)

  actual_output = y
  actual_gradients = [tape.gradient(y, x), tape.gradient(y, layer.kernel), tape.gradient(y, layer.bias)]
  #print(actual_output, actual_gradients)

  assert np.allclose(actual_output.numpy(), expected_output.numpy(), rtol=1e-07, atol=0)
  assert np.allclose(actual_gradients[0].numpy(), expected_gradients[0].numpy(), rtol=1e-07, atol=0)
  assert np.allclose(actual_gradients[1].numpy(), expected_gradients[1].numpy(), rtol=1e-07, atol=0)
  assert np.allclose(actual_gradients[2].numpy(), expected_gradients[2].numpy(), rtol=1e-07, atol=0)

  # Find actual output and gradients (inputs contain NaN)
  x = tf.convert_to_tensor(input_data_w_nan)

  with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
    tape.watch(x)
    tape.watch(layer.kernel)
    tape.watch(layer.bias)
    y = model(x)

  actual_output_w_nan = y
  actual_gradients_w_nan = [tape.gradient(y, x), tape.gradient(y, layer.kernel), tape.gradient(y, layer.bias)]
  #print(actual_output_w_nan, actual_gradients_w_nan)

  assert np.allclose(actual_output_w_nan.numpy(), expected_output_w_nan.numpy(), rtol=1e-07, atol=0)
  assert np.allclose(actual_gradients_w_nan[0].numpy(), expected_gradients_w_nan[0].numpy(), rtol=1e-07, atol=0)
  assert np.allclose(actual_gradients_w_nan[1].numpy(), expected_gradients_w_nan[1].numpy(), rtol=1e-07, atol=0)
  assert np.allclose(actual_gradients_w_nan[2].numpy(), expected_gradients_w_nan[2].numpy(), rtol=1e-07, atol=0)
