"""Testing MutatedBatchNormalization."""

import numpy as np

import tensorflow as tf

from emtf_nnet.keras.layers import MutatedBatchNormalization


def test_me():
  input_shape = (4, 40)
  x = tf.random.normal(input_shape)
  y = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=0.1)(x)
  expected_output = y.numpy()
  y = MutatedBatchNormalization(momentum=0.9, epsilon=0.1)(x)
  actual_output = y.numpy()

  assert expected_output.shape == actual_output.shape
  assert np.allclose(actual_output, expected_output, rtol=1e-07, atol=0)
