"""Testing MutatedDepthwiseConv2D."""

import numpy as np

import tensorflow as tf

from emtf_nnet.keras.layers import MutatedDepthwiseConv2D


def test_me():
  input_shape = (4, 28, 28, 3)
  x = tf.random.normal(input_shape)
  y = tf.keras.layers.DepthwiseConv2D(3, use_bias=False, activation='relu')(x)
  expected_output = y.numpy()
  y = MutatedDepthwiseConv2D(3, use_bias=False, activation='relu')(x)
  actual_output = y.numpy()

  assert expected_output.shape == actual_output.shape
  #assert np.allclose(actual_output, expected_output, rtol=1e-07, atol=0)
