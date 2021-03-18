"""Testing InverseIntegerLookup."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from emtf_nnet.keras.layers import InverseIntegerLookup


def test_me():
  # Note that the integer 4 in data, which is out of the vocabulary space, returns an OOV token.
  vocab = [12, 36, 1138, 42]
  data = tf.constant([[0, 2, 3], [3, 4, 1]])
  layer = InverseIntegerLookup(vocabulary=vocab)
  out = layer(data)
  # flake8: noqa:E201
  gold = [[  12, 1138,   42],
          [  42,   -1,   36]]
  assert out.numpy().tolist() == gold
