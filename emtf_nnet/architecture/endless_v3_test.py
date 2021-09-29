"""Testing"""

import numpy as np

import tensorflow as tf

import emtf_nnet

from .endless_v3 import (configure,
                         set_config,
                         set_nnet_model,
                         set_pattern_bank,
                         create_model)


def test_me():
  config = configure(strict=False)
  num_emtf_zones = config['num_emtf_zones']
  num_emtf_patterns = config['num_emtf_patterns']
  num_emtf_features = config['num_emtf_features']
  num_img_rows = config['num_img_rows']
  num_img_channels = config['num_img_channels']
  num_box_cols = config['num_box_cols']

  # Set up a fake pattern bank
  _patterns_shape = (num_emtf_zones, num_emtf_patterns, num_img_rows, 3)
  _patt_filters_shape = (num_emtf_zones, num_img_channels, num_box_cols, num_img_rows,
                         num_emtf_patterns)
  _patt_activations_shape = (num_emtf_zones, 2 ** num_img_rows)

  patterns = np.zeros(_patterns_shape, dtype=np.int32)
  patt_filters = np.zeros(_patt_filters_shape, dtype=bool)
  patt_activations = np.zeros(_patt_activations_shape, dtype=np.int32)
  fake_pattern_bank = emtf_nnet.keras.utils.PatternBank(
      patterns=patterns, patt_filters=patt_filters, patt_activations=patt_activations)

  # Set up a fake NN model
  nodes_in, nodes_out = (num_emtf_features, 1)
  fake_nnet_model = tf.keras.Sequential()
  fake_nnet_model.add(tf.keras.layers.InputLayer(input_shape=(nodes_in,), name='inputs'))
  fake_nnet_model.add(tf.keras.layers.Dense(nodes_out, activation='relu'))

  # Configure
  set_pattern_bank(fake_pattern_bank)
  set_nnet_model(fake_nnet_model)
  config = configure()
  set_config(config)

  # Create model
  model = create_model()
  assert model
