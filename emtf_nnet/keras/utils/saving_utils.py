"""Utilities for saving and loading model."""

import json
import numpy as np

import keras
from keras.saving.saved_model import json_utils


class PatternBank(object):
  """Stores pattern bank objects as a Keras serializable object."""

  def __init__(self, patterns, patt_filters, patt_activations, name=None):
    if not name:
      name = 'pattern_bank'
    self.patterns = np.asarray(patterns, dtype=np.int32)
    self.patt_filters = np.asarray(patt_filters, dtype=bool)
    self.patt_activations = np.asarray(patt_activations, dtype=np.int32)
    self.name = name

  def get_config(self):
    config = {
      'patterns': self.patterns,
      'patt_filters': self.patt_filters,
      'patt_activations': self.patt_activations,
      'name': self.name,
    }
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


def save_serializable_object(obj, path):
  if not path.endswith('.json'):
    raise ValueError('Expected a .json file, got: {}'.format(path))
  with open(path, 'w') as f:
    json.dump(obj, f, default=json_utils.get_json_type)


def load_serializable_object(path):
  if not path.endswith('.json'):
    raise ValueError('Expected a .json file, got: {}'.format(path))
  with open(path, 'r') as f:
    return json.load(f)


def save_pattern_bank(pattern_bank, name=None):
  if not isinstance(pattern_bank, PatternBank):
    raise TypeError('pattern_bank must be an instance of PatternBank.')
  if name is None:
    name = pattern_bank.name
  config = pattern_bank.get_config()
  path = name + '.json'
  save_serializable_object(config, path)


def load_pattern_bank(path):
  config = load_serializable_object(path)
  return PatternBank.from_config(config)


def save_nnet_model(nnet_model, name=None):
  # Example usage:
  #     save_nnet_model(nnet_model, 'nnet_model')
  #     -> write nnet_model.h5, nnet_model_weights.h5, and nnet_model.json
  if not isinstance(nnet_model, (keras.Sequential, keras.Model)):
    raise TypeError('nnet_model must be a Keras Sequential or Functional model.')
  if name is None:
    name = nnet_model.name
  nnet_model.save(name + '.h5')
  nnet_model.save_weights(name + '_weights.h5')
  with open(name + '.json', 'w') as f:
    f.write(nnet_model.to_json())


def load_nnet_model(path, w_path, custom_objects=None):
  # Example usage:
  #     nnet_model = load_nnet_model('nnet_model.json', 'nnet_model_weights.h5')
  if not path.endswith('.json'):
    raise ValueError('Expected a .json file, got: {}'.format(path))
  if not w_path.endswith('.h5'):
    raise ValueError('Expected a .h5 file, got: {}'.format(w_path))
  with open(path, 'r') as f:
    json_string = json.dumps(json.load(f))
    nnet_model = keras.models.model_from_json(json_string, custom_objects=custom_objects)
    nnet_model.load_weights(w_path)
    return nnet_model
