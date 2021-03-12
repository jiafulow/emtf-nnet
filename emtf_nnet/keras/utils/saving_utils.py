"""Utilities for saving and loading model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pickle

import tensorflow as tf


def save_model(model, name=None, custom_objects=None):
  # Save as model.h5, model_weights.h5, and model.json
  if name is None:
    name = model.name
  model.save(name + '.h5')
  model.save_weights(name + '_weights.h5')
  with open(name + '.json', 'w') as f:
    f.write(model.to_json())
  if custom_objects is not None:
    with open(name + '_objects.pkl', 'wb') as f:
      pickle.dump(custom_objects, f, protocol=pickle.HIGHEST_PROTOCOL)
  return


def load_model(path, w_path, obj_path=None):
  # Example usage:
  #     loaded_model = load_model('model.json', 'model_weights.h5', 'model_objects.pkl')
  if not path.endswith('.json'):
    raise ValueError('Expected a .json file, got: {}'.format(path))
  if not w_path.endswith('.h5'):
    raise ValueError('Expected a .h5 file, got: {}'.format(w_path))
  if obj_path is not None and not obj_path.endswith('.pkl'):
    raise ValueError('Expected a .pkl file, got: {}'.format(obj_path))

  if obj_path is not None:
    with open(obj_path, 'rb') as f:
      custom_objects = pickle.load(f)
      tf.keras.utils.get_custom_objects().update(custom_objects)

  with open(path, 'r') as f:
    json_string = json.dumps(json.load(f))
    model = tf.keras.models.model_from_json(json_string)

  model.load_weights(w_path)
  return model
