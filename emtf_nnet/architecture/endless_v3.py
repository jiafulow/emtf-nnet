"""Architecture layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

import emtf_nnet

from .endless_configs import configure_v3 as configure
from .endless_configs import (get_config,
                              get_nnet_model,
                              get_pattern_bank,
                              set_config,
                              set_nnet_model,
                              set_pattern_bank)


def _pack_data(ragged):
  assert isinstance(ragged.values, np.ndarray) and isinstance(ragged.row_splits, np.ndarray)

  config = get_config()
  zone_hits_fields = config['zone_hits_fields']
  num_emtf_segments = config['num_emtf_segments']

  # Unstack
  values = ragged.values
  fields = zone_hits_fields
  x_emtf_chamber = values[..., fields.emtf_chamber]
  x_emtf_segment = values[..., fields.emtf_segment]
  x_emtf_phi = values[..., fields.emtf_phi]
  x_emtf_bend = values[..., fields.emtf_bend]
  x_emtf_theta1 = values[..., fields.emtf_theta]
  x_emtf_theta2 = values[..., fields.emtf_theta_alt]
  x_emtf_qual1 = values[..., fields.emtf_qual]
  x_emtf_qual2 = values[..., fields.emtf_qual_alt]
  x_emtf_time = values[..., fields.emtf_time]
  x_zones = values[..., fields.zones]
  x_tzones = values[..., fields.timezones]
  x_fr = values[..., fields.fr]
  x_dl = values[..., fields.detlayer]
  x_bx = values[..., fields.bx]

  # Set valid flag
  x_valid = (x_emtf_segment < num_emtf_segments)
  x_valid = x_valid.astype(values.dtype)

  # Stack
  cham_indices = (
    x_emtf_chamber,
    x_emtf_segment,
  )
  cham_indices = np.stack(cham_indices, axis=-1)

  cham_values = (
    x_emtf_phi,
    x_emtf_bend,
    x_emtf_theta1,
    x_emtf_theta2,
    x_emtf_qual1,
    x_emtf_qual2,
    x_emtf_time,
    x_zones,
    x_tzones,
    x_fr,
    x_dl,
    x_bx,
    x_valid,
  )
  cham_values = np.stack(cham_values, axis=-1)

  # Build ragged arrays
  cham_indices = ragged.with_values(cham_indices)
  cham_values = ragged.with_values(cham_values)
  return (cham_indices, cham_values)


def _to_dense(indices, values):
  assert isinstance(indices, np.ndarray) and isinstance(values, np.ndarray)
  assert indices.shape[0] == values.shape[0]

  config = get_config()
  num_emtf_chambers = config['num_emtf_chambers']
  num_emtf_segments = config['num_emtf_segments']
  num_emtf_variables = config['num_emtf_variables']

  # Sparse -> Dense, but also remove invalid segments
  dense_shape = (num_emtf_chambers, num_emtf_segments, num_emtf_variables)
  dense = np.zeros(dense_shape, dtype=values.dtype)
  x_valid = values[:, -1].astype(np.bool)  # get valid flag
  indices = indices[x_valid]
  values = values[x_valid]
  dense[indices[:, 0], indices[:, 1]] = values  # unsparsify
  return dense


def _get_sparse_transformed_samples(x_batch):
  # Get sparsified chamber data
  cham_indices, cham_values = _pack_data(x_batch)
  # Concatenate sparsified chamber indices and values
  outputs = [
      np.concatenate((cham_indices[i], cham_values[i]), axis=-1)
      for i in range(len(cham_indices))
  ]
  return outputs


def _get_transformed_samples(x_batch):
  # Get sparsified chamber data
  cham_indices, cham_values = _pack_data(x_batch)
  # Get unsparsified chamber data as images
  outputs = np.array([
      _to_dense(cham_indices[i], cham_values[i])
      for i in range(len(cham_indices))
  ])
  return outputs


def get_datagen_sparse(x, batch_size=1024):
  # Input data generator for human beings
  assert isinstance(x, tuple) and len(x) == 2
  x = emtf_nnet.ragged.RaggedTensorValue(values=x[0], row_splits=x[1])
  return emtf_nnet.keras.utils.TransformedDataGenerator(
      x, batch_size=batch_size, transform_fn=_get_sparse_transformed_samples)


def get_datagen(x, batch_size=1024):
  # Input data generator for machines
  assert isinstance(x, tuple) and len(x) == 2
  x = emtf_nnet.ragged.RaggedTensorValue(values=x[0], row_splits=x[1])
  return emtf_nnet.keras.utils.TransformedDataGenerator(
      x, batch_size=batch_size, transform_fn=_get_transformed_samples)


__all__ = [
  'configure',
  'get_config',
  'get_nnet_model',
  'get_pattern_bank',
  'set_config',
  'set_nnet_model',
  'set_pattern_bank',
  'get_datagen',
  'get_datagen_sparse',
]
