"""Architecture common classes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types

import tensorflow as tf


class _BaseLayer(tf.keras.layers.Layer):
  """Layer used as the base."""
  def __init__(self, **kwargs):
    super(_BaseLayer, self).__init__(**kwargs)

  def get_config(self):
    config = {}
    if hasattr(self, 'zone'):
      config['zone'] = self.zone
    if hasattr(self, 'timezone'):
      config['timezone'] = self.timezone
    base_config = super(_BaseLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def _create_base_layer_module():
  """Create a base_layer module that contains the _BaseLayer class."""
  base_layer = types.ModuleType('base_layer')
  # flake8: noqa:B010
  setattr(base_layer, 'Layer', _BaseLayer)  # export as base_layer.Layer
  return base_layer


base_layer = _create_base_layer_module()


__all__ = [
  'base_layer',
]
