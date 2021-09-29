"""Architecture common classes."""

import types

import tensorflow as tf


class _BaseLayer(tf.keras.layers.Layer):
  """Layer used as the base."""
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def get_config(self):
    config = super().get_config()
    if hasattr(self, 'zone'):
      config.update({'zone': self.zone})
    if hasattr(self, 'timezone'):
      config.update({'timezone': self.timezone})
    return config


# Export as base_layer.Layer
base_layer = types.ModuleType('base_layer')

base_layer.Layer = _BaseLayer


__all__ = [
  'base_layer',
]
