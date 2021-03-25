"""Architecture layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

import emtf_nnet

from .endless_common import base_layer
from .endless_configs import configure_v3 as configure
from .endless_configs import (get_config,
                              get_nnet_model,
                              get_pattern_bank,
                              set_config,
                              set_nnet_model,
                              set_pattern_bank)


class Zoning(base_layer.Layer):
  def __init__(self, zone, timezone, **kwargs):
    if 'dtype' in kwargs:
      kwargs.pop('dtype')
    kwargs['dtype'] = 'int32'

    super(Zoning, self).__init__(**kwargs)
    self.zone = zone
    self.timezone = timezone

    # Import config
    config = get_config()
    self.num_emtf_zones = config['num_emtf_zones']
    self.num_emtf_timezones = config['num_emtf_timezones']
    self.coarse_emtf_strip = config['coarse_emtf_strip']
    self.min_emtf_strip = config['min_emtf_strip']
    self.image_shape = config['image_shape']
    self.packed_hits_fields = config['packed_hits_fields']
    self.img_row_to_chamber_luts = config['img_row_to_chamber_luts']
    self.img_row_number_luts = config['img_row_number_luts']

    # Derived from config
    self.zone_chamber_indices = np.concatenate(self.img_row_to_chamber_luts[self.zone], axis=0)
    self.img_row_numbers = np.concatenate(self.img_row_number_luts[self.zone], axis=0)

  def _find_emtf_img_col(self, x):
    # Returns: (emtf_phi - min_emtf_strip) // coarse_emtf_strip
    coarse_emtf_strip = tf.constant(self.coarse_emtf_strip, dtype=self.dtype)
    min_emtf_strip = tf.constant(self.min_emtf_strip, dtype=self.dtype)
    return tf.math.floordiv(x - min_emtf_strip, coarse_emtf_strip)

  @tf.function
  def single_example_call(self, inputs):
    x = inputs

    # x shape is (C, S, V). C is num of chambers, S is num of segments, V is num of variables.
    if not x.shape.rank == 3:
      raise ValueError('inputs must be rank 3.')

    # Unstack
    fields = self.packed_hits_fields
    x_emtf_phi_init = x[..., fields.emtf_phi]
    x_zones_init = x[..., fields.zones]
    x_tzones_init = x[..., fields.tzones]
    x_valid_init = x[..., fields.valid]

    # Constants
    zero_value = tf.constant(0, dtype=self.dtype)
    one_value = tf.constant(1, dtype=self.dtype)
    min_emtf_strip = tf.constant(self.min_emtf_strip, dtype=self.dtype)

    # Find zone chambers
    # Result shape is (C', S). C' is num of chambers in this zone, S is num of segments.
    def _gather_from(t):
      return tf.gather(t, zone_chamber_indices)
    zone_chamber_indices = tf.convert_to_tensor(self.zone_chamber_indices)
    zone_chamber_indices = tf.cast(zone_chamber_indices, dtype=self.dtype)
    x_emtf_phi = _gather_from(x_emtf_phi_init)
    x_zones = _gather_from(x_zones_init)
    x_tzones = _gather_from(x_tzones_init)
    x_valid = _gather_from(x_valid_init)
    assert (x_emtf_phi.shape.rank == 2) and (x_emtf_phi.shape[0] == zone_chamber_indices.shape[0])

    # Find cols, rows
    img_row_numbers = tf.convert_to_tensor(self.img_row_numbers)
    img_row_numbers = tf.cast(img_row_numbers, dtype=self.dtype)
    # Translate from emtf_phi to img_col
    selected_cols = self._find_emtf_img_col(x_emtf_phi)
    selected_rows = tf.broadcast_to(tf.expand_dims(img_row_numbers, axis=-1), selected_cols.shape)
    assert selected_cols.shape == selected_rows.shape

    # Valid for this zone/timezone
    the_zone_bitpos = tf.constant((self.num_emtf_zones - 1) - self.zone, dtype=self.dtype)
    the_tzone_bitpos = tf.constant((self.num_emtf_timezones - 1) - self.timezone, dtype=self.dtype)
    the_zone = tf.bitwise.left_shift(one_value, the_zone_bitpos)
    the_tzone = tf.bitwise.left_shift(one_value, the_tzone_bitpos)
    boolean_mask_init = [
      tf.math.not_equal(x_valid, zero_value),
      tf.math.greater_equal(x_emtf_phi, min_emtf_strip),
      tf.math.not_equal(tf.bitwise.bitwise_and(x_zones, the_zone), zero_value),
      tf.math.not_equal(tf.bitwise.bitwise_and(x_tzones, the_tzone), zero_value),
    ]
    boolean_mask = tf.math.reduce_all(tf.stack(boolean_mask_init), axis=0)
    assert boolean_mask.shape == x_valid.shape

    # Suppress invalid cols, rows before creating the output image.
    selected_cols = tf.where(boolean_mask, selected_cols, tf.zeros_like(selected_cols))
    selected_rows = tf.where(boolean_mask, selected_rows, tf.zeros_like(selected_rows))

    # Scatter update the output image. Image shape is (8, 288, 1).
    boolean_mask_flat = tf.reshape(boolean_mask, [-1])  # flatten
    selected_cols_flat = tf.reshape(selected_cols, [-1])
    selected_rows_flat = tf.reshape(selected_rows, [-1])
    # scatter_init is 2-D
    scatter_shape = self.image_shape[:-1]  # without channel axis
    scatter_init = tf.zeros(scatter_shape, dtype=self.dtype)
    upd_indices = tf.stack([selected_rows_flat, selected_cols_flat], axis=-1)
    upd_values = tf.cast(boolean_mask_flat, dtype=self.dtype)
    assert scatter_init.shape.rank == upd_indices.shape[-1]
    assert upd_values.shape == upd_indices.shape[:-1]

    scatter = tf.tensor_scatter_nd_max(scatter_init, indices=upd_indices, updates=upd_values)
    scatter = tf.expand_dims(scatter, axis=-1)  # add channel axis
    assert scatter.shape == self.image_shape
    return scatter

  @tf.function
  def call(self, inputs):
    # Run on inputs individually
    outputs = tf.map_fn(self.single_example_call, inputs, fn_output_signature=self.dtype)
    #outputs = tf.vectorized_map(self.single_example_call, inputs)
    return outputs


class Pooling(base_layer.Layer):
  def __init__(self, zone, **kwargs):
    if 'dtype' in kwargs:
      kwargs.pop('dtype')
    kwargs['dtype'] = 'int32'

    super(Pooling, self).__init__(**kwargs)
    self.zone = zone

    # Import config
    config = get_config()
    self.num_emtf_patterns = config['num_emtf_patterns']
    self.num_img_rows = config['num_img_rows']
    self.patt_filters = config['patt_filters']
    self.patt_brightness = config['patt_brightness']

    # Derived from config
    self._build_conv2d()
    self._build_lookup()

  def _build_conv2d(self):
    """Builds a DepthwiseConv2D layer."""
    filt = self.patt_filters[self.zone].astype(np.float32)
    self.conv2d = emtf_nnet.keras.layers.MutatedDepthwiseConv2D(
        kernel_size=(filt.shape[0], filt.shape[1]),
        depth_multiplier=self.num_emtf_patterns,
        strides=(1, 1),
        padding='same',
        activation=None,
        use_bias=False,
        depthwise_initializer=tf.keras.initializers.Constant(filt),
        trainable=False)

  def _build_lookup(self):
    """Builds a IntegerLookup layer."""
    vocab = self.patt_brightness[self.zone].astype(np.int32)
    self.lookup = emtf_nnet.keras.layers.InverseIntegerLookup(
        vocabulary=vocab,
        trainable=False)

  @tf.function
  def call(self, inputs):
    x = inputs  # NHWC, which is (None, 8, 288, 1)

    if not x.shape.rank == 4:
      raise ValueError('inputs must be rank 4.')

    # Constants
    zero_value = tf.constant(0, dtype=self.dtype)
    one_value = tf.constant(1, dtype=self.dtype)

    # Using DepthwiseConv2D
    # Some shape and axis manipulations are necessary to get the correct results.
    def _reshape_4d_to_5d(t):
      assert t.shape.rank == 4
      shape = tf.shape(t)
      new_dim = tf.constant(self.num_emtf_patterns, dtype=shape.dtype)  # D is depth_multiplier
      new_dim = tf.reshape(new_dim, [-1])  # 0-D to 1-D
      new_shape = tf.concat([shape[:-1], tf.math.floordiv(shape[-1:], new_dim), new_dim],
                            axis=-1)
      return tf.reshape(t, new_shape)

    x = tf.cast(x, dtype=tf.keras.backend.floatx())  # conv2d input must be float
    x = tf.transpose(x, perm=(0, 3, 2, 1))  # NHWC -> NCWH, H & C are swapped
    x = self.conv2d(x)  # NCWH -> NCWH', H' is dim of size H * D. D is depth_multiplier
    x = _reshape_4d_to_5d(x)  # NCWH' -> NCWHD
    x = tf.transpose(x, perm=(0, 1, 2, 4, 3))  # NCWHD -> NCWDH
    x = tf.squeeze(x, axis=1)  # NCWDH -> NWDH, C is dim of size 1 and has been dropped
    x = tf.cast(tf.math.round(x), dtype=self.dtype)  # round and then saturate
    x = tf.clip_by_value(x, zero_value, one_value)
    assert x.shape.rank == 4

    # Dot product coeffs for packing the last axis: [1,2,4,8,16,32,64,128]
    po2_coeffs = tf.convert_to_tensor(2 ** np.arange(self.num_img_rows))
    po2_coeffs = tf.cast(po2_coeffs, dtype=self.dtype)

    # Pack 8 bits as a single number
    x = tf.math.reduce_sum(x * po2_coeffs, axis=-1)  # NWDH -> NWD, H has been dropped after reduce
    assert x.shape.rank == 3

    # Using IntegerLookup
    x = self.lookup(x)  # NWD -> NWD

    # Find max and argmax brightness
    idx_h = tf.math.argmax(x, axis=-1, output_type=self.dtype)  # NWD -> NW
    x = tf.gather(x, idx_h, axis=-1, batch_dims=2)  # NWD -> NW
    return (x, idx_h)


class Suppression(base_layer.Layer):
  def __init__(self, **kwargs):
    super(Suppression, self).__init__(**kwargs)

  @tf.function
  def call(self, inputs):
    x, idx_h = inputs

    # Non-max suppression
    x_padded = tf.pad(x, paddings=((0, 0), (1, 1)))  # ((pad_t, pad_b), (pad_l, pad_r))
    # Condition: x > x_left && x >= x_right
    mask = tf.math.logical_and(tf.math.greater(x, x_padded[:, :-2]),
                               tf.math.greater_equal(x, x_padded[:, 2:]))
    mask = tf.cast(mask, dtype=x.dtype)
    x = x * mask
    return (x, idx_h)


class ZoneSorting(base_layer.Layer):
  def __init__(self, **kwargs):
    super(ZoneSorting, self).__init__(**kwargs)

    # Import config
    config = get_config()
    self.num_emtf_tracks = config['num_emtf_tracks']

  @tf.function
  def call(self, inputs):
    x, idx_h = inputs

    # Gather top-k elements
    x, idx_w = tf.nn.top_k(
        x, k=self.num_emtf_tracks)  # NW -> NW', W' is dim of size num_emtf_tracks
    idx_h = tf.gather(idx_h, idx_w, axis=-1, batch_dims=1)  # NW -> NW'
    return (x, idx_h, idx_w)


class ZoneMerging(base_layer.Layer):
  def __init__(self, **kwargs):
    super(ZoneMerging, self).__init__(**kwargs)

    # Import config
    config = get_config()
    self.num_emtf_tracks = config['num_emtf_tracks']

  @tf.function
  def call(self, inputs):
    x, idx_h, idx_w = inputs

    # Gather top-k elements
    x, idx_d = tf.nn.top_k(
        x, k=self.num_emtf_tracks)  # NW' -> NW", W" is dim of size num_emtf_tracks
    idx_h = tf.gather(idx_h, idx_d, axis=-1, batch_dims=1)  # NW' -> NW"
    idx_w = tf.gather(idx_w, idx_d, axis=-1, batch_dims=1)  # NW' -> NW"
    num_emtf_tracks = tf.constant(self.num_emtf_tracks, dtype=idx_d.dtype)
    idx_d = tf.math.floordiv(idx_d, num_emtf_tracks)   # keep zone number
    return (x, idx_h, idx_w, idx_d)


class TrkBuilding(base_layer.Layer):
  def __init__(self, **kwargs):
    super(TrkBuilding, self).__init__(**kwargs)


class DupeRemoval(base_layer.Layer):
  def __init__(self, **kwargs):
    super(DupeRemoval, self).__init__(**kwargs)


class FullyConnect(base_layer.Layer):
  def __init__(self, **kwargs):
    super(FullyConnect, self).__init__(**kwargs)


def create_model():
  """Create the entire architecture composed of these layers.

  Make sure configure() and set_config() are called before creating the model.
  """
  # Import config
  config = get_config()
  #num_emtf_zones = config['num_emtf_zones']
  num_emtf_chambers = config['num_emtf_chambers']
  num_emtf_segments = config['num_emtf_segments']
  num_emtf_variables = config['num_emtf_variables']

  # Input layer
  input_shape = (num_emtf_chambers, num_emtf_segments, num_emtf_variables)
  inputs = tf.keras.layers.Input(shape=input_shape, dtype='int32', name='inputs')

  # Block of layers
  def block(x, i, timezone=0):
    x = Zoning(zone=i, timezone=timezone, name='zoning_{0}'.format(i))(x)
    x = Pooling(zone=i, name='pooling_{0}'.format(i))(x)
    x = Suppression(name='suppression_{0}'.format(i))(x)
    x = ZoneSorting(name='zonesorting_{0}'.format(i))(x)
    return x

  # Block concatenation
  def concat(x_list):
    x = [
        tf.keras.layers.Concatenate(axis=-1, name='concatenate_{0}'.format(i))(x)
        for (i, x) in enumerate(zip(*x_list))
    ]
    i = 0
    x = ZoneMerging(name='zonemerging_{0}'.format(i))(x)
    return x

  # Final block of layers
  def block_final(x):
    i = 0
    x = TrkBuilding(name='trkbuilding_{0}'.format(i))(x)
    x = DupeRemoval(name='duperemoval_{0}'.format(i))(x)
    return x

  # Build
  x = inputs
  x_zone0 = block(x, 0)
  x_zone1 = block(x, 1)
  x_zone2 = block(x, 2)
  x = concat([x_zone0, x_zone1, x_zone2])
  #x = (inputs,) + x
  #x = block_final(x)
  outputs = x

  # Model
  model = tf.keras.Model(inputs=inputs, outputs=outputs, name='endless_v3')
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
  return model


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
  'Zoning',
  'Pooling',
  'Suppression',
  'ZoneSorting',
  'ZoneMerging',
  'TrkBuilding',
  'DupeRemoval',
  'FullyConnect',
  'configure',
  'get_config',
  'get_nnet_model',
  'get_pattern_bank',
  'set_config',
  'set_nnet_model',
  'set_pattern_bank',
  'create_model',
  'get_datagen',
  'get_datagen_sparse',
]
