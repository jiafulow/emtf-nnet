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
    if 'dtype' in kwargs:
      kwargs.pop('dtype')
    kwargs['dtype'] = 'int32'
    super(TrkBuilding, self).__init__(**kwargs)

    # Import config
    config = get_config()
    self.mask_value = config['mask_value']
    self.num_emtf_zones = config['num_emtf_zones']
    self.num_emtf_timezones = config['num_emtf_timezones']
    self.num_emtf_chambers = config['num_emtf_chambers']
    self.num_emtf_segments = config['num_emtf_segments']
    self.num_emtf_sites = config['num_emtf_sites']
    self.num_emtf_features = config['num_emtf_features']
    self.coarse_emtf_strip = config['coarse_emtf_strip']
    self.min_emtf_strip = config['min_emtf_strip']
    self.fw_ph_diff_bitwidth = config['fw_ph_diff_bitwidth']
    self.fw_th_diff_bitwidth = config['fw_th_diff_bitwidth']
    self.fw_th_window = config['fw_th_window']
    self.fw_th_invalid = config['fw_th_invalid']
    self.num_img_cols = config['num_img_cols']
    self.patterns = config['patterns']
    self.packed_hits_fields = config['packed_hits_fields']
    self.site_to_img_row_luts = config['site_to_img_row_luts']
    self.site_to_chamber_lut = config['site_to_chamber_lut']
    self.site_number_lut = config['site_number_lut']
    self.trk_theta_indices = config['trk_theta_indices']
    self.trk_theta_indices_alt = config['trk_theta_indices_alt']
    self.trk_theta_indices_me1 = config['trk_theta_indices_me1']
    self.trk_bendable_indices = config['trk_bendable_indices']

    # Derived from config
    self.img_col_sector = self.num_img_cols // 2
    self.site_chamber_indices = np.concatenate(self.site_to_chamber_lut, axis=0)
    self.site_numbers = np.concatenate(self.site_number_lut, axis=0)
    self.invalid_marker_ph_seg = self.num_emtf_chambers * self.num_emtf_segments
    self.invalid_marker_ph_diff = (2 ** self.fw_ph_diff_bitwidth) - 1
    self.invalid_marker_th_diff = (2 ** self.fw_th_diff_bitwidth) - 1
    self.invalid_marker_th = self.fw_th_invalid

  def _find_emtf_img_col(self, x):
    # Returns: (emtf_phi - min_emtf_strip) // coarse_emtf_strip
    coarse_emtf_strip = tf.constant(self.coarse_emtf_strip, dtype=self.dtype)
    min_emtf_strip = tf.constant(self.min_emtf_strip, dtype=self.dtype)
    return tf.math.floordiv(x - min_emtf_strip, coarse_emtf_strip)

  def _find_emtf_img_col_inverse(self, x):
    # Returns: (emtf_img_col * coarse_emtf_strip) + (coarse_emtf_strip // 2) + min_emtf_strip
    coarse_emtf_strip = tf.constant(self.coarse_emtf_strip, dtype=self.dtype)
    half_coarse_emtf_strip = tf.constant(self.coarse_emtf_strip // 2, dtype=self.dtype)
    min_emtf_strip = tf.constant(self.min_emtf_strip, dtype=self.dtype)
    return (x * coarse_emtf_strip) + half_coarse_emtf_strip + min_emtf_strip

  def _find_median_of_three(self, x):
    # Returns median of 3
    invalid_marker_th = tf.constant(self.invalid_marker_th, dtype=self.dtype)
    zero_value = tf.constant(0, dtype=self.dtype)
    three_value = tf.constant(3, dtype=self.dtype)
    mask_value = tf.constant(self.mask_value, dtype=self.dtype)

    boolean_mask = tf.math.not_equal(x, invalid_marker_th)
    cnt = tf.math.count_nonzero(boolean_mask, dtype=self.dtype)
    # Exchange invalid_marker_th (which is 0) with mask_value
    x_tmp = tf.where(boolean_mask, x, tf.zeros_like(x) + mask_value)
    # median(a0, a1, a2) = max(min(a0, a1), min(max(a0, a1), a2))
    # If not all 3 values are valid, select the minimum.
    # If none of the values are valid, select the first.
    median_tmp = tf.cond(
        tf.math.not_equal(cnt, zero_value),  # cnt != 0
        lambda: tf.math.minimum(tf.math.minimum(x_tmp[0], x_tmp[1]), x_tmp[2]),
        lambda: x[0])
    median = tf.cond(
        tf.math.equal(cnt, three_value),  # cnt == 3
        lambda: tf.math.maximum(
            tf.math.minimum(x_tmp[0], x_tmp[1]),
            tf.math.minimum(tf.math.maximum(x_tmp[0], x_tmp[1]), x_tmp[2])),
        lambda: median_tmp)
    return median

  def _find_median_of_nine(self, x):
    # Returns median of 9 by finding median of 3 medians
    assert x.shape == (9,)
    x_tmp = [
        x[i * 3:(i + 1) * 3] for i in range(3)
    ]
    median_tmp = tf.map_fn(self._find_median_of_three, tf.stack(x_tmp))
    median = self._find_median_of_three(median_tmp)
    return median

  @tf.function
  def find_pattern_windows(self, trk_zone, trk_patt, trk_col):
    """Retrieves pattern windows given (zone, patt, col) indices."""
    site_to_img_row_luts = tf.convert_to_tensor(self.site_to_img_row_luts)
    site_to_img_row_luts = tf.cast(site_to_img_row_luts, dtype=self.dtype)
    # patterns shape is (num_emtf_zones, num_emtf_patterns, num_img_rows, 3).
    # A pattern window is encoded as (start, mid, stop). Hence the last dim is of size 3.
    patterns = tf.convert_to_tensor(self.patterns)
    patterns = tf.cast(patterns, dtype=self.dtype)
    # reference is set to (0, 0, 4, 1) which means zone 0 patt 'straightest' row 'ME2/1'
    # window_param 'mid'.
    reference = tf.constant(self.patterns[0, 0, 4, 1], dtype=self.dtype)

    # Look up window params, subtract reference, then shift by col.
    img_row_numbers = tf.gather(site_to_img_row_luts, trk_zone)
    trk_zone = tf.broadcast_to(trk_zone, img_row_numbers.shape)
    trk_patt = tf.broadcast_to(trk_patt, img_row_numbers.shape)
    trk_col = tf.broadcast_to(trk_col, img_row_numbers.shape)
    # batch_indices shape is (num_emtf_sites, 3), which is used as a list of 3-D indices.
    # windows_init shape is (num_emtf_sites, 3), which is a 1-D list of 3-D window params.
    batch_indices = tf.stack([trk_zone, trk_patt, img_row_numbers], axis=-1)
    #windows_init = tf.gather_nd(patterns, batch_indices)  # indexing into 4-D array

    # Convert 3-D indices into 1-D indices. This replaces tf.gather_nd() by tf.gather().
    def _lookup(batch_indices):
      a = batch_indices[..., 0]
      b = batch_indices[..., 1]
      c = batch_indices[..., 2]
      nb = tf.constant(patterns.shape[1], dtype=self.dtype)
      nc = tf.constant(patterns.shape[2], dtype=self.dtype)
      _shape = patterns.shape.as_list()
      _patterns = tf.reshape(patterns, [np.prod(_shape[:-1]), _shape[-1]])  # 4-D to 2-D
      _indices = (((a * nb) + b) * nc) + c  # 3-D to 1-D
      return tf.gather(_patterns, _indices)
    windows_init = _lookup(batch_indices)
    windows = (windows_init - reference) + tf.expand_dims(trk_col, axis=-1)
    # Translate from img_col to emtf_phi
    phi_windows = self._find_emtf_img_col_inverse(windows)
    return (windows, phi_windows)

  @tf.function
  def unsorted_segment_argmin(self, data, segment_ids, num_segments):
    """Computes the argmin along segments of a tensor.

    Similar to tf.math.unsorted_segment_min().
    """
    def _as_numpy_dtype(dtype):
      return np.dtype(dtype)
    max_value = tf.constant(np.iinfo(_as_numpy_dtype(self.dtype)).max, dtype=self.dtype)

    def loop_body(i, ta):
      # Find argmin over elements with a matched segment id
      boolean_mask = tf.math.equal(segment_ids, i)
      augmented_data = tf.where(boolean_mask, data, tf.zeros_like(data) + max_value)
      idx_min = tf.math.argmin(augmented_data, axis=-1, output_type=self.dtype)
      ta = ta.write(i, idx_min)
      return (i + 1, ta)

    # Loop over range(num_segments)
    i = tf.constant(0, dtype=self.dtype)
    n = tf.constant(num_segments, dtype=self.dtype)
    ta = tf.TensorArray(dtype=self.dtype, size=n, dynamic_size=False, infer_shape=True)
    _, ta = tf.while_loop(
        lambda i, _: i < n, loop_body, (i, ta), parallel_iterations=num_segments)
    return ta.stack()

  @tf.function
  def find_phi_median(self, trk_col):
    # Translate from img_col to emtf_phi
    median = self._find_emtf_img_col_inverse(trk_col)
    return median

  @tf.function
  def find_theta_median(self, feat_emtf_theta_init):
    # Returns a best guess value of theta_median
    invalid_marker_th = tf.constant(self.invalid_marker_th, dtype=self.dtype)
    zero_value = tf.constant(0, dtype=self.dtype)

    # Prepare theta_values, theta_values_alt, theta_values_me1 for different scenarios.
    # theta_values are canonical stations 2,3,4 thetas; theta_values_alt are alternative
    # stations 2,3,4 thetas; theta_values_me1 are station 1 thetas.
    # trk_theta_indices shape is (9, 2) for 9 thetas and 2-fold ambiguity. It is used
    # as a list of 2-D indices.
    def _gather_by(t):
      return tf.gather_nd(feat_emtf_theta_init, t)  # indexing into 3-D array
    trk_theta_indices = tf.convert_to_tensor(self.trk_theta_indices)
    trk_theta_indices = tf.cast(trk_theta_indices, dtype=self.dtype)
    trk_theta_indices_alt = tf.convert_to_tensor(self.trk_theta_indices_alt)
    trk_theta_indices_alt = tf.cast(trk_theta_indices_alt, dtype=self.dtype)
    trk_theta_indices_me1 = tf.convert_to_tensor(self.trk_theta_indices_me1)
    trk_theta_indices_me1 = tf.cast(trk_theta_indices_me1, dtype=self.dtype)
    theta_values = _gather_by(trk_theta_indices)
    theta_values_alt = _gather_by(trk_theta_indices_alt)
    theta_values_me1 = _gather_by(trk_theta_indices_me1)
    assert (theta_values.shape.rank == 1) and (theta_values.shape[0] == 9)
    assert ((theta_values_alt.shape == theta_values.shape) and
            (theta_values_me1.shape == theta_values.shape))

    # Select from either theta_values or theta_values_alt
    boolean_mask = tf.math.not_equal(theta_values, invalid_marker_th)
    theta_values = tf.where(boolean_mask, theta_values, theta_values_alt)
    # Select from either theta_values or theta_values_me1
    boolean_mask = tf.math.not_equal(theta_values, invalid_marker_th)
    cnt = tf.math.count_nonzero(boolean_mask, dtype=self.dtype)
    median = tf.cond(
        tf.math.not_equal(cnt, zero_value),  # cnt != 0
        lambda: self._find_median_of_nine(theta_values),
        lambda: self._find_median_of_nine(theta_values_me1))
    return median

  @tf.function
  def resolve_theta_ambiguity(self, feat_emtf_theta_init, theta_median):
    # Returns best guess values of the thetas
    invalid_marker_th = tf.constant(self.invalid_marker_th, dtype=self.dtype)
    invalid_marker_th_diff = tf.constant(self.invalid_marker_th_diff, dtype=self.dtype)
    fw_th_window = tf.constant(self.fw_th_window, dtype=self.dtype)

    # Calculate abs(delta-theta) from theta_median. Suppress invalid th_diff.
    # Also apply theta window cut.
    boolean_mask = tf.math.not_equal(feat_emtf_theta_init, invalid_marker_th)
    th_diff = tf.math.abs(tf.math.subtract(feat_emtf_theta_init, theta_median))
    th_diff = tf.where(boolean_mask, th_diff, tf.zeros_like(th_diff) +
                       invalid_marker_th_diff)
    # Condition: |x| < window
    boolean_mask = tf.math.less(th_diff, fw_th_window)
    th_diff = tf.where(boolean_mask, th_diff, tf.zeros_like(th_diff) +
                       invalid_marker_th_diff)

    # Resolve theta ambiguity by picking the min th_diff elements
    th_diff_min = tf.math.reduce_min(th_diff, axis=-1)
    th_diff_min_valid = tf.math.not_equal(th_diff_min, invalid_marker_th_diff)
    feat_emtf_theta_best = tf.where(tf.math.equal(th_diff_min, th_diff[..., 0]),
                                    feat_emtf_theta_init[..., 0], feat_emtf_theta_init[..., 1])
    return (th_diff_min, th_diff_min_valid, feat_emtf_theta_best)

  @tf.function
  def extract_features(self,
                       feat_emtf_phi,
                       feat_emtf_bend,
                       feat_emtf_theta_best,
                       feat_emtf_qual_best,
                       feat_emtf_time,
                       additional_features):
    # Concatenate the tensors
    assert (36 + len(additional_features)) == self.num_emtf_features
    trk_bendable_indices = tf.convert_to_tensor(self.trk_bendable_indices)
    trk_bendable_indices = tf.cast(trk_bendable_indices, dtype=self.dtype)
    features = [
      feat_emtf_phi,
      feat_emtf_theta_best,
      tf.gather(feat_emtf_bend, trk_bendable_indices),
      tf.gather(feat_emtf_qual_best, trk_bendable_indices),
      tf.stack(additional_features),
    ]
    return tf.concat(features, axis=-1)

  @tf.function
  def single_example_call(self, inputs):
    x, trk_qual, trk_patt, trk_col, trk_zone = inputs

    # x shape is (C, S, V). C is num of chambers, S is num of segments, V is num of variables.
    # trk_qual and friends are scalars.
    if not x.shape.rank == 3:
      raise ValueError('inputs[0] must be rank 3.')
    if not ((trk_qual.shape.rank == 0) and (trk_patt.shape.rank == 0) and
            (trk_col.shape.rank == 0) and (trk_zone.shape.rank == 0)):
      raise ValueError('Each of inputs[1:] must be rank 0.')

    trk_tzone = tf.zeros_like(trk_qual)  # default timezone
    trk_bx = tf.zeros_like(trk_qual)  # default bx

    # Unstack
    fields = self.packed_hits_fields
    x_emtf_phi_init = x[..., fields.emtf_phi]
    x_emtf_bend_init = x[..., fields.emtf_bend]
    x_emtf_theta1_init = x[..., fields.emtf_theta1]
    x_emtf_theta2_init = x[..., fields.emtf_theta2]
    x_emtf_qual1_init = x[..., fields.emtf_qual1]
    #x_emtf_qual2_init = x[..., fields.emtf_qual2]  # unused
    x_emtf_time_init = x[..., fields.emtf_time]
    x_zones_init = x[..., fields.zones]
    x_tzones_init = x[..., fields.tzones]
    x_valid_init = x[..., fields.valid]

    # Constants
    zero_value = tf.constant(0, dtype=self.dtype)
    one_value = tf.constant(1, dtype=self.dtype)
    mask_value = tf.constant(self.mask_value, dtype=self.dtype)
    min_emtf_strip = tf.constant(self.min_emtf_strip, dtype=self.dtype)
    img_col_sector = tf.constant(self.img_col_sector, dtype=self.dtype)
    invalid_marker_ph_seg = tf.constant(self.invalid_marker_ph_seg, dtype=self.dtype)
    invalid_marker_ph_diff = tf.constant(self.invalid_marker_ph_diff, dtype=self.dtype)

    # Find site chambers
    # Result shape is (C', S). C' is num of chambers in all sites, S is num of segments.
    def _gather_from(t):
      return tf.gather(t, site_chamber_indices)
    site_chamber_indices = tf.convert_to_tensor(self.site_chamber_indices)
    site_chamber_indices = tf.cast(site_chamber_indices, dtype=self.dtype)
    x_emtf_phi = _gather_from(x_emtf_phi_init)
    x_emtf_bend = _gather_from(x_emtf_bend_init)
    x_emtf_theta1 = _gather_from(x_emtf_theta1_init)
    x_emtf_theta2 = _gather_from(x_emtf_theta2_init)
    x_emtf_qual1 = _gather_from(x_emtf_qual1_init)
    #x_emtf_qual2 = _gather_from(x_emtf_qual2_init)  # unused
    x_emtf_time = _gather_from(x_emtf_time_init)
    x_zones = _gather_from(x_zones_init)
    x_tzones = _gather_from(x_tzones_init)
    x_valid = _gather_from(x_valid_init)
    assert (x_emtf_phi.shape.rank == 2) and (x_emtf_phi.shape[0] == site_chamber_indices.shape[0])

    # Find cols, sites
    site_numbers = tf.convert_to_tensor(self.site_numbers)
    site_numbers = tf.cast(site_numbers, dtype=self.dtype)
    # Translate from emtf_phi to img_col
    selected_cols = self._find_emtf_img_col(x_emtf_phi)
    selected_sites = tf.broadcast_to(tf.expand_dims(site_numbers, axis=-1), selected_cols.shape)
    assert selected_cols.shape == selected_sites.shape

    # Valid for this zone/timezone
    the_zone_bitpos = tf.constant((self.num_emtf_zones - 1), dtype=self.dtype) - trk_zone
    the_tzone_bitpos = tf.constant((self.num_emtf_timezones - 1), dtype=self.dtype) - trk_tzone
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

    # Find windows for this zone/pattern/col
    # windows shape is (num_emtf_sites, 3). A pattern window is encoded as (start, mid, stop).
    windows, phi_windows = self.find_pattern_windows(trk_zone, trk_patt, trk_col)
    assert (windows.shape.rank == 2) and (windows.shape[0] == self.num_emtf_sites)
    assert windows.shape == phi_windows.shape

    # Valid for this pattern window
    # selected_sites is used as a 2-D tensor of 1-D indices.
    def _gather_from_windows(t):
      return tf.gather(t, selected_sites)
    windows_start = _gather_from_windows(windows[..., 0])
    windows_stop = _gather_from_windows(windows[..., 2])
    # Condition: start <= x <= stop
    boolean_mask_patt_init = [
      tf.math.less_equal(windows_start, selected_cols),
      tf.math.less_equal(selected_cols, windows_stop),
    ]
    boolean_mask_patt = tf.math.reduce_all(tf.stack(boolean_mask_patt_init), axis=0)
    assert boolean_mask_patt.shape == boolean_mask.shape

    # Valid for this zone/timezone/pattern window
    boolean_mask = tf.math.logical_and(boolean_mask, boolean_mask_patt)

    # Calculate abs(delta-phi) for this pattern
    phi_windows_mid = _gather_from_windows(phi_windows[..., 1])
    ph_diff = tf.math.abs(tf.math.subtract(x_emtf_phi, phi_windows_mid))
    ph_diff = tf.clip_by_value(ph_diff, zero_value, invalid_marker_ph_diff)
    assert (ph_diff.shape == boolean_mask.shape) and (ph_diff.shape == selected_sites.shape)

    # Suppress invalid ph_diff before finding the min ph_diff
    ph_diff = tf.where(boolean_mask, ph_diff, tf.zeros_like(ph_diff) + invalid_marker_ph_diff)

    # Find the min ph_diff for all sites simultaneously
    ph_diff_flat = tf.reshape(ph_diff, [-1])  # flatten
    selected_sites_flat = tf.reshape(selected_sites, [-1])
    ph_diff_argmin = self.unsorted_segment_argmin(
        ph_diff_flat, selected_sites_flat, self.num_emtf_sites)
    ph_diff_min = tf.gather(ph_diff_flat, ph_diff_argmin)
    ph_diff_argmin_valid = tf.math.not_equal(ph_diff_min, invalid_marker_ph_diff)
    assert (ph_diff_argmin.shape.rank == 1) and (ph_diff_argmin.shape[0] == self.num_emtf_sites)
    assert ph_diff_argmin_valid.shape == ph_diff_argmin.shape

    # Take features corresponding to the min ph_diff elements.
    # Set to 0 where ph_diff_min is not valid. Currently, this works as invalid_marker_th
    # is 0. Or else, should use mask_value.
    def _take_feature_values(t):
      t = tf.reshape(t, [-1])  # flatten
      t = tf.gather(t, ph_diff_argmin)
      t = tf.where(ph_diff_argmin_valid, t, tf.zeros_like(t))  # set to 0
      return t
    feat_emtf_phi = _take_feature_values(x_emtf_phi)
    feat_emtf_bend = _take_feature_values(x_emtf_bend)
    feat_emtf_theta1 = _take_feature_values(x_emtf_theta1)
    feat_emtf_theta2 = _take_feature_values(x_emtf_theta2)
    feat_emtf_qual1 = _take_feature_values(x_emtf_qual1)
    #feat_emtf_qual2 = _take_feature_values(x_emtf_qual2)
    feat_emtf_time = _take_feature_values(x_emtf_time)

    # Find phi_median and theta_median
    # Due to theta ambiguity, there are multiple scenarios for finding theta_median.
    phi_median = self.find_phi_median(trk_col)
    phi_median_signed = tf.math.subtract(phi_median, self.find_phi_median(img_col_sector))
    feat_emtf_theta_init = tf.stack([feat_emtf_theta1, feat_emtf_theta2], axis=-1)
    theta_median = self.find_theta_median(feat_emtf_theta_init)

    # Calculate abs(delta-theta) from theta_median. Then, resolve theta ambiguity by
    # picking the min th_diff elements.
    th_diff_min, th_diff_min_valid, feat_emtf_theta_best = self.resolve_theta_ambiguity(
        feat_emtf_theta_init, theta_median)
    assert (th_diff_min.shape.rank == 1) and (th_diff_min.shape[0] == self.num_emtf_sites)
    assert th_diff_min_valid.shape == th_diff_min.shape
    assert feat_emtf_theta_best.shape == th_diff_min.shape

    # For phi and theta values, subtract phi_median and theta_median respectively.
    feat_emtf_phi = tf.math.subtract(feat_emtf_phi, phi_median)
    feat_emtf_theta_best = tf.math.subtract(feat_emtf_theta_best, theta_median)
    feat_emtf_qual_best = feat_emtf_qual1  # just a rename

    # Mask features where th_diff_min is not valid.
    def _mask_feature_values(t):
      return tf.where(th_diff_min_valid, t, tf.zeros_like(t) + mask_value)
    feat_emtf_phi = _mask_feature_values(feat_emtf_phi)
    feat_emtf_bend = _mask_feature_values(feat_emtf_bend)
    feat_emtf_theta_best = _mask_feature_values(feat_emtf_theta_best)
    feat_emtf_qual_best = _mask_feature_values(feat_emtf_qual_best)
    feat_emtf_time = _mask_feature_values(feat_emtf_time)

    # Finally, extract features including additional features
    additional_features = [
      phi_median_signed,
      theta_median,
      trk_qual,
      trk_bx,
    ]
    trk_feat = self.extract_features(feat_emtf_phi, feat_emtf_bend, feat_emtf_theta_best,
                                     feat_emtf_qual_best, feat_emtf_time, additional_features)
    # Also keep ph_diff_argmin indices
    trk_seg = tf.where(th_diff_min_valid, ph_diff_argmin, tf.zeros_like(ph_diff_argmin) +
                       invalid_marker_ph_seg)  # mask indices where th_diff_min is not valid
    outputs = (trk_feat, trk_seg)
    return outputs

  @tf.function
  def call(self, inputs):
    x, trk_qual, trk_patt, trk_col, trk_zone = inputs

    # trk_qual and friends have shape (None, num_emtf_tracks)
    if not x.shape.rank == 4:
      raise ValueError('inputs[0] must be rank 4.')
    if not ((trk_qual.shape.rank == 2) and (trk_patt.shape.rank == 2) and
            (trk_col.shape.rank == 2) and (trk_zone.shape.rank == 2)):
      raise ValueError('Each of inputs[1:] must be rank 2.')

    batch_dim = trk_qual.shape[0]
    non_batch_dim = trk_qual.shape[-1]
    if batch_dim is None:
      batch_dim = tf.shape(trk_qual)[0]

    # Duplicate x for each track. Unroll trk_qual and friends.
    inputs_flat = (
      tf.repeat(x, repeats=non_batch_dim, axis=0),  # duplicate
      tf.reshape(trk_qual, [-1]),  # flatten
      tf.reshape(trk_patt, [-1]),
      tf.reshape(trk_col, [-1]),
      tf.reshape(trk_zone, [-1]),
    )

    # Run on inputs individually
    outputs = tf.map_fn(self.single_example_call, inputs_flat,
                        fn_output_signature=(self.dtype, self.dtype))
    #outputs = tf.vectorized_map(self.single_example_call, inputs_flat)
    assert isinstance(outputs, tuple) and len(outputs) == 2

    # Roll back to the shape of trk_qual and friends.
    output_shapes = ([batch_dim, non_batch_dim, o.shape[-1]] for o in outputs)
    outputs = tuple(tf.reshape(o, s) for (o, s) in zip(outputs, output_shapes))
    return outputs


class DupeRemoval(base_layer.Layer):
  def __init__(self, **kwargs):
    if 'dtype' in kwargs:
      kwargs.pop('dtype')
    kwargs['dtype'] = 'int32'
    super(DupeRemoval, self).__init__(**kwargs)

    # Import config
    config = get_config()
    self.mask_value = config['mask_value']
    self.num_emtf_chambers = config['num_emtf_chambers']
    self.num_emtf_segments = config['num_emtf_segments']
    self.num_emtf_features = config['num_emtf_features']
    self.num_emtf_tracks = config['num_emtf_tracks']
    self.site_rm_member_lut = config['site_rm_member_lut']

    # Derived from config
    self.invalid_marker_ph_seg = self.num_emtf_chambers * self.num_emtf_segments

  @tf.function
  def find_dupes(self, trk_seg_reduced):
    invalid_marker_ph_seg = tf.constant(self.invalid_marker_ph_seg, dtype=self.dtype)
    trk_seg_reduced_v = tf.math.not_equal(trk_seg_reduced, invalid_marker_ph_seg)

    dupes_init = [
        tf.constant(False) for _ in range(self.num_emtf_tracks)
    ]

    # Mark duplicates for removal
    for i in range(self.num_emtf_tracks - 1):
      for j in range(i + 1, self.num_emtf_tracks):
        # Same shape as trk_seg_reduced
        has_shared_seg_init = tf.math.logical_and(
            tf.math.logical_and(trk_seg_reduced_v[i], trk_seg_reduced_v[j]),
            tf.math.equal(trk_seg_reduced[i], trk_seg_reduced[j]))
        # Logical OR reduce to a scalar
        has_shared_seg = tf.math.reduce_any(has_shared_seg_init, axis=None)
        dupes_init[j] = tf.math.logical_or(dupes_init[j], has_shared_seg)

    dupes = tf.stack(dupes_init)
    return dupes

  @tf.function
  def remove_dupes(self, trk_feat, trk_seg, dupes):
    invalid_marker_ph_seg = tf.constant(self.invalid_marker_ph_seg, dtype=self.dtype)
    mask_value = tf.constant(self.mask_value, dtype=self.dtype)
    not_dupes = ~dupes
    not_dupes_cumsum = tf.math.cumsum(tf.cast(not_dupes, dtype=self.dtype))

    trk_feat_rm_init = tf.broadcast_to(mask_value, trk_feat.shape)
    trk_feat_rm_init = tf.unstack(trk_feat_rm_init)
    trk_seg_rm_init = tf.broadcast_to(invalid_marker_ph_seg, trk_seg.shape)
    trk_seg_rm_init = tf.unstack(trk_seg_rm_init)

    # Copy if not duplicate
    for i in range(self.num_emtf_tracks):
      for j in reversed(range(i, self.num_emtf_tracks)):  # priority encoder with reverse priority
        trk_feat_tmp = trk_feat_rm_init[i]  # write after read
        trk_feat_rm_init[i] = tf.cond(
            tf.math.equal(not_dupes_cumsum[j], tf.constant(i + 1, dtype=self.dtype)),
            lambda: trk_feat[j],
            lambda: trk_feat_tmp)
        trk_seg_tmp = trk_seg_rm_init[i]  # write after read
        trk_seg_rm_init[i] = tf.cond(
            tf.math.equal(not_dupes_cumsum[j], tf.constant(i + 1, dtype=self.dtype)),
            lambda: trk_seg[j],
            lambda: trk_seg_tmp)

    trk_feat_rm = tf.stack(trk_feat_rm_init)
    trk_seg_rm = tf.stack(trk_seg_rm_init)
    return (trk_feat_rm, trk_seg_rm)

  @tf.function
  def single_example_call(self, inputs):
    trk_feat, trk_seg = inputs

    # trk_feat shape is (num_emtf_tracks, num_emtf_features)
    # trk_seg shape is (num_emtf_tracks, num_emtf_sites)
    if not (trk_feat.shape.rank == 2) and (trk_seg.shape.rank == 2):
      raise ValueError('trk_feat and trk_seg must be rank 2.')

    # Constants
    invalid_marker_ph_seg = tf.constant(self.invalid_marker_ph_seg, dtype=self.dtype)

    # Reduce trk_seg last dim from num_emtf_sites (which is 12) to num_emtf_sites_rm (which is 5)
    def reduce_trk_seg(trk_seg):
      trk_seg_reduced_init = [
          invalid_marker_ph_seg for _ in range(len(self.site_rm_member_lut))
      ]

      # Loop over the nested site_rm_member_lut
      for i in range(len(self.site_rm_member_lut)):
        for j in reversed(self.site_rm_member_lut[i]):  # priority encoder with reverse priority
          trk_seg_tmp = trk_seg_reduced_init[i]  # write after read
          trk_seg_reduced_init[i] = tf.cond(
              tf.math.not_equal(trk_seg[j], invalid_marker_ph_seg),
              lambda: trk_seg[j],
              lambda: trk_seg_tmp)

      trk_seg_reduced = tf.stack(trk_seg_reduced_init)
      return trk_seg_reduced

    # Run on individual tracks
    trk_seg_reduced = tf.map_fn(reduce_trk_seg, trk_seg)
    assert (trk_seg_reduced.shape.rank == 2) and (trk_seg_reduced.shape[0] == trk_seg.shape[0])

    # Find and remove duplicates
    dupes = self.find_dupes(trk_seg_reduced)
    trk_feat_rm, trk_seg_rm = self.remove_dupes(trk_feat, trk_seg, dupes)
    return (trk_feat_rm, trk_seg_rm)

  @tf.function
  def call(self, inputs):
    # Run on inputs individually
    outputs = tf.map_fn(self.single_example_call, inputs,
                        fn_output_signature=(self.dtype, self.dtype))
    #outputs = tf.vectorized_map(self.single_example_call, inputs)
    return outputs


class FullyConnect(base_layer.Layer):
  def __init__(self, **kwargs):
    super(FullyConnect, self).__init__(**kwargs)


def create_model():
  """Create the entire architecture composed of these layers.

  Make sure configure() and set_config() are called before creating the model.
  """
  # Import config
  config = get_config()
  num_emtf_zones = config['num_emtf_zones']
  num_emtf_chambers = config['num_emtf_chambers']
  num_emtf_segments = config['num_emtf_segments']
  num_emtf_variables = config['num_emtf_variables']

  # Input
  input_shape = (num_emtf_chambers, num_emtf_segments, num_emtf_variables)
  inputs = tf.keras.layers.Input(shape=input_shape, dtype='int32', name='inputs')

  # Building blocks
  # 1. Pattern recognition
  def block_pr(x):
    def branch(x, i):
      timezone = 0  # default timezone
      x = Zoning(zone=i, timezone=timezone, name='zoning_{}'.format(i))(x)
      x = Pooling(zone=i, name='pooling_{}'.format(i))(x)
      x = Suppression(name='suppression_{}'.format(i))(x)
      x = ZoneSorting(name='zonesorting_{}'.format(i))(x)
      return x

    def merge(x_list):
      x = [
          tf.keras.layers.Concatenate(axis=-1, name='concatenate_{}'.format(i))(x)
          for (i, x) in enumerate(zip(*x_list))
      ]
      i = 0
      x = ZoneMerging(name='zonemerging_{}'.format(i))(x)
      return x

    x_list = [
        branch(x, i) for i in range(num_emtf_zones)
    ]
    x = merge(x_list)
    return x

  # 2. Feature extraction
  def block_fe(x):
    i = 0
    x = TrkBuilding(name='trkbuilding_{}'.format(i))(x)
    x = DupeRemoval(name='duperemoval_{}'.format(i))(x)
    return x

  # 3. Model inference
  def block_mi(x):
    #i = 0
    #x = FullyConnect(name='fullyconnect_{}'.format(i))(x)  #FIXME
    return x

  # Architecture/Layout
  # 1. Pattern recognition
  x = inputs
  x = block_pr(x)
  # 2. Feature extraction
  x = (inputs,) + x
  x = block_fe(x)
  # 3. Model inference
  x, x_cached = x[0], x[1:]
  x = block_mi(x)
  x = (x,) + x_cached
  # Output
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
