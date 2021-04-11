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
    #outputs = tf.map_fn(self.single_example_call, inputs, fn_output_signature=self.dtype)
    outputs = tf.vectorized_map(self.single_example_call, inputs)
    return outputs


class Pooling(base_layer.Layer):
  def __init__(self, zone, **kwargs):
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
    x_dtype = x.dtype

    if not x.shape.rank == 4:
      raise ValueError('inputs must be rank 4.')

    # Constants
    zero_value = tf.constant(0, dtype=x_dtype)
    one_value = tf.constant(1, dtype=x_dtype)

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
    x = tf.cast(tf.math.round(x), dtype=x_dtype)  # round and then saturate
    x = tf.clip_by_value(x, zero_value, one_value)
    assert x.shape.rank == 4

    # Dot product coeffs for packing the last axis: [1,2,4,8,16,32,64,128]
    po2_coeffs = tf.convert_to_tensor(2 ** np.arange(self.num_img_rows))
    po2_coeffs = tf.cast(po2_coeffs, dtype=x_dtype)

    # Pack 8 bits as a single number
    x = tf.math.reduce_sum(x * po2_coeffs, axis=-1)  # NWDH -> NWD, H has been dropped after reduce
    assert x.shape.rank == 3

    # Using IntegerLookup
    x = self.lookup(x)  # NWD -> NWD

    # Find max and argmax brightness
    idx_h = tf.math.argmax(x, axis=-1, output_type=x_dtype)  # NWD -> NW
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
    num_emtf_tracks = tf.constant(self.num_emtf_tracks, dtype=x.dtype)
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
    self.num_emtf_features_addl = config['num_emtf_features_addl']
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
    assert len(additional_features) == self.num_emtf_features_addl
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
    #outputs = tf.map_fn(self.single_example_call, inputs_flat,
    #                    fn_output_signature=(self.dtype, self.dtype))
    outputs = tf.vectorized_map(self.single_example_call, inputs_flat)
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
    self.num_emtf_sites_rm = config['num_emtf_sites_rm']
    self.num_emtf_tracks = config['num_emtf_tracks']
    self.site_rm_to_many_sites_lut = config['site_rm_to_many_sites_lut']
    self.site_to_site_rm_lut = config['site_to_site_rm_lut']

    # Derived from config
    self.invalid_marker_ph_seg = self.num_emtf_chambers * self.num_emtf_segments
    self.site_prior_min_value = np.min(self.site_to_site_rm_lut)
    self.site_prior_max_value = np.max(self.site_to_site_rm_lut)
    self.site_priors = np.max(self.site_to_site_rm_lut, axis=-1)
    self.site_targets = np.argmax(self.site_to_site_rm_lut, axis=-1)

  @tf.function
  def arg_first_occurrence(self, elems, targets):
    """Find the first occurrence indices where elems == targets."""
    num_elements = elems.shape[0]
    if num_elements is None:
      raise ValueError('elems dim 0 cannot be None.')
    indices = tf.range(num_elements, dtype=self.dtype)

    def loop_body(i, accum):
      elem, idx = elems[i], indices[i]
      boolean_mask = tf.math.equal(elem, targets)
      accum = tf.where(boolean_mask, tf.zeros_like(accum) + idx, accum)
      return (i - 1, accum)

    # Loop over range(num_elements) in reverse loop iteration
    n = tf.constant(num_elements, dtype=self.dtype)
    i = n - 1
    initial_value = n  # if not found, return n
    accum = tf.broadcast_to(initial_value, targets.shape)
    _, accum = tf.while_loop(
        lambda i, _: i >= 0, loop_body, (i, accum), parallel_iterations=num_elements)
    return accum

  @tf.function
  def reduce_trk_seg(self, trk_seg):
    invalid_marker_ph_seg = tf.constant(self.invalid_marker_ph_seg, dtype=self.dtype)
    site_prior_min_value = tf.constant(self.site_prior_min_value, dtype=self.dtype)
    site_prior_max_value = tf.constant(self.site_prior_max_value, dtype=self.dtype)

    # Prepare priorities and targets
    site_priors = tf.convert_to_tensor(self.site_priors)
    site_priors = tf.cast(site_priors, dtype=self.dtype)
    site_targets = tf.convert_to_tensor(self.site_targets)
    site_targets = tf.cast(site_targets, dtype=self.dtype)
    trk_numbers = tf.range(self.num_emtf_tracks, dtype=self.dtype)
    # Broadcast to the same shape
    site_priors = tf.broadcast_to(site_priors, trk_seg.shape)
    site_targets = tf.broadcast_to(site_targets, trk_seg.shape)
    trk_numbers = tf.broadcast_to(tf.expand_dims(trk_numbers, axis=-1), trk_seg.shape)

    # Suppress priorities where trk_seg is not valid
    trk_seg_v = tf.math.not_equal(trk_seg, invalid_marker_ph_seg)
    initial_value = site_prior_min_value  # for scatter
    site_priors = tf.where(trk_seg_v, site_priors, tf.zeros_like(site_priors) +
                           initial_value)

    # Scatter update the priorities and targets
    # Use [trk_numbers, site_targets] as indices, site_priors as updates.
    site_priors_flat = tf.reshape(site_priors, [-1])  # flatten
    site_targets_flat = tf.reshape(site_targets, [-1])
    trk_numbers_flat = tf.reshape(trk_numbers, [-1])
    # scatter_init is 2-D
    scatter_shape = trk_seg.shape[:-1] + [self.num_emtf_sites_rm]  # reduce the last dim
    scatter_init = tf.broadcast_to(initial_value, scatter_shape)
    upd_indices = tf.stack([trk_numbers_flat, site_targets_flat], axis=-1)
    upd_values = site_priors_flat
    assert scatter_init.shape.rank == upd_indices.shape[-1]
    assert upd_values.shape == upd_indices.shape[:-1]

    # Determine the priorities. After that, discover the trk_seg with these priorities.
    scatter = tf.tensor_scatter_nd_max(scatter_init, indices=upd_indices, updates=upd_values)
    # Translate priorities to indices_init. idx = max_priority - priority.
    # indices_init is an intermediate step.
    indices_init = tf.math.subtract(site_prior_max_value, scatter)
    indices_init = tf.where(tf.math.not_equal(scatter, initial_value), indices_init,
                            tf.zeros_like(indices_init))  # protect against invalid indices

    # Use indices_init to get the trk_seg indices.
    # Some transpose gymnastics is required to get the batch_dims to match.
    site_rm_to_many_sites_lut = tf.convert_to_tensor(self.site_rm_to_many_sites_lut)
    site_rm_to_many_sites_lut = tf.cast(site_rm_to_many_sites_lut, dtype=self.dtype)
    # Translate indices_init to indices via a lut.
    indices = tf.transpose(tf.gather(site_rm_to_many_sites_lut, tf.transpose(indices_init),
                                     axis=-1, batch_dims=1))
    # Gather from trk_seg by indices
    trk_seg_reduced = tf.gather(trk_seg, indices, axis=-1, batch_dims=1)
    return trk_seg_reduced

  @tf.function
  def find_dupes(self, trk_seg_reduced):
    invalid_marker_ph_seg = tf.constant(self.invalid_marker_ph_seg, dtype=self.dtype)
    zero_value = tf.constant(0, dtype=self.dtype)
    n0 = self.num_emtf_tracks  # n0 is not a tf.Tensor

    # trk_pairs are the (i,j) indices from the unrolled for loop over i & j.
    # For n = 4, trk_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]].
    trk_pairs = [
        (i, j) for i in range(n0 - 1) for j in range(i + 1, n0)
    ]
    trk_pairs = tf.convert_to_tensor(trk_pairs)
    trk_pairs = tf.cast(trk_pairs, dtype=self.dtype)
    assert trk_pairs.shape.rank == 2
    assert trk_pairs.shape == ((n0 * (n0 - 1)) // 2, 2)

    # Check for shared segments between each track pair
    # trk at lhs takes precedence over trk at rhs (i < j). If there are any shared segments,
    # trk at rhs is marked as duplicate.
    def _gather_by(t):
      return tf.gather(trk_seg_reduced, t)
    trk_seg_lhs = _gather_by(trk_pairs[..., 0])
    trk_seg_rhs = _gather_by(trk_pairs[..., 1])
    trk_seg_v_lhs = tf.math.not_equal(trk_seg_lhs, invalid_marker_ph_seg)
    trk_seg_v_rhs = tf.math.not_equal(trk_seg_rhs, invalid_marker_ph_seg)
    assert trk_seg_lhs.shape.rank == 2
    assert trk_seg_lhs.shape == (trk_pairs.shape[0], trk_seg_reduced.shape[-1])
    assert trk_seg_lhs.shape == trk_seg_rhs.shape

    # Compare trk_seg_lhs and trk_seg_rhs
    has_shared_seg_init = [
      trk_seg_v_lhs,
      trk_seg_v_rhs,
      tf.math.equal(trk_seg_lhs, trk_seg_rhs),
    ]
    # Reduce logical AND for all 3 conditions
    has_shared_seg = tf.math.reduce_all(tf.stack(has_shared_seg_init), axis=0)
    # Reduce logical OR for all sites
    has_any_shared_seg = tf.math.reduce_any(has_shared_seg, axis=-1)

    # Mark duplicates for removal
    upd_indices = trk_pairs[..., 1]  # trk at rhs
    upd_indices = tf.expand_dims(upd_indices, axis=-1)  # at least 2-D
    upd_values = tf.cast(has_any_shared_seg, dtype=self.dtype)
    # scatter is 1-D
    scatter_shape = (trk_seg_reduced.shape[0],)
    scatter = tf.scatter_nd(indices=upd_indices, updates=upd_values, shape=scatter_shape)
    dupes = tf.math.not_equal(scatter, zero_value)  # cnt != 0
    return dupes

  @tf.function
  def remove_dupes(self, trk_feat, trk_seg, dupes):
    invalid_marker_ph_seg = tf.constant(self.invalid_marker_ph_seg, dtype=self.dtype)
    mask_value = tf.constant(self.mask_value, dtype=self.dtype)

    # Calculate the target positions of all not-dupe tracks
    # When duplicates are found, they are replaced by the next not-dupe tracks.
    # For n = 4, suppose not_dupes = [1, 0, 0, 1], then elems = [0, 0, 0, 1] where
    # the first not-dupe trk has target pos = 0, and the second has target pos = 1.
    not_dupes = ~dupes
    elems = tf.math.cumsum(tf.cast(not_dupes, dtype=self.dtype)) - 1
    trk_numbers = tf.range(self.num_emtf_tracks, dtype=self.dtype)

    # Find indices where elems == trk_numbers
    indices_init = self.arg_first_occurrence(elems, trk_numbers)

    # Duplicates are no longer valid
    initial_value = tf.constant(elems.shape[0], dtype=self.dtype)
    trk_valid = tf.math.not_equal(indices_init, initial_value)
    indices = tf.where(trk_valid, indices_init,
                       tf.zeros_like(indices_init))  # protect against invalid indices

    # Collect the not-dupe tracks
    def _gather_from(t):
      return tf.gather(t, indices)
    trk_feat_rm_init = _gather_from(trk_feat)
    trk_seg_rm_init = _gather_from(trk_seg)
    # Mask trk_feat_rm and trk_seg_rm where trk is not valid
    trk_valid_exd = tf.expand_dims(trk_valid, axis=-1)
    trk_feat_rm = tf.where(trk_valid_exd, trk_feat_rm_init, tf.zeros_like(trk_feat_rm_init) +
                           mask_value)
    trk_seg_rm = tf.where(trk_valid_exd, trk_seg_rm_init, tf.zeros_like(trk_seg_rm_init) +
                          invalid_marker_ph_seg)
    outputs = (trk_feat_rm, trk_seg_rm)
    return outputs

  @tf.function
  def single_example_call(self, inputs):
    trk_feat, trk_seg = inputs

    # trk_feat shape is (num_emtf_tracks, num_emtf_features)
    # trk_seg shape is (num_emtf_tracks, num_emtf_sites)
    if not ((trk_feat.shape.rank == 2) and (trk_seg.shape.rank == 2)):
      raise ValueError('trk_feat and trk_seg must be rank 2.')

    # Reduce trk_seg last dim from num_emtf_sites (which is 12) to num_emtf_sites_rm (which is 5)
    trk_seg_reduced = self.reduce_trk_seg(trk_seg)
    assert (trk_seg_reduced.shape.rank == 2) and (trk_seg_reduced.shape[0] == trk_seg.shape[0])

    # Find and remove duplicates
    dupes = self.find_dupes(trk_seg_reduced)
    assert ((dupes.shape.rank == 1) and (dupes.shape[0] == trk_feat.shape[0]) and
            (dupes.shape[0] == trk_seg.shape[0]))

    outputs = self.remove_dupes(trk_feat, trk_seg, dupes)
    return outputs

  @tf.function
  def call(self, inputs):
    # Run on inputs individually
    #outputs = tf.map_fn(self.single_example_call, inputs,
    #                    fn_output_signature=(self.dtype, self.dtype))
    outputs = tf.vectorized_map(self.single_example_call, inputs)
    return outputs


class TrainFilter(base_layer.Layer):
  def __init__(self, **kwargs):
    super(TrainFilter, self).__init__(**kwargs)

    # Import config
    config = get_config()
    self.mask_value = config['mask_value']
    self.features_fields = config['features_fields']
    self.site_to_site_enum_lut = config['site_to_site_enum_lut']

  @tf.function
  def apply_train_rules(self, x):
    """Apply the following rules.

    1. theta_median != 0 and trk_qual != 0
    2. at least one station-1 hit (ME1/1, GE1/1, ME1/2, RE1/2, ME0)
       with one of the following requirements on station-2,3,4
       a. if there is ME1/1 or GE1/1, require 2 more stations
       b. if there is ME1/2 or RE1/2, require 1 more station
       c. if there is ME0,
          i.  if there is ME1/1 or GE1/1, require 1 more station
          ii. else, require 2 more stations
    """
    # Unstack
    fields = self.features_fields
    x_emtf_theta = x[..., fields.emtf_theta_begin:fields.emtf_theta_end]
    x_theta_median = x[..., fields.theta_median:(fields.theta_median + 1)]
    x_trk_qual = x[..., fields.trk_qual:(fields.trk_qual + 1)]
    x_dtype = x.dtype

    # Constants
    zero_value = tf.constant(0, dtype=x_dtype)
    one_value = tf.constant(1, dtype=x_dtype)
    two_value = tf.constant(2, dtype=x_dtype)
    mask_value = tf.constant(self.mask_value, dtype=x_dtype)

    # Rule 1
    rule1 = tf.math.logical_and(
        tf.math.not_equal(x_theta_median, zero_value),
        tf.math.not_equal(x_trk_qual, zero_value))

    # Rule 2
    def _count_nonzero(k):
      station_mask = tf.math.equal(site_to_site_enum_lut, tf.constant(k, dtype=x_dtype))
      return tf.math.count_nonzero(
          tf.math.logical_and(boolean_mask, station_mask), axis=-1, keepdims=True, dtype=x_dtype)
    site_to_site_enum_lut = tf.convert_to_tensor(self.site_to_site_enum_lut)
    site_to_site_enum_lut = tf.cast(site_to_site_enum_lut, dtype=x_dtype)
    boolean_mask = tf.math.not_equal(x_emtf_theta, mask_value)

    cnt_me14 = _count_nonzero(14)  # ME0
    cnt_me11 = _count_nonzero(11)  # ME1/1, GE1/1
    cnt_me12 = _count_nonzero(12)  # ME1/2, RE1/2
    cnt_me22 = _count_nonzero(22)  # ME2, GE2/1, RE2
    cnt_me23 = _count_nonzero(23)  # ME3, RE3
    cnt_me24 = _count_nonzero(24)  # ME4, RE4
    cnt_me20 = tf.math.count_nonzero(
        tf.stack([cnt_me22, cnt_me23, cnt_me24]), axis=0, dtype=x_dtype)

    rule2_a = tf.math.logical_and(
        tf.math.greater_equal(cnt_me11, one_value),
        tf.math.greater_equal(cnt_me20, two_value))
    rule2_b = tf.math.logical_and(
        tf.math.greater_equal(cnt_me12, one_value),
        tf.math.greater_equal(cnt_me20, one_value))
    rule2_c_i = tf.math.logical_and(
        tf.math.logical_and(
            tf.math.greater_equal(cnt_me14, one_value),
            tf.math.greater_equal(cnt_me11, one_value)),
        tf.math.greater_equal(cnt_me20, one_value))
    rule2_c_ii = tf.math.logical_and(
        tf.math.greater_equal(cnt_me14, one_value),
        tf.math.greater_equal(cnt_me20, two_value))
    rule2_init = [
      rule2_a,
      rule2_b,
      rule2_c_i,
      rule2_c_ii,
    ]
    rule2 = tf.math.reduce_any(tf.stack(rule2_init), axis=0)

    # All rules
    passed = tf.math.logical_and(rule1, rule2)
    return passed

  @tf.function
  def call(self, inputs):
    features = inputs  # features shape is (None, num_emtf_tracks, num_emtf_features)
    if not features.shape.rank == 3:
      raise ValueError('features must be rank 3.')

    outputs = (features, self.apply_train_rules(features))
    return outputs


class FullyConnect(base_layer.Layer):
  def __init__(self, **kwargs):
    super(FullyConnect, self).__init__(**kwargs)

    # Import config
    config = get_config()
    self.mask_value = config['mask_value']
    self.nnet_model = config['nnet_model']

  @tf.function
  def call(self, inputs):
    features, passed = inputs  # features shape is (None, num_emtf_tracks, num_emtf_features)
    if not features.shape.rank == 3:
      raise ValueError('features must be rank 3.')

    x = tf.cast(features, dtype=tf.keras.backend.floatx())  # NN model input must be float

    # Constants
    mask_value = tf.constant(self.mask_value, dtype=features.dtype)
    nan_value = tf.constant(np.nan, dtype=x.dtype)

    # Handle mask_value
    boolean_mask = tf.math.not_equal(features, mask_value)
    x = tf.where(boolean_mask, x, tf.zeros_like(x) + nan_value)

    # Using NN model
    x = tf.transpose(x, perm=(1, 0, 2))  # swap the first two axes
    x = tf.map_fn(self.nnet_model, x)    # call self.nnet_model(x) for each track
    x = tf.transpose(x, perm=(1, 0, 2))  # swap back
    outputs = (features, passed, x)
    return outputs


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
    i = 0
    x, x_cached = x[0], x[1:]
    x = TrainFilter(name='trainfilter_{}'.format(i))(x)
    x = FullyConnect(name='fullyconnect_{}'.format(i))(x)
    x = x[:1] + x_cached + x[1:]
    return x

  # Architecture/Layout
  # 1. Pattern recognition
  x = inputs
  x = block_pr(x)
  # 2. Feature extraction
  x = (inputs,) + x
  x = block_fe(x)
  # 3. Model inference
  x = block_mi(x)
  # Output
  outputs = x

  # Model
  model = tf.keras.Model(inputs=inputs, outputs=outputs, name='endless_v3')
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
  return model


def create_zone_hits(out_part, out_hits, out_simhits):
  assert isinstance(out_part, np.ndarray)
  assert isinstance(out_hits, tuple) and len(out_hits) == 2
  assert isinstance(out_simhits, tuple) and len(out_simhits) == 2

  # Import config
  config = get_config()
  part_fields = config['part_fields']
  num_emtf_zones = config['num_emtf_zones']

  # Unstack
  fields = part_fields
  out_part_zone = out_part[..., fields.part_zone].astype(np.int32)
  out_part_eta = np.abs(out_part[..., fields.part_eta])  # absolute value
  out_hits_rt = emtf_nnet.ragged.RaggedTensorValue(values=out_hits[0], row_splits=out_hits[1])
  out_simhits_rt = emtf_nnet.ragged.RaggedTensorValue(values=out_simhits[0], row_splits=out_simhits[1])

  # Make zone_mask
  atleast_1part_mask = ((0 <= out_part_zone) & (out_part_zone < num_emtf_zones) &
                        (1.0 <= out_part_eta) & (out_part_eta <= 2.4))
  atleast_1hit_mask = (out_simhits_rt.row_lengths >= 1) & (out_hits_rt.row_lengths >= 2)
  zone_mask = atleast_1part_mask & atleast_1hit_mask

  # Apply zone_mask
  zone_part = out_part[zone_mask]
  zone_hits = emtf_nnet.ragged.ragged_row_boolean_mask(out_hits_rt, zone_mask)
  zone_simhits = emtf_nnet.ragged.ragged_row_boolean_mask(out_simhits_rt, zone_mask)
  return (zone_part, (zone_hits.values, zone_hits.row_splits), (zone_simhits.values, zone_simhits.row_splits))


def _pack_zone_hits(zone_hits_rt):
  assert isinstance(zone_hits_rt.values, np.ndarray) and isinstance(zone_hits_rt.row_splits, np.ndarray)

  # Import config
  config = get_config()
  zone_hits_fields = config['zone_hits_fields']
  num_emtf_segments = config['num_emtf_segments']

  # Unstack
  values = zone_hits_rt.values
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
  cham_indices = zone_hits_rt.with_values(cham_indices)
  cham_values = zone_hits_rt.with_values(cham_values)
  return (cham_indices, cham_values)


def _to_dense(indices, values):
  assert isinstance(indices, np.ndarray) and isinstance(values, np.ndarray)
  assert indices.shape[0] == values.shape[0]

  # Import config
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
  cham_indices, cham_values = _pack_zone_hits(x_batch)
  # Concatenate sparsified chamber indices and values
  outputs = [
      np.concatenate((cham_indices[i], cham_values[i]), axis=-1)
      for i in range(len(cham_indices))
  ]
  return outputs


def _get_transformed_samples(x_batch):
  # Get sparsified chamber data
  cham_indices, cham_values = _pack_zone_hits(x_batch)
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
  'create_zone_hits',
  'get_datagen',
  'get_datagen_sparse',
]
