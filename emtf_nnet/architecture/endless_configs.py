"""Architecture configs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import warnings

from emtf_nnet.slices import (gather_indices_by_values,
                              gather_inputs_by_outputs)

_CONFIG = None
_NNET_MODEL = None
_PATTERN_BANK = None


def set_config(config):
  global _CONFIG
  _CONFIG = config


def get_config():
  if _CONFIG is None:
    raise ValueError('config has not been set. Please use set_config() to set it.')
  return _CONFIG


def set_nnet_model(nnet_model):
  global _NNET_MODEL
  _NNET_MODEL = nnet_model


def get_nnet_model():
  if _NNET_MODEL is None:
    raise ValueError('nnet_model has not been set. Please use set_nnet_model() to set it.')
  return _NNET_MODEL


def set_pattern_bank(pattern_bank):
  global _PATTERN_BANK
  _PATTERN_BANK = pattern_bank


def get_pattern_bank():
  if _PATTERN_BANK is None:
    raise ValueError('pattern_bank has not been set. Please use set_pattern_bank() to set it.')
  return _PATTERN_BANK


def configure_v3(strict=True):
  config = {}

  # From emtf_utils.py
  mask_value = 999999  # obtained from ma_fill_value()

  # From emtf_algos.py
  num_emtf_zones = 3
  num_emtf_timezones = 3
  num_emtf_chambers = 115
  num_emtf_segments = 2
  num_emtf_variables = 13
  num_emtf_sites = 12
  num_emtf_sites_rm = 5
  num_emtf_tracks = 4
  num_emtf_patterns = 7
  num_emtf_features = 36 + 4
  num_emtf_features_addl = 4
  coarse_emtf_strip = 8 * 2  # 'doublestrip' unit
  min_emtf_strip = (315 - 288) * coarse_emtf_strip  # 7.2 deg
  max_emtf_strip = (315 - 0) * coarse_emtf_strip  # 84 deg
  fw_ph_diff_bitwidth = 11
  fw_th_diff_bitwidth = 6
  fw_th_window = 8
  fw_th_invalid = 0

  config['mask_value'] = mask_value
  config['num_emtf_zones'] = num_emtf_zones
  config['num_emtf_timezones'] = num_emtf_timezones
  config['num_emtf_chambers'] = num_emtf_chambers
  config['num_emtf_segments'] = num_emtf_segments
  config['num_emtf_variables'] = num_emtf_variables
  config['num_emtf_sites'] = num_emtf_sites
  config['num_emtf_sites_rm'] = num_emtf_sites_rm
  config['num_emtf_tracks'] = num_emtf_tracks
  config['num_emtf_patterns'] = num_emtf_patterns
  config['num_emtf_features'] = num_emtf_features
  config['num_emtf_features_addl'] = num_emtf_features_addl
  config['coarse_emtf_strip'] = coarse_emtf_strip
  config['min_emtf_strip'] = min_emtf_strip
  config['max_emtf_strip'] = max_emtf_strip
  config['fw_ph_diff_bitwidth'] = fw_ph_diff_bitwidth
  config['fw_th_diff_bitwidth'] = fw_th_diff_bitwidth
  config['fw_th_window'] = fw_th_window
  config['fw_th_invalid'] = fw_th_invalid

  # Image format
  num_img_rows = 8
  num_img_cols = 288  # 80 degrees
  num_img_channels = 1
  num_box_cols = 111  # 30 degrees
  image_shape = (num_img_rows, num_img_cols, num_img_channels)

  config['num_img_rows'] = num_img_rows
  config['num_img_cols'] = num_img_cols
  config['num_img_channels'] = num_img_channels
  config['num_box_cols'] = num_box_cols
  config['image_shape'] = image_shape

  # Pattern bank
  try:
    pattern_bank = get_pattern_bank()
  except ValueError as e:
    if strict:
      raise ValueError(e)
    else:
      warnings.warn(str(e), UserWarning)
    pattern_bank = None

  if pattern_bank is not None:
    _patterns_shape = (num_emtf_zones, num_emtf_patterns, num_img_rows, 3)
    _patt_filters_shape = (num_emtf_zones, num_img_channels, num_box_cols, num_img_rows,
                           num_emtf_patterns)
    _patt_brightness_shape = (num_emtf_zones, 2 ** num_img_rows)
    assert pattern_bank.patterns.shape == _patterns_shape
    assert pattern_bank.patt_filters.shape == _patt_filters_shape
    assert pattern_bank.patt_brightness.shape == _patt_brightness_shape
    config['patterns'] = pattern_bank.patterns
    config['patt_filters'] = pattern_bank.patt_filters
    config['patt_brightness'] = pattern_bank.patt_brightness

  # particle info
  _part_fields = [
    'part_invpt',
    'part_eta',
    'part_phi',
    'part_vx',
    'part_vy',
    'part_vz',
    'part_d0',
    'part_sector',
    'part_zone',
  ]
  PartFields = collections.namedtuple('PartFields', _part_fields)
  part_fields = PartFields(*range(len(_part_fields)))
  config['part_fields'] = part_fields

  # zone_hits info
  _zone_hits_fields = [
    'emtf_site',
    'emtf_host',
    'emtf_chamber',
    'emtf_segment',
    'zones',
    'timezones',
    'emtf_phi',
    'emtf_bend',
    'emtf_theta',
    'emtf_theta_alt',
    'emtf_qual',
    'emtf_qual_alt',
    'emtf_time',
    'strip',
    'wire',
    'fr',
    'detlayer',
    'bx',
  ]

  ZoneHitsFields = collections.namedtuple('ZoneHitsFields', _zone_hits_fields)
  zone_hits_fields = ZoneHitsFields(*range(len(_zone_hits_fields)))
  config['zone_hits_fields'] = zone_hits_fields

  # packed_hits info
  _packed_hits_fields = [
    'emtf_phi',
    'emtf_bend',
    'emtf_theta1',
    'emtf_theta2',
    'emtf_qual1',
    'emtf_qual2',
    'emtf_time',
    'zones',
    'tzones',
    'fr',
    'dl',
    'bx',
    'valid',
  ]

  PackedHitsFields = collections.namedtuple('PackedHitsFields', _packed_hits_fields)
  packed_hits_fields = PackedHitsFields(*range(len(_packed_hits_fields)))
  config['packed_hits_fields'] = packed_hits_fields

  # features info
  _features_fields = [
    'emtf_phi_begin',
    'emtf_phi_end',
    'emtf_theta_begin',
    'emtf_theta_end',
    'emtf_bend_begin',
    'emtf_bend_end',
    'emtf_qual_begin',
    'emtf_qual_end',
    'phi_median',
    'theta_median',
    'trk_qual',
    'trk_bx',
  ]

  _features_enums = [
    (num_emtf_sites * 0),
    (num_emtf_sites * 1),
    (num_emtf_sites * 1),
    (num_emtf_sites * 2),
    (num_emtf_sites * 4 // 2),
    (num_emtf_sites * 5 // 2),
    (num_emtf_sites * 5 // 2),
    (num_emtf_sites * 6 // 2),
    (num_emtf_sites * 3) + 0,
    (num_emtf_sites * 3) + 1,
    (num_emtf_sites * 3) + 2,
    (num_emtf_sites * 3) + 3,
  ]

  FeaturesFields = collections.namedtuple('FeaturesFields', _features_fields)
  features_fields = FeaturesFields(*_features_enums)
  config['features_fields'] = features_fields

  # Various mapping for use in Zoning and TrkBuilding
  def _to_array(x):
    found_ragged = len(set(len(x_i) for x_i in x)) > 1
    if found_ragged:
      return np.array([np.array(x_i) for x_i in x], dtype=np.object)
    else:
      return np.array([x_i for x_i in x])

  site_to_img_row_luts = np.array([
    [2, 2, 4, 5, 7, 2, 4, 6, 7, 1, 3, 0],
    [1, 2, 4, 5, 7, 2, 4, 6, 7, 0, 3, 0],
    [0, 0, 3, 4, 6, 1, 2, 5, 7, 0, 3, 0],
  ], dtype=np.int32)

  # flake8: noqa:E231
  site_rm_to_many_sites_lut = np.array([
    [  0,   9,   1,   5],  # ME1/1, GE1/1, ME1/2, RE1/2
    [  2,  10,   6, -99],  # ME2, GE2/1, RE2/2
    [  3,   7, -99, -99],  # ME3, RE3
    [  4,   8, -99, -99],  # ME4, RE4
    [ 11, -99, -99, -99],  # ME0
  ], dtype=np.int32)

  chamber_to_host_lut = np.array([
     0, 0, 0, 1, 1, 1, 2, 2, 2,  # ME1/1 sub 1, ME1/2 sub 1, ME1/3 sub 1
     0, 0, 0, 1, 1, 1, 2, 2, 2,  # ME1/1 sub 2, ME1/2 sub 2, ME1/3 sub 2
     3, 3, 3, 4, 4, 4, 4, 4, 4,  # ME2/1, ME2/2
     5, 5, 5, 6, 6, 6, 6, 6, 6,  # ME3/1, ME3/2
     7, 7, 7, 8, 8, 8, 8, 8, 8,  # ME4/1, ME4/2
     0, 1, 2, 3, 4, 5, 6, 7, 8,  # neigh
    #
     9, 9, 9,10,10,10,11,11,11,  # GE1/1 sub 1, RE1/2 sub 1, RE1/3 sub 1
     9, 9, 9,10,10,10,11,11,11,  # GE1/1 sub 2, RE1/2 sub 2, RE1/3 sub 2
    12,12,12,13,13,13,13,13,13,  # GE2/1, RE2/2
    14,14,14,15,15,15,15,15,15,  # RE3/1, RE3/2
    16,16,16,17,17,17,17,17,17,  # RE4/1, RE4/2
     9,10,11,12,13,14,15,16,17,  # neigh
    #
    18,18,18,18,18,18,18,        # ME0
  ], dtype=np.int32)

  # Ignore ME1/3 and RE1/3
  ignored_chambers = np.array([
     6,  7,  8, 15, 16, 17, 47,  # ME1/3
    60, 61, 62, 69, 70, 71,101,  # RE1/3
  ], dtype=np.int32)

  host_to_site_lut = np.array([
    0, 1, 1, 2, 2, 3, 3, 4, 4, 9, 5, 5, 10, 6, 7, 7, 8, 8, 11
  ], dtype=np.int32)

  # Obtained from find_emtf_img_row_lut()
  host_to_img_row_luts = np.array([
    [  2,   1, -99],
    [-99,   2,   0],
    [-99, -99, -99],
    [  4,   4, -99],
    [-99, -99,   3],
    [  5,   5, -99],
    [-99,   5,   4],
    [  7,   7, -99],
    [-99,   7,   6],
    [  1,   0, -99],
    [-99,   2,   1],
    [-99, -99, -99],
    [  3,   3, -99],
    [-99, -99,   2],
    [  6,   6, -99],
    [-99,   6,   5],
    [  7,   7, -99],
    [-99,   7,   7],
    [  0, -99, -99],
  ], dtype=np.int32)

  config['site_to_img_row_luts'] = site_to_img_row_luts

  host_to_chamber_lut = gather_indices_by_values(chamber_to_host_lut)
  site_to_host_lut = gather_indices_by_values(host_to_site_lut)
  site_to_chamber_lut = _to_array([
      [c for host in hosts for c in host_to_chamber_lut[host]]
      for hosts in site_to_host_lut
  ])
  site_to_chamber_lut = _to_array([
      [c for c in chambers if c not in ignored_chambers]
      for (i, chambers) in enumerate(site_to_chamber_lut)
  ])
  site_number_lut = _to_array([
      np.repeat(i, len(chambers))
      for (i, chambers) in enumerate(site_to_chamber_lut)
  ])

  config['site_to_chamber_lut'] = site_to_chamber_lut
  config['site_number_lut'] = site_number_lut

  img_row_to_chamber_luts = np.empty(num_emtf_zones, dtype=np.object)
  img_row_number_luts = np.empty(num_emtf_zones, dtype=np.object)

  # Loop over zones
  for z in range(num_emtf_zones):
    host_to_img_row_lut = host_to_img_row_luts[:, z]
    img_row_to_host_lut = gather_indices_by_values(host_to_img_row_lut)
    img_row_to_chamber_lut = _to_array([
        [c for host in hosts for c in host_to_chamber_lut[host]]
        for hosts in img_row_to_host_lut
    ])
    img_row_to_chamber_lut = _to_array([
        [c for c in chambers if c not in ignored_chambers]
        for (i, chambers) in enumerate(img_row_to_chamber_lut)
    ])
    img_row_number_lut = _to_array([
        np.repeat(i, len(chambers))
        for (i, chambers) in enumerate(img_row_to_chamber_lut)
    ])
    img_row_to_chamber_luts[z] = img_row_to_chamber_lut
    img_row_number_luts[z] = img_row_number_lut

  config['img_row_to_chamber_luts'] = img_row_to_chamber_luts
  config['img_row_number_luts'] = img_row_number_luts

  # Various theta indices for use in TrkBuilding
  # th1_ME2, th1_ME3, th1_ME4, th2_ME2, th2_ME3, th2_ME4, th1_RE2, th1_RE3, th1_RE4
  trk_theta_indices = np.array([
    (2, 0), (3, 0), (4, 0), (2, 1), (3, 1), (4, 1), (6, 0), (7, 0), (8, 0),
  ], dtype=np.int32)

  # th1_ME2, th1_ME3, th1_ME4, th2_ME2, th2_ME3, th2_ME4, th1_GE2, th1_RE3, th1_RE4
  trk_theta_indices_alt = np.array([
    (2, 0), (3, 0), (4, 0), (2, 1), (3, 1), (4, 1), (10, 0), (7, 0), (8, 0),
  ], dtype=np.int32)

  # th1_ME12, th1_ME11, th2_ME0, th2_ME12, th2_ME11, th2_ME0, th1_RE12, th1_GE11, th1_ME0
  trk_theta_indices_me1 = np.array([
    (1, 0), (0, 0), (11, 1), (1, 1), (0, 1), (11, 1), (5, 0), (9, 0), (11, 0),
  ], dtype=np.int32)

  # ME11, ME12, ME2, ME3, ME4, ME0
  trk_bendable_indices = np.array([0, 1, 2, 3, 4, 11,], dtype=np.int32)

  config['trk_theta_indices'] = trk_theta_indices
  config['trk_theta_indices_alt'] = trk_theta_indices_alt
  config['trk_theta_indices_me1'] = trk_theta_indices_me1
  config['trk_bendable_indices'] = trk_bendable_indices

  # Mapping site to site_rm in DupeRemoval
  site_to_site_rm_lut = gather_inputs_by_outputs(site_rm_to_many_sites_lut)
  assert site_to_site_rm_lut.shape[0] == num_emtf_sites
  assert site_to_site_rm_lut.shape[1] == num_emtf_sites_rm

  config['site_rm_to_many_sites_lut'] = site_rm_to_many_sites_lut
  config['site_to_site_rm_lut'] = site_to_site_rm_lut

  # Mapping site to site_enum in TrainFilter
  site_to_site_enum_lut = np.array([
    11, 12, 22, 23, 24, 12, 22, 23, 24, 11, 22, 14
  ], dtype=np.int32)
  assert (num_emtf_features - num_emtf_features_addl) == (num_emtf_sites * 3)
  assert len(site_to_site_enum_lut) == num_emtf_sites

  config['site_to_site_enum_lut'] = site_to_site_enum_lut
  return config
