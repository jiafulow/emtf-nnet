# The following source code was originally obtained from:
# https://github.com/keras-team/keras/blob/r2.6/keras/layers/preprocessing/integer_lookup.py
# https://github.com/keras-team/keras/blob/r2.6/keras/layers/preprocessing/index_lookup.py
# ==============================================================================

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Integer lookup preprocessing layer."""

import tensorflow.compat.v2 as tf

import numpy as np

from keras.engine.base_layer import Layer


def listify_tensors(x):
  """Convert any tensors or numpy arrays to lists for config serialization."""
  if tf.is_tensor(x):
    x = x.numpy()
  if isinstance(x, np.ndarray):
    x = x.tolist()
  return x


class InverseIntegerLookup(Layer):
  """Maps integer indices to integer vocabulary items."""

  def __init__(self,
               vocabulary,
               max_tokens=None,
               num_oov_indices=0,
               mask_token=None,
               oov_token=-1,
               invert=True,
               output_mode="int",
               sparse=False,
               pad_to_max_tokens=False,
               **kwargs):
    allowed_dtypes = [tf.int32]

    if "dtype" in kwargs and kwargs["dtype"] not in allowed_dtypes:
      raise ValueError("The value of the dtype argument for IntegerLookup may "
                       "only be one of %s." % (allowed_dtypes,))

    if "dtype" not in kwargs:
      kwargs["dtype"] = tf.int32

    # If max_tokens is set, the token must be greater than 1 - otherwise we
    # are creating a 0-element vocab, which doesn't make sense.
    if max_tokens is not None and max_tokens <= 1:
      raise ValueError("If set, max_tokens must be greater than 1. "
                       "You passed %s" % (max_tokens,))

    if num_oov_indices < 0:
      raise ValueError(
          "num_oov_indices must be greater than or equal to 0. You passed %s" %
          (num_oov_indices,))

    if vocabulary is None:
      raise ValueError("Vocabulary must be provided.")

    # Make sure mask and oov are of the dtype we want.
    mask_token = None if mask_token is None else np.int32(mask_token)
    oov_token = None if oov_token is None else np.int32(oov_token)

    super().__init__(**kwargs)
    self.input_vocabulary = vocabulary
    self._has_input_vocabulary = True
    self.invert = invert  # unused
    self.max_tokens = max_tokens  # unused
    self.num_oov_indices = num_oov_indices  # unused
    self.mask_token = mask_token  # unused
    self.oov_token = oov_token
    self.output_mode = output_mode  # unused
    self.sparse = sparse  # unused
    self.pad_to_max_tokens = pad_to_max_tokens  # unused

    self._key_dtype = tf.as_dtype(self.dtype)
    self._value_dtype = tf.as_dtype(self.dtype)
    self._default_value = self.oov_token

  def build(self, input_shape):
    tokens = np.array(self.input_vocabulary)
    indices = np.arange(len(tokens))
    keys, values = (indices, tokens)
    initializer = tf.lookup.KeyValueTensorInitializer(keys, values,
                                                      self._key_dtype,
                                                      self._value_dtype)
    self._table = tf.lookup.StaticHashTable(initializer, self._default_value)
    self.built = True

  def call(self, inputs):
    inputs = tf.convert_to_tensor(inputs)
    if inputs.dtype != self._key_dtype:
      inputs = tf.cast(inputs, self._key_dtype)

    outputs = self._table.lookup(inputs)
    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def compute_output_signature(self, input_spec):
    output_shape = self.compute_output_shape(input_spec.shape.as_list())
    output_dtype = self._value_dtype
    return tf.TensorSpec(shape=output_shape, dtype=output_dtype)

  def get_config(self):
    config = super().get_config()
    config.update({
        "invert": self.invert,
        "max_tokens": self.max_tokens,
        "num_oov_indices": self.num_oov_indices,
        "oov_token": self.oov_token,
        "mask_token": self.mask_token,
        "output_mode": self.output_mode,
        "sparse": self.sparse,
        "pad_to_max_tokens": self.pad_to_max_tokens,
        "vocabulary": listify_tensors(self.input_vocabulary),
    })
    return config
