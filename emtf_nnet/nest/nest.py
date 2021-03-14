# The following source code was originally obtained from:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/nest.py
# ==============================================================================

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Functions for working with arbitrarily nested sequences of elements."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six


def _is_sequence(seq):
  """Returns true if its input is a `tuple`, `list`, or `range`.

  Args:
    seq: an input sequence.

  Returns:
    True if the sequence is a `tuple`, `list`, or `range`.
  """
  return isinstance(seq, (tuple, list, six.moves.range))


def _sequence_like(instance, args):
  """Converts the sequence `args` to the same type as `instance`.

  Args:
    instance: an instance of `tuple`, `list`, or `range`.
    args: elements to be converted to the `instance` type.

  Returns:
    `args` with the type of `instance`.
  """
  if isinstance(instance, six.moves.range):
    return _sequence_like(list(instance), args)
  else:
    return type(instance)(args)


def _packed_nest_with_indices(structure, flat, index, is_seq, sequence_fn=None):
  """Helper function for pack_sequence_as.

  Args:
    structure: Substructure (list / tuple / dict) to mimic.
    flat: Flattened values to output substructure for.
    index: Index at which to start reading from flat.
    is_seq: Function used to test if a value should be treated as a sequence.
    sequence_fn: Function used to generate a new sequence instance.

  Returns:
    The tuple (new_index, child), where:
      * new_index - the updated index into `flat` having processed `structure`.
      * packed - the subset of `flat` corresponding to `structure`,
                 having started at `index`, and packed into the same nested
                 format.

  Raises:
    ValueError: if `structure` contains more elements than `flat`
      (assuming indexing starts from `index`).
  """
  packed = []
  sequence_fn = sequence_fn or _sequence_like
  for s in structure:
    if is_seq(s):
      new_index, child = _packed_nest_with_indices(s, flat, index, is_seq,
                                                   sequence_fn)
      packed.append(sequence_fn(s, child))
      index = new_index
    else:
      packed.append(flat[index])
      index += 1
  return index, packed


def _pack_sequence_as(structure, flat_sequence, expand_composites,
                      is_seq, sequence_fn=None):
  """Implements sequence packing, with the option to alter the structure."""
  #is_seq = is_sequence_or_composite if expand_composites else is_sequence
  sequence_fn = sequence_fn or _sequence_like

  def truncate(value, length):
    value_str = str(value)
    return value_str[:length] + (value_str[length:] and "...")

  if not is_seq(flat_sequence):
    raise TypeError(
        "Attempted to pack value:\n  {}\ninto a sequence, but found "
        "incompatible type `{}` instead."
        .format(truncate(flat_sequence, 100), type(flat_sequence)))

  if not is_seq(structure):
    if len(flat_sequence) != 1:
      raise ValueError(
          "The target structure is of type `{}`\n  {}\nHowever the input "
          "structure is a sequence ({}) of length {}.\n  {}\nnest cannot "
          "guarantee that it is safe to map one to the other.".format(
              type(structure), truncate(structure, 100), type(flat_sequence),
              len(flat_sequence), truncate(flat_sequence, 100)))
    return flat_sequence[0]

  try:
    final_index, packed = _packed_nest_with_indices(structure, flat_sequence,
                                                    0, is_seq, sequence_fn)
    if final_index < len(flat_sequence):
      raise IndexError
  except IndexError:
    flat_structure = flatten(structure, expand_composites=expand_composites)
    if len(flat_structure) != len(flat_sequence):
      raise ValueError(
          "Could not pack sequence. Structure had %d elements, but "
          "flat_sequence had %d elements.  Structure: %s, flat_sequence: %s." %
          (len(flat_structure), len(flat_sequence), structure, flat_sequence))
  return sequence_fn(structure, packed)


def pack_sequence_as(structure, flat_sequence, expand_composites=False):
  """Returns a given flattened sequence packed into a given structure.

  If `structure` is a scalar, `flat_sequence` must be a single-element list;
  in this case the return value is `flat_sequence[0]`.

  Examples:

  1. For a nested python tuple:
    >>> structure = (('a','b'), ('c','d','e'), 'f')
    >>> flat_sequence = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> pack_sequence_as(structure, flat_sequence)
    ((1.0, 2.0), (3.0, 4.0, 5.0), 6.0)

  2. Numpy array (considered a scalar):
    >>> structure = ['a']
    >>> flat_sequence = [np.array([[1, 2], [3, 4]])]
    >>> pack_sequence_as(structure, flat_sequence)
    [array([[1, 2],
            [3, 4]])]

  Args:
    structure: Nested structure, whose structure is given by nested lists,
      tuples, and dicts. Note: numpy arrays and strings are considered
      scalars.
    flat_sequence: flat sequence to pack.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    packed: `flat_sequence` converted to have the same recursive structure as
      `structure`.

  Raises:
    ValueError: If `flat_sequence` and `structure` have different
      element counts.
    TypeError: `structure` is or contains a dict with non-sortable keys.
  """
  return _pack_sequence_as(structure, flat_sequence, expand_composites,
                           is_seq=_is_sequence, sequence_fn=_sequence_like)


def _yield_value_flat(sequence, is_seq):
  """Yields values from a sequence recursively."""
  if is_seq(sequence):
    for item in sequence:
      yield from _yield_value_flat(item, is_seq)
  else:
    yield sequence


def flatten(structure, expand_composites=False):
  """Returns a flat list from a given nested structure.

  If nest is not a structure, then returns a single-element list:
    [nest].

  Examples:

  1. Nested python tuple:

    >>> tup = ((1.0, 2.0), (3.0, 4.0, (5.0, (6.0))))
    >>> flatten(tup)
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

  2. Numpy array (will not flatten):

    >>> arr = np.array([[1, 2], [3, 4]])
    >>> flatten(arr)
    [array([[1, 2],
            [3, 4]])]

  Args:
    structure: an arbitrarily nested structure. Note, numpy arrays are
      considered atoms and are not flattened.
    expand_composites: If true, then composite tensors such as
      `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
      component tensors.

  Returns:
    A Python list, the flattened version of the input.

  Raises:
    TypeError: The nest is or contains a dict with non-sortable keys.
  """
  if _is_sequence(structure):
    return list(_yield_value_flat(structure, is_seq=_is_sequence))
  else:
    return [structure]


def map_structure(func, *structure, **kwargs):
  """Applies `func` to each entry in `structure` and returns a new structure.

  Applies `func(x[0], x[1], ...)` where x[i] is an entry in
  `structure[i]`.  All structures in `structure` must have the same arity,
  and the return value will contain results with the same structure layout.

  Examples:

  * A single Python dict:

  >>> a = {"hello": 24, "world": 76}
  >>> tf.nest.map_structure(lambda p: p * 2, a)
  {'hello': 48, 'world': 152}

  * Multiple Python dictionaries:

  >>> d1 = {"hello": 24, "world": 76}
  >>> d2 = {"hello": 36, "world": 14}
  >>> tf.nest.map_structure(lambda p1, p2: p1 + p2, d1, d2)
  {'hello': 60, 'world': 90}

  * A single Python list:

  >>> a = [24, 76, "ab"]
  >>> tf.nest.map_structure(lambda p: p * 2, a)
  [48, 152, 'abab']

  * Scalars:

  >>> tf.nest.map_structure(lambda x, y: x + y, 3, 4)
  7

  * Empty structures:

  >>> tf.nest.map_structure(lambda x: x + 1, ())
  ()

  *. Check the types of iterables:

  >>> s1 = (((1, 2), 3), 4, (5, 6))
  >>> s1_list = [[[1, 2], 3], 4, [5, 6]]
  >>> tf.nest.map_structure(lambda x, y: None, s1, s1_list)
  Traceback (most recent call last):
  ...
  TypeError: The two structures don't have the same nested structure

  * Type check is set to False:

  >>> s1 = (((1, 2), 3), 4, (5, 6))
  >>> s1_list = [[[1, 2], 3], 4, [5, 6]]
  >>> tf.nest.map_structure(lambda x, y: None, s1, s1_list, check_types=False)
  (((None, None), None), None, (None, None))

  Args:
    func: A callable that accepts as many arguments as there are structures.
    *structure: scalar, or tuple or dict or list of constructed scalars and/or
      other tuples/lists, or scalars.  Note: numpy arrays are considered as
      scalars.
    **kwargs: Valid keyword args are:

      * `check_types`: If set to `True` (default) the types of
        iterables within the structures have to be same (e.g.
        `map_structure(func, [1], (1,))` raises a `TypeError`
        exception). To allow this set this argument to `False`.
        Note that namedtuples with identical name and fields are always
        considered to have the same shallow structure.
      * `expand_composites`: If set to `True`, then composite tensors such
        as `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into
        their component tensors.  If `False` (the default), then composite
        tensors are not expanded.

  Returns:
    A new structure with the same arity as `structure`, whose values correspond
    to `func(x[0], x[1], ...)` where `x[i]` is a value in the corresponding
    location in `structure[i]`. If there are different sequence types and
    `check_types` is `False` the sequence types of the first structure will be
    used.

  Raises:
    TypeError: If `func` is not callable or if the structures do not match
      each other by depth tree.
    ValueError: If no structure is provided or if the structures do not match
      each other by type.
    ValueError: If wrong keyword arguments are provided.
  """
  if not callable(func):
    raise TypeError("func must be callable, got: %s" % func)

  if not structure:
    raise ValueError("Must provide at least one structure")

  # flake8: noqa:F841
  check_types = kwargs.pop("check_types", True)
  expand_composites = kwargs.pop("expand_composites", False)

  if kwargs:
    raise ValueError(
        "Only valid keyword arguments are `check_types` and "
        "`expand_composites`, not: `%s`" % ("`, `".join(kwargs.keys())))

  #for other in structure[1:]:
  #  assert_same_structure(structure[0], other, check_types=check_types,
  #                        expand_composites=expand_composites)

  flat_structure = (flatten(s, expand_composites) for s in structure)
  entries = zip(*flat_structure)

  return pack_sequence_as(
      structure[0], [func(*x) for x in entries],
      expand_composites=expand_composites)
