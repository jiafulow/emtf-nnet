"""Testing nest functions."""

import numpy as np

from emtf_nnet.nest import flatten, pack_sequence_as


def test_me():
  structure = ((3, 4), 5, (6, 7, (9, 10), 8))
  flat = ["a", "b", "c", "d", "e", "f", "g", "h"]
  assert flatten(structure) == [3, 4, 5, 6, 7, 9, 10, 8]
  assert pack_sequence_as(structure, flat) == (("a", "b"), "c",
                                               ("d", "e", ("f", "g"), "h"))
