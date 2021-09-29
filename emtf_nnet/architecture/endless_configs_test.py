"""Testing"""

import pytest

from .endless_configs import configure_v3


def test_me():
  config = configure_v3(strict=False)
  assert config

  with pytest.raises(ValueError):
    config = configure_v3()
