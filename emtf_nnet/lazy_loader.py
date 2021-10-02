# The following source code was originally obtained from:
# https://github.com/keras-team/keras/blob/r2.6/keras/utils/generic_utils.py#L1170-L1191
# ==============================================================================

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Python utilities required by Keras."""

import importlib
import types


class LazyLoader(types.ModuleType):
  """Lazily import a module, mainly to avoid pulling in large dependencies."""

  def __init__(self, local_name, parent_module_globals, name):
    self._local_name = local_name
    self._parent_module_globals = parent_module_globals
    super().__init__(name=name)

  def _load(self):
    """Load the module and insert it into the parent's globals."""
    # Import the target module and insert it into the parent's namespace
    module = importlib.import_module(self.__name__)
    self._parent_module_globals[self._local_name] = module
    # Update this object's dict so that if someone keeps a reference to the
    #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
    #   that fail).
    self.__dict__.update(module.__dict__)
    return module

  def __getattr__(self, item):
    module = self._load()
    return getattr(module, item)
