# The following source code was originally obtained from:
# https://github.com/keras-team/keras/blob/r2.6/keras/callbacks.py#L1899-L1962
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
"""Callbacks: utilities called at certain points during model training."""

from keras import backend
from keras.optimizer_v2 import learning_rate_schedule
from keras.callbacks import Callback


class LearningRateLogger(Callback):
  """Learning rate logger."""

  def __init__(self):
    super().__init__()

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    lr_schedule = getattr(self.model.optimizer, 'lr', None)
    if isinstance(lr_schedule, learning_rate_schedule.LearningRateSchedule):
      logs['lr'] = backend.get_value(lr_schedule(self.model.optimizer.iterations))
    else:
      logs['lr'] = backend.get_value(self.model.optimizer.lr)

    if hasattr(self.model.optimizer, 'gradient_maxnorm'):
      gradient_maxnorm = backend.get_value(self.model.optimizer.gradient_maxnorm)
      logs['gradient_maxnorm'] = gradient_maxnorm
