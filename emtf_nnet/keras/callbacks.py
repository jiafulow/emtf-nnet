# The following source code was originally obtained from:
# https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/callbacks.py#L1858-L1920
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule
from tensorflow.python.keras.callbacks import Callback


class LearningRateLogger(Callback):
  """Learning rate logger."""

  def __init__(self):
    super(LearningRateLogger, self).__init__()

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    lr_schedule = getattr(self.model.optimizer, 'lr', None)
    if isinstance(lr_schedule, LearningRateSchedule):
      lr = lr_schedule(self.model.optimizer.iterations)
      lr = K.get_value(lr)
    else:
      lr = K.get_value(self.model.optimizer.lr)
    logs['lr'] = lr

    if getattr(self.model.optimizer, 'gradient_maxnorm', None):
      gradient_maxnorm = K.get_value(self.model.optimizer.gradient_maxnorm)
      logs['gradient_maxnorm'] = gradient_maxnorm
