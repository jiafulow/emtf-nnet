"""Testing."""

import numpy as np

from .endless_nnet_v3 import (get_x_y_data,
                              create_preprocessing_layer,
                              create_lr_schedule,
                              create_optimizer,
                              create_model,
                              create_pure_model,
                              create_quant_model)


def test_me():
  features = np.random.randint(2, size=(10000, 40), dtype=np.int32)
  truths = np.random.random(size=(10000, 9)).astype(np.float32)
  x_train, x_test, y_train, y_test = get_x_y_data(features, truths)
  assert (x_train.shape[0] == y_train.shape[0]) and (x_test.shape[0] == y_test.shape[0])
  assert (x_train.shape[1] == x_test.shape[1]) and (y_train.shape[1] == y_test.shape[1])

  model = create_pure_model()
  assert model

  preprocessing_layer = create_preprocessing_layer(x_train)
  lr_schedule = create_lr_schedule(x_train.shape[0])
  optimizer = create_optimizer(lr_schedule)
  model = create_model(preprocessing_layer=preprocessing_layer, optimizer=optimizer)
  assert model

  base_model = model
  model = create_quant_model(base_model, optimizer=optimizer)
  assert model
