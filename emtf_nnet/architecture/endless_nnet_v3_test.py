"""Testing."""

import numpy as np

from .endless_nnet_v3 import (get_x_y_data,
                              create_preprocessing_layer,
                              create_regularization_layer,
                              create_lr_schedule,
                              create_optimizer,
                              create_simple_model,
                              create_model,
                              create_quant_model,
                              create_sparsity_m_by_n_list,
                              create_pruning_schedule,
                              create_pruned_model)


def test_me():
  features = np.random.randint(2, size=(10000, 40), dtype=np.int32)
  truths = np.random.random_sample(size=(10000, 9)).astype(np.float32)
  noises = np.random.randint(2, size=(10000, 40), dtype=np.int32)
  x_train, x_test, y_train, y_test = get_x_y_data(features, truths)
  assert (x_train.shape[0] == y_train.shape[0]) and (x_test.shape[0] == y_test.shape[0])
  assert (x_train.shape[1] == x_test.shape[1]) and (y_train.shape[1] == y_test.shape[1])

  model = create_simple_model()
  assert model

  preprocessing_layer = create_preprocessing_layer(x_train)
  regularization_layer = create_regularization_layer(noises)
  lr_schedule = create_lr_schedule(x_train.shape[0])
  optimizer = create_optimizer(lr_schedule)
  model = create_model(preprocessing_layer, regularization_layer, optimizer)
  assert model

  base_model = model
  pruning_schedule = create_pruning_schedule(x_train.shape[0])
  sparsity_m_by_n_list = create_sparsity_m_by_n_list(2, 4)
  model = create_pruned_model(base_model, optimizer,
                              layers_to_prune={'dense'},
                              layers_to_preserve={'dense', 'dense_1', 'dense_2'},
                              pruning_schedule=pruning_schedule,
                              sparsity_m_by_n=sparsity_m_by_n_list[-1])
  assert model

  model = create_quant_model(base_model, optimizer)
  assert model
