from ..lazy_loader import LazyLoader

# Delay the imports
callbacks = LazyLoader('callbacks', globals(), 'emtf_nnet.keras.callbacks')
layers = LazyLoader('layers', globals(), 'emtf_nnet.keras.layers')
losses = LazyLoader('losses', globals(), 'emtf_nnet.keras.losses')
optimizers = LazyLoader('optimizers', globals(), 'emtf_nnet.keras.optimizers')
quantization = LazyLoader('quantization', globals(), 'emtf_nnet.keras.quantization')
sparsity = LazyLoader('sparsity', globals(), 'emtf_nnet.keras.sparsity')
utils = LazyLoader('utils', globals(), 'emtf_nnet.keras.utils')
