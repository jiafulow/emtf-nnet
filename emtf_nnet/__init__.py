from .lazy_loader import LazyLoader
from .version import __version__

# Delay the import
keras = LazyLoader('keras', globals(), 'emtf_nnet.keras')
ragged = LazyLoader('ragged', globals(), 'emtf_nnet.ragged')
sparse = LazyLoader('sparse', globals(), 'emtf_nnet.sparse')
