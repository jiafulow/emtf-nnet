from .lazy_loader import LazyLoader
from .version import __version__

# Delay the import
keras = LazyLoader('keras', globals(), 'emtf_nnet.keras')
