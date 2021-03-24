from .lazy_loader import LazyLoader
from .version import __version__

# Delay the import
architecture = LazyLoader('architecture', globals(), 'emtf_nnet.architecture')
keras = LazyLoader('keras', globals(), 'emtf_nnet.keras')
nest = LazyLoader('nest', globals(), 'emtf_nnet.nest')
ragged = LazyLoader('ragged', globals(), 'emtf_nnet.ragged')
slices = LazyLoader('slices', globals(), 'emtf_nnet.slices')
sparse = LazyLoader('sparse', globals(), 'emtf_nnet.sparse')
