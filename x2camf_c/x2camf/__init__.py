__version__ = '0.1'

# Low-level interface (ctypes binding to libx2camf_c). Only requires numpy.
from x2camf import libx2camf

# High-level molecular interface. Requires pyscf.
try:
    from x2camf import x2camf
    amfi = x2camf.amfi
except ImportError:
    pass
