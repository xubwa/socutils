__version__ = '0.1'

# Backend selection (pure-C 'c' by default, pybind11 C++ reference 'cpp').
from x2camf import backend

# Low-level dispatcher (amfi / atm_integrals / pcc_K). Routes to the
# selected backend; importing it does not load any library yet.
from x2camf import libx2camf

# High-level molecular interface. Requires pyscf.
try:
    from x2camf import x2camf
    amfi = x2camf.amfi
except ImportError:
    pass
