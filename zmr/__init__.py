'''zmr: relativistic multireference perturbation theory and CI.

Core modules (active/, uc/, integrals, fock, laplace) depend on numpy/scipy
only.  PySCF and socutils are imported exclusively by zmr.interface (integral
construction), zmr.reference (brute-force reference implementation) and the
tests.
'''

__version__ = '0.0.1'
