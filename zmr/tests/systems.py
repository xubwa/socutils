'''Small molecular test systems shared by the test suite.'''
import functools
import numpy
from pyscf import gto, scf, mcscf


@functools.lru_cache(maxsize=None)
def h2o_rhf(basis='sto-3g'):
    mol = gto.M(atom='O 0 0 0.1; H 0 -0.757 -0.47; H 0 0.757 -0.47',
                basis=basis, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    return mf


def h2o_casci(ncas=4, nelecas=4, nroots=1, basis='sto-3g'):
    mf = h2o_rhf(basis)
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver.nroots = nroots
    mc.fcisolver.conv_tol = 1e-13
    mc.verbose = 0
    mc.kernel()
    return mc
