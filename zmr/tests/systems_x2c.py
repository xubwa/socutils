'''X2C spinor test systems (socutils).'''
import functools

from pyscf import gto
from socutils.scf import spinor_hf

O2_GEOM = 'O 0 0 -0.6035; O 0 0 0.6035'


@functools.lru_cache(maxsize=None)
def o2_dianion_orbitals(basis='cc-pvdz', soc=True):
    '''Closed-shell O2^2- spinor orbitals (pi_g* fully occupied, so the four
    pi_g* spinors are symmetry-equivalent), used for a CAS(2e, 4 spinors)
    treatment of neutral O2.'''
    mol = gto.M(atom=O2_GEOM, basis=basis, charge=-2, verbose=0)
    mf = spinor_hf.SCF(mol)
    if soc:
        mf = mf.x2camf()
    else:
        mf = mf.x2camf(flavor='sf1e', with_gaunt=False, with_breit=False)
    mf.verbose = 0
    mf.conv_tol = 1e-11
    mf.kernel()
    neutral = gto.M(atom=O2_GEOM, basis=basis, charge=0, spin=2, verbose=0)
    return mf, neutral
