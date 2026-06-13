from functools import reduce

import os
import warnings
import numpy
import scipy

from pyscf import gto, lib, scf
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf.lib import chkfile, logger
from pyscf.x2c import x2c
from pyscf.scf import hf, dhf, ghf

from socutils.somf.amf import SpinOrbitalX2CAMFHelper
from socutils.scf import frac_dhf, spinor_hf
#, zquatev

try:
    import zquatev
except ImportError:
    pass

# ----------------------------------------------------------------------------
# Deprecated spinor X2CAMF drivers.
#
# spinor_hf is now the single spinor HF driver, and X2CAMF attaches through
# ``with_x2c`` -- mirroring how pyscf's x2c works.  The classes below predate
# that abstraction (they were the original spinor X2CAMF SCF, hence the name
# x2camf_hf) and are kept only as thin, warning-emitting shims that forward to
# spinor_hf + ``.x2camf()``.  ``.x2camf()`` defaults to the ``x2camf`` flavor of
# SpinorX2CMPHelper, which is numerically identical to the old
# SpinorX2CAMFHelper, so existing results are reproduced.
#
# New code should use:
#     spinor_hf.SCF(mol).x2camf(...)    # Kramers-unrestricted
#     spinor_hf.KRHF(mol).x2camf(...)   # Kramers-restricted (needs zquatev)
#
# The fractional-occupation (nopen/nact) helper that used to live here is gone;
# set ``mf.get_occ`` to a custom callable if you need it.
# ----------------------------------------------------------------------------

_DEPRECATION_MSG = (
    'socutils.scf.x2camf_hf is deprecated. spinor_hf is the single spinor HF '
    'driver and X2CAMF attaches through with_x2c. Use '
    'spinor_hf.SCF(mol).x2camf(...) (Kramers-unrestricted) or '
    'spinor_hf.KRHF(mol).x2camf(...) (Kramers-restricted) instead.'
)


class SCF(spinor_hf.SpinorSCF):
    '''Deprecated. Use ``spinor_hf.SCF(mol).x2camf(...)``.'''

    def __init__(self, mol, with_gaunt=True, with_breit=True, with_aoc=False,
                 with_pcc=False, **kwargs):
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        super().__init__(mol)
        self.x2camf(with_gaunt=with_gaunt, with_breit=with_breit,
                    with_pcc=with_pcc, with_aoc=with_aoc)

    def to_ks(self, xc='HF'):
        from pyscf.x2c import dft
        mf = self.view(dft.UKS)
        mf.converged = False
        return mf


X2CAMF_SCF = SCF
UHF = SCF
X2CAMF_UHF = UHF


class RHF(spinor_hf.KRHF):
    '''Deprecated. Use ``spinor_hf.KRHF(mol).x2camf(...)`` (needs zquatev).'''

    def __init__(self, mol, with_gaunt=True, with_breit=True, with_aoc=False,
                 with_pcc=False, **kwargs):
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        super().__init__(mol)
        self.x2camf(with_gaunt=with_gaunt, with_breit=with_breit,
                    with_pcc=with_pcc, with_aoc=with_aoc)


X2CAMF_RHF = RHF


def x2camf_ghf(mf, with_gaunt=True, with_breit=True, with_aoc=False, prog="sph_atm"):
    '''
    For the given *GHF* object, generate X2C-GSCF object in GHF spin-orbital
    basis. Note the orbital basis of X2C_GSCF is different to the X2C_RHF and
    X2C_UHF objects. X2C_RHF and X2C_UHF use j-adapated spinor basis.

    Args:
        mf : an GHF/GKS object

    Returns:
        An GHF/GKS object

    Examples:

    >>> mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.GHF(mol).x2c1e().run()
    '''
    assert isinstance(mf, ghf.GHF)

    if isinstance(mf, x2c._X2C_SCF):
        if mf.with_x2c is None:
            mf.with_x2c = SpinOrbitalX2CAMFHelper(mf.mol, with_gaunt=with_gaunt,
                with_breit=with_breit, with_aoc=with_aoc, prog=prog)
            return mf
        elif not isinstance(mf.with_x2c, SpinOrbitalX2CAMFHelper):
            # An object associated to sfx2c1e.SpinFreeX2CHelper
            raise NotImplementedError
        else:
            return mf

    mf_class = mf.__class__
    if mf_class.__doc__ is None:
        doc = ''
    else:
        doc = mf_class.__doc__

    class X2CAMF_GSCF(x2c._X2C_SCF, mf_class):
        __doc__ = doc + '''
        Attributes for spin-orbital X2C with AMF correction:
            with_x2c : X2C object
        '''

        def __init__(self, mol, *args, **kwargs):
            mf_class.__init__(self, mol, *args, **kwargs)
            self.with_x2c = SpinOrbitalX2CAMFHelper(mf.mol, with_gaunt=with_gaunt,
                with_breit=with_breit, with_aoc=with_aoc, prog=prog)
            self._keys = self._keys.union(['with_x2c'])

        def get_hcore(self, mol=None):
            if mol is None: mol = self.mol
            hcore = self.with_x2c.get_hcore(mol)
            return hcore

        def dump_flags(self, verbose=None):
            mf_class.dump_flags(self, verbose)
            if self.with_x2c:
                self.with_x2c.dump_flags(verbose)
            return self

        def reset(self, mol):
            self.with_x2c.reset(mol)
            return mf_class.reset(self, mol)

    mf.with_x2c = SpinOrbitalX2CAMFHelper(mf.mol, with_gaunt=with_gaunt,
        with_breit=with_breit, with_aoc=with_aoc, prog=prog)
    return mf.view(X2CAMF_GSCF)


if __name__ == '__main__':
    mol = gto.M(verbose=3,
                atom=[["O", (0.,          0., -0.12390941)],
                      ['O',   (0., -1.42993701,  0.98326612)],
                      ['O',   (0.,  1.42993701,  0.98326612)]],
                basis='ccpvtzdk',
                unit='Bohr')
    #mf = X2CAMF_RHF(mol, with_gaunt=False, with_breit=False)
    #e_spinor = mf.scf()
    #os.system('rm amf.chk')
    #mf = X2CAMF_RHF(mol, with_gaunt=True, with_breit=False)
    #e_gaunt = mf.scf()
    #os.system('rm amf.chk')
    mf = X2CAMF_RHF(mol, with_gaunt=True, with_breit=True)
    e_breit = mf.scf()
    gmf = x2camf_ghf(scf.GHF(mol), with_gaunt=True, with_breit=True)
    e_ghf = gmf.kernel()
    #print("Energy from spinor X2CAMF(Coulomb):    %16.10g" % e_spinor)
    #print("Energy from spinor X2CAMF(Gaunt):      %16.10g" % e_gaunt)
    print("Energy from spinor X2CAMF(Breit):      %16.10g" % e_breit)
    print("Energy from ghf-based X2CAMF(Breit):   %16.10g" % e_ghf)
