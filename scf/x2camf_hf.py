from functools import reduce

import os
import numpy
import scipy

from pyscf import gto, lib, scf
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf.lib import chkfile, logger
from pyscf.x2c import x2c
from pyscf.scf import hf, dhf, ghf

from socutils.somf.x2camf import SpinorX2CAMFHelper, SpinOrbitalX2CAMFHelper
from socutils.scf import frac_dhf, spinor_hf
#, zquatev

try:
    import zquatev
except ImportError:
    pass

class SCF(spinor_hf.JHF):
    nopen = None
    nact = None

    def __init__(self, mol, nopen=0, nact=0, with_gaunt=True, with_breit=True, with_gaunt_sd=False, with_aoc=False, with_pcc=False, prog="sph_atm"):
        super().__init__(mol)
        print(with_pcc)
        self.with_x2c = SpinorX2CAMFHelper(mol,
                                           with_gaunt=with_gaunt,
                                           with_breit=with_breit,
                                           with_gaunt_sd=with_gaunt_sd,
                                           with_aoc=with_aoc,
                                           with_pcc=with_pcc,
                                           prog=prog)
        self._keys = self._keys.union(['with_x2c'])
        self.nopen = nopen
        self.nact = nact

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        nopen = self.nopen
        nact = self.nact
        nclose = mol.nelectron - nact
        n2c = len(mo_energy)
        mo_occ = numpy.zeros(n2c)

        if nopen == 0:
            mo_occ[:mol.nelectron] = 1
        else:
            mo_occ[:nclose] = 1
            mo_occ[nclose:nclose + nopen] = 1. * nact / nopen

        if self.verbose >= logger.INFO:
            if nopen == 0:
                homo_ndx = mol.nelectron
            else:
                homo_ndx = nclose + nopen
            logger.info(self, 'HOMO %d = %.12g  LUMO %d = %.12g', homo_ndx, mo_energy[homo_ndx - 1], homo_ndx + 1,
                        mo_energy[homo_ndx])
            logger.debug(self, 'mo_energy = %s', mo_energy[:])
        return mo_occ


X2CAMF_SCF = SCF


class UHF(SCF):

    def to_ks(self, xc='HF'):
        from pyscf.x2c import dft
        mf = self.view(dft.UKS)
        mf.converged = False
        return mf


X2CAMF_UHF = UHF


class RHF(UHF):

    def __init__(self, mol, nopen=0, nact=0, with_gaunt=True, with_breit=True, with_aoc=False, prog="sph_atm"):
        super().__init__(mol, nopen, nact, with_gaunt, with_breit, with_aoc, prog)
        if dhf.zquatev is None:
            raise RuntimeError('zquatev library is required to perform Kramers-restricted X2C-RHF')

    def _eigh(self, h, s):
        return dhf.zquatev.solve_KR_FCSCE(self.mol, h, s)

    def to_ks(self, xc='HF'):
        from pyscf.x2c import dft
        mf = self.view(dft.RKS)
        mf.converged = False
        return mf


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
