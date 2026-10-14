#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
# Gamma-point two-component (j-adapted spinor) SCF for a cell, with the
# Coulomb interactions from socutils.pbc.rsdf.RSDF.
#
# The molecular SpinorSCF machinery is reused unchanged: RSDF plays the role
# of the density-fitting object (scf.spinor_hf._DFJHF for J and the MO-based
# K, mcscf.zmc_ao2mo._CDERIS for the CASCI/CASSCF integral transformation),
# and the one-electron part is replaced by its periodic counterpart:
#     S, T      lattice sums at Gamma (cell.pbc_intor)
#     V_ne      RSDF.get_nuc(): Gaussian nuclei through the same kernel as J
#     E_nn      RSDF.energy_nuc()
# All Coulomb terms use one convention (G = 0 removed, tin-foil), so the
# total energy of the neutral cell is well defined.
#
# Exchange finite-size term (exxdiv): the two-electron integrals are those of
# the G = 0-free kernel.  The G = 0 part of (pq|rs) is v delta_pq delta_rs, so
# for ANY N-electron wavefunction it contributes v N(N-1)/2, and with the
# neutralising background the constant -nu_M N / 2 (nu_M: Madelung constant of
# the cell).  exxdiv='ewald' (default, as in pyscf) adds this constant to
# energy_nuc(); it equals pyscf's ewald exchange correction for HF at
# convergence and leaves orbitals, CI vectors and correlation energies
# unchanged, so CASCI/CASSCF on top see the same constant.  exxdiv=None gives
# the bare G = 0-free energy (error ~ 1/L for a molecule in a box).
#
# Usage:
#     r = RSDF(cell, auxbasis).build()
#     mf = spinor_scf.SpinorRSDF(cell, r).run()
#     mc = zcasci.CASCI(mf, ncas, nelecas).run()
#

import numpy
from pyscf import lib
from socutils.scf import spinor_hf


def _sph_blockdiag(h):
    nao = h.shape[-1]
    out = numpy.zeros((2 * nao, 2 * nao), dtype=h.dtype)
    out[:nao, :nao] = out[nao:, nao:] = h
    return out


class _PeriodicSpinorSCF(spinor_hf._DFJHF):
    '''Mixin over a molecular spinor SCF class: periodic one-electron integrals
    and RSDF Coulomb interactions (with_df).'''

    exxdiv = 'ewald'
    _keys = {'exxdiv'}

    def madelung_energy(self):
        '''-nu_M N / 2 for exxdiv='ewald', else 0 (see the module notes).'''
        if self.exxdiv is None:
            return 0.
        if self.exxdiv != 'ewald':
            raise NotImplementedError(self.exxdiv)
        from pyscf.pbc import tools
        return -.5 * tools.pbc.madelung(self.mol, numpy.zeros((1, 3))) * self.mol.nelectron

    def get_ovlp(self, mol=None):
        if mol is None:
            mol = self.mol
        s = numpy.asarray(mol.pbc_intor('int1e_ovlp', hermi=1))
        return spinor_hf.sph2spinor(mol, _sph_blockdiag(s))

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        if getattr(self, 'with_x2c', None) is not None:
            raise NotImplementedError('periodic X2C/X2CAMF one-electron Hamiltonian')
        t = numpy.asarray(mol.pbc_intor('int1e_kin', hermi=1))
        return spinor_hf.sph2spinor(mol, _sph_blockdiag(t + self.with_df.get_nuc()))

    def energy_nuc(self):
        return self.with_df.energy_nuc() + self.madelung_energy()


def periodic(mf, rsdf):
    '''Turn a molecular socutils spinor SCF object on a cell into the
    Gamma-point periodic one with RSDF Coulomb interactions.'''
    if rsdf.cderi is None:
        rsdf.build()
    dfmf = _PeriodicSpinorSCF(mf, rsdf, False)
    dfmf.direct_scf = False   # the MO-based K path needs the full (tagged) DM
    return lib.set_class(dfmf, (_PeriodicSpinorSCF, mf.__class__))


def SpinorRSDF(cell, rsdf):
    '''Gamma-point j-adapted spinor HF of a cell (socutils SpinorSCF + RSDF).'''
    return periodic(spinor_hf.SpinorSCF(cell), rsdf)
