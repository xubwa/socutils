#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
# Gamma-point two-component (j-adapted spinor) Kohn-Sham DFT for a cell:
# socutils.pbc.spinor_scf (periodic one-electron part, RSDF Coulomb, one-centre
# X2CAMF) plus an exchange-correlation potential integrated on all-electron
# atom-centred grids.
#
#   grids      pyscf.pbc.dft.gen_grid.BeckeGrids: atomic radial x angular grids
#              around every atom, Becke partition over the lattice images
#              (the compact core of a heavy atom needs atomic grids; a uniform
#              FFT mesh cannot hold it)
#   numint     pyscf.pbc.dft.numint2c.NumInt2C at Gamma (AO values summed over
#              lattice images), run on the scalar-AO x spin (GHF) form of the
#              spinor density, as socutils.dft.numint_2c does for molecules:
#              an exact unitary change of basis of an exact operator
#   exchange   for hybrids, a fraction of the RSDF exchange with its G = 0
#              term (nu_M S D S); none for pure functionals (XC is local)
#
# Energy: E = tr(D h) + E_J + E_xc - (hyb/2) tr(D K') + E_nn, with E_nn the
# bare nuclear repulsion (the -nu_M N/2 wavefunction constant of energy_nuc
# belongs to HF/CASCI and is taken out, as in spinor_scf).
#
# Usage:
#     r = RSDF(cell, auxbasis, nuc_zeta=spinor_scf.nuclear_zeta(cell)).build()
#     mf = spinor_ks.periodic_ks(spinor_hf.KRHF(cell), r, xc='PBE').x2camf()
#     mf.kernel()
#

import numpy
from pyscf import lib
from pyscf.lib import logger
from socutils.pbc import spinor_scf


class PeriodicSpinorNumInt2C(lib.StreamObject):
    '''Gamma-point 2c numint on a j-adapted spinor density: rotate to the
    scalar-AO x spin basis, integrate with pyscf.pbc.dft.numint2c, rotate back.'''

    def __init__(self, cell, collinear='col'):
        from pyscf.pbc.dft import numint2c
        self.cell = cell
        self._ni = numint2c.NumInt2C()
        self._ni.collinear = collinear
        self._U = numpy.vstack(cell.sph2spinor_coeff())
        self.libxc = self._ni.libxc

    def rsh_and_hybrid_coeff(self, xc, spin=0):
        return self._ni.rsh_and_hybrid_coeff(xc, spin)

    def get_vxc(self, cell, grids, xc, dm, max_memory=2000):
        U = self._U
        dm_g = U @ dm @ U.conj().T
        n, exc, v_g = self._ni.nr_vxc(cell, grids, xc, dm_g, hermi=1, kpt=numpy.zeros(3),
                                      max_memory=max_memory)
        return n, exc, U.conj().T @ v_g @ U


class _PeriodicSpinorKS(spinor_scf._PeriodicSpinorSCF):
    '''Mixin: Kohn-Sham on top of the periodic spinor SCF.'''

    xc = 'PBE'
    grids_level = 4
    _keys = {'xc', 'grids', 'grids_level', '_numint'}

    def _ks_setup(self):
        if getattr(self, 'grids', None) is None:
            from pyscf.pbc.dft.gen_grid import BeckeGrids
            self.grids = BeckeGrids(self.mol)
            self.grids.level = self.grids_level
            self.grids.verbose = self.verbose
            self.grids.stdout = self.stdout
        if self.grids.coords is None:
            self.grids.build()
        if getattr(self, '_numint', None) is None:
            self._numint = PeriodicSpinorNumInt2C(self.mol)

    def hybrid_coeff(self):
        omega, alpha, hyb = self._numint.rsh_and_hybrid_coeff(self.xc)
        if omega:
            raise NotImplementedError('range-separated hybrids with the periodic spinor KS')
        return hyb

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        self._ks_setup()
        t0 = (logger.process_clock(), logger.perf_counter())
        n, exc, vxc = self._numint.get_vxc(mol, self.grids, self.xc, numpy.asarray(dm),
                                           max_memory=self.max_memory - lib.current_memory()[0])
        logger.debug(self, 'nelec by numerical integration = %.8f', n)
        t0 = logger.timer(self, 'vxc', *t0)
        hyb = self.hybrid_coeff()
        if hyb:
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            nu = self.madelung()
            if nu:
                s = self.get_ovlp()
                vk = vk + nu * (s @ dm @ s)
            vk = vk * hyb
            exc = exc - .5 * numpy.einsum('ij,ji', dm, vk).real
            vxc = vxc - vk
        else:
            vj = self.get_jk(mol, dm, hermi=hermi, with_k=False)[0]
        ecoul = .5 * numpy.einsum('ij,ji', dm, vj).real
        return lib.tag_array(vxc + vj, ecoul=ecoul, exc=exc)

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        if h1e is None:
            h1e = self.get_hcore()
        if vhf is None or getattr(vhf, 'ecoul', None) is None:
            vhf = self.get_veff(self.mol, dm)
        e1 = numpy.einsum('ij,ji', h1e, dm).real
        e2 = vhf.ecoul + vhf.exc
        self.scf_summary['e1'] = e1
        self.scf_summary['coul'] = vhf.ecoul
        self.scf_summary['exc'] = vhf.exc
        return e1 + e2, e2

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, 'XC functional = %s (periodic spinor KS, Becke grids level %d)',
                    self.xc, self.grids_level)
        return self


def periodic_ks(mf, rsdf, xc='PBE', collinear='col'):
    '''Gamma-point periodic spinor Kohn-Sham from a molecular socutils spinor
    SCF object on a cell (spinor_hf.SpinorSCF or KRHF).'''
    spinor_scf.check_nuclear_model(mf.mol, rsdf)
    if rsdf.cderi is None:
        rsdf.build()
    ks = _PeriodicSpinorKS(mf, rsdf, False)
    ks.direct_scf = False
    ks = lib.set_class(ks, (_PeriodicSpinorKS, mf.__class__))
    ks.xc = xc
    ks.grids = None
    ks._numint = PeriodicSpinorNumInt2C(mf.mol, collinear)
    return ks
