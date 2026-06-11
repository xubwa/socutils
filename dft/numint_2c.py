#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
# Fast 2-component numerical integration for a j-adapted spinor DFT reference.
#
# The relativistic numint (pyscf.dft.r_numint) evaluates the *complex* j-spinor
# AOs on the grid.  For a non-relativistic-integral X2C/spinor calculation the
# exact same Vxc operator can be obtained in the cheaper GHF (scalar-AO x spin)
# representation: rotate the spinor density matrix to the scalar-AO x spin basis
# with the fixed transform U = sph2spinor_coeff, let the standard real-AO
# 2-component numint (pyscf.dft.numint2c.NumInt2C) do the grid work, and rotate
# the resulting Vxc / fxc back.  This is an exact unitary change of basis of an
# exact operator -- it reproduces r_numint to machine precision -- but evaluates
# real scalar AOs on the grid instead of complex spinor AOs, which is several
# times faster.
#
# Note on TDDFT (fxc): the default *collinear* kernel only couples the diagonal
# (alpha-alpha, beta-beta) spin blocks and structurally drops the off-diagonal
# (alpha-beta) spin-flip response, so it cannot describe spin-mixed (SOC)
# excitations.  Complex orbitals / amplitudes themselves are fine.  For a
# correct two-component response set ``collinear='mcol'`` (multi-collinear,
# mcfun), which fills the full spin structure and handles complex transition
# densities.
#

import numpy as np
from pyscf import lib
from pyscf.dft import numint2c


class SpinorNumInt2C(numint2c.NumInt2C):
    '''NumInt2C that runs on a j-adapted spinor reference by rotating to and
    from the scalar-AO x spin (GHF) representation.'''

    # mcfun integrates the XC over `spin_samples` spin-quantization axes per
    # grid point (default 770), all in pure Python -- the dominant cost of the
    # multicollinear (mcol) path.  Default to a coarse angular quadrature here;
    # raise it (e.g. 770) for production-quality non-collinear results.
    spin_samples = 14

    def __init__(self, mol):
        numint2c.NumInt2C.__init__(self)
        self.mol = mol
        self._U = None

    @property
    def U(self):
        '''sph2spinor transform, shape (2*nao, n2c): spinor AO -> scalar x spin.'''
        if self._U is None:
            self._U = np.vstack(self.mol.sph2spinor_coeff())
        return self._U

    # spinor DM -> GHF DM,  V(GHF) -> V(spinor)
    def _dm_to_ghf(self, dm):
        U = self.U
        dm = np.asarray(dm)
        if dm.ndim == 2:
            return U.dot(dm).dot(U.conj().T)
        return lib.einsum('mp,...pq,nq->...mn', U, dm, U.conj())

    def _mat_to_spinor(self, mat):
        U = self.U
        mat = np.asarray(mat)
        if mat.ndim == 2:
            return U.conj().T.dot(mat).dot(U)
        return lib.einsum('mp,...mn,nq->...pq', U.conj(), mat, U)

    def get_vxc(self, mol, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
                max_memory=2000, verbose=None):
        dm_g = self._dm_to_ghf(dms)
        n, exc, v_g = super().get_vxc(mol, grids, xc_code, dm_g, spin,
                                      relativity, hermi, max_memory, verbose)
        return n, exc, self._mat_to_spinor(v_g)
    nr_vxc = nr_gks_vxc = get_vxc

    def get_fxc(self, mol, grids, xc_code, dm0, dms, spin=0, relativity=0,
                hermi=0, rho0=None, vxc=None, fxc=None, max_memory=2000,
                verbose=None):
        dm0_g = None if dm0 is None else self._dm_to_ghf(dm0)
        dms_g = self._dm_to_ghf(dms)
        _fxc = super().get_fxc
        if self.collinear[0] == 'c' and np.iscomplexobj(dms_g):
            # The collinear fxc kernel is real and linear in the perturbing
            # density, but pyscf routes it through the 1-component nr_uks_fxc,
            # which rejects complex (TDDFT amplitude) density matrices.  Split
            # the transition density into real/imag parts and recombine -- this
            # is exact for the collinear kernel.  (It only couples the diagonal
            # alpha-alpha/beta-beta spin blocks: the spin-flip response is
            # absent by construction.  Use collinear='mcol' for that.)
            fr = _fxc(mol, grids, xc_code, dm0_g, dms_g.real.copy(), spin,
                      relativity, hermi, rho0, vxc, fxc, max_memory, verbose)
            fi = _fxc(mol, grids, xc_code, dm0_g, dms_g.imag.copy(), spin,
                      relativity, hermi, rho0, vxc, fxc, max_memory, verbose)
            v_g = fr + 1j * fi
        else:
            v_g = _fxc(mol, grids, xc_code, dm0_g, dms_g, spin, relativity,
                       hermi, rho0, vxc, fxc, max_memory, verbose)
        return self._mat_to_spinor(v_g)
    nr_fxc = nr_gks_fxc = get_fxc

    def nr_nlc_vxc(self, mol, grids, xc_code, dms, spin=0, relativity=0,
                   hermi=1, max_memory=2000, verbose=None):
        dm_g = self._dm_to_ghf(dms)
        n, exc, v_g = super().nr_nlc_vxc(mol, grids, xc_code, dm_g, spin,
                                         relativity, hermi, max_memory, verbose)
        return n, exc, self._mat_to_spinor(v_g)

    def get_rho(self, mol, dm, grids, max_memory=2000):
        return super().get_rho(mol, self._dm_to_ghf(dm), grids, max_memory)

    def cache_xc_kernel(self, mol, grids, xc_code, mo_coeff, mo_occ, spin=0,
                        max_memory=2000):
        '''mo_coeff is in the spinor basis; rotate to scalar x spin first.'''
        mo_g = self.U.dot(mo_coeff)
        return super().cache_xc_kernel(mol, grids, xc_code, mo_g, mo_occ, spin,
                                       max_memory)
