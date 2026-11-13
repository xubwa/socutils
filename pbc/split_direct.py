#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
# Gamma-point Coulomb interactions without stored short-range DF vectors:
# the SR (erfc) part of J and K directly from four-centre integrals (pyscf's
# RangeSeparatedJKBuilder, density screened) in a basis split by exponent, the
# LR part and the nuclei from RSDF(lr='gspace', sr_fit=False).
#
# Generally contracted shells (Dyall, x2c-*) defeat shell-pair screening: a
# shell carries the core and the valence contractions on the same primitives,
# so every quartet pays for all primitives and reaches as far as the most
# diffuse contraction.  The integrals are therefore evaluated in the basis
# split at the exponents `cuts` (socutils.pbc.bench.sr_locality.split_basis:
# each contracted shell -> one shell per exponent interval, the span of its
# contraction columns), an exact linear map of the cell's basis,
#     chi_cell = chi_split T,   D_split = T D T^T,   V_cell = T^T V_split T.
#
# The kernel convention is that of RSDF lr='gspace' (pbc/rsdf.py notes):
#     (mn|ls)' = (mn|ls)_SR,exact + (mn|ls)_LR rows - c0 S_mn S_ls,
# so c_g0, g0_constant, the nuclear terms and energy_nuc are the RSDF ones and
# socutils.pbc.spinor_scf uses the object as it uses an RSDF (with_df).
#

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import ghf
from socutils.pbc.rsdf import RSDF


class SplitDirectJK(lib.StreamObject):
    '''Gamma-point J/K: direct SR in a split basis + exact LR rows.

    Args:
        cell : pyscf.pbc.gto.Cell
        cuts : exponent(s) at which the contracted shells are split (default 8)
        omega : range-separation parameter (as RSDF)
        precision : RSDF precision (LR rows, nuclei)
        sr_precision : cell precision for the direct SR part (image range and
            pyscf's direct_scf_tol); pyscf's 1e-10 gives tol ~1e-15, needlessly tight
        nuc_zeta : Gaussian nuclear exponent(s), as RSDF
    '''

    def __init__(self, cell, cuts=8., omega=0.5, precision=1e-10, sr_precision=1e-8, nuc_zeta=1e12):
        self.cell = cell
        self.cuts = cuts
        self.omega = omega
        self.precision = precision
        self.sr_precision = sr_precision
        self.nuc_zeta = nuc_zeta
        self.verbose = cell.verbose
        self.stdout = cell.stdout
        self.max_memory = cell.max_memory
        self.rsdf = None
        self.split_cell = None
        self.T = None
        self.rsjk = None

    def build(self):
        from pyscf.pbc import gto as pgto
        from pyscf.pbc.scf.rsjk import RangeSeparatedJKBuilder
        from socutils.pbc.bench.sr_locality import split_basis
        log = logger.new_logger(self)
        t0 = (logger.process_clock(), logger.perf_counter())
        cell = self.cell
        # LR rows, nuclei, S, c0: RSDF without the SR fit; its aux basis is unused (one s per atom kind)
        aux = {lab: [[0, [1., 1.]]] for lab in cell._basis}
        r = RSDF(cell, aux, omega=self.omega, precision=self.precision, nuc_zeta=self.nuc_zeta, lr='gspace')
        r.sr_fit = False
        r.cderi_in_memory = True
        r.verbose = self.verbose
        r.build()
        self.rsdf = r
        t1 = log.timer('SplitDirectJK: LR rows and nuclei (%d rows)' % r.nlr, *t0)
        # split basis and the exact map T
        cs = cell.copy(deep=True)
        cs.basis = split_basis(cell._basis, self.cuts)
        cs.build(False, False)
        S_ss = cs.pbc_intor('int1e_ovlp')
        S_sc = pgto.intor_cross('int1e_ovlp', cs, cell)
        T = np.linalg.lstsq(S_ss, S_sc, rcond=1e-12)[0]                 # (nao_split, nao)
        err = abs(T.T @ S_ss @ T - cell.pbc_intor('int1e_ovlp')).max()
        log.info('SplitDirectJK: basis split at %s, nao %d -> %d (%d -> %d shells), |T^T S T - S| %.1e',
                 self.cuts, cell.nao, cs.nao, cell.nbas, cs.nbas, err)
        if err > 1e-8:
            log.warn('SplitDirectJK: the split basis does not represent the cell basis (%.1e)', err)
        self.split_cell, self.T = cs, T
        # direct SR builder: SR kernel only, no smooth-smooth exclusion
        c = cs.copy(deep=False)
        c.omega = -self.omega
        c.precision = self.sr_precision
        rs = RangeSeparatedJKBuilder(c, kpts=np.zeros((1, 3)))
        rs.exclude_dd_block = False
        rs.allow_drv_nodddd = False
        rs.verbose = self.verbose
        rs.build(omega=self.omega)
        self.rsjk = rs
        log.timer('SplitDirectJK: SR builder (supmol %d shells, direct_scf_tol %.1e)'
                  % (rs.supmol_sr.nbas, rs.direct_scf_tol), *t1)
        return self

    # the RSDF attributes spinor_scf reads
    @property
    def c_g0(self):
        return self.rsdf.c_g0

    @property
    def cderi(self):
        return None if self.rsdf is None else self.rsdf.cderi

    def g0_constant(self, nelectron):
        return self.rsdf.g0_constant(nelectron)

    def get_nuc(self):
        return self.rsdf.get_nuc()

    def energy_nuc(self):
        return self.rsdf.energy_nuc()

    @property
    def S(self):
        return self.rsdf.S

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=None, omega=None, exxdiv=None,
               with_g0=False):
        '''J and K (pyscf conventions; dm real or complex, (..., nao, nao)).'''
        if omega:
            raise NotImplementedError('SplitDirectJK.get_jk with omega')
        if self.rsjk is None:
            self.build()
        dm = np.asarray(dm)
        nao = self.cell.nao
        dms = dm.reshape(-1, nao, nao)
        vj, vk = self.rsdf.get_jk(dms, hermi, with_j, with_k, exxdiv=exxdiv, with_g0=with_g0)
        T = self.T
        ds = lib.einsum('pi,nij,qj->npq', T, dms, T)[:, None]          # (n_dm, nk=1, ns, ns)
        vs = self.rsjk._get_jk_sr(ds, hermi=hermi, kpts=np.zeros((1, 3)), with_j=with_j, with_k=with_k)
        k = 0
        if with_j:
            vj = vj.reshape(dms.shape) + lib.einsum('pi,npq,qj->nij', T, vs[k][:, 0], T)
            vj = vj.reshape(dm.shape)
            k += 1
        if with_k:
            vk = vk.reshape(dms.shape) + lib.einsum('pi,npq,qj->nij', T, vs[k][:, 0], T)
            if not np.iscomplexobj(dms):
                vk = vk.real
            vk = vk.reshape(dm.shape)
        if with_j and not np.iscomplexobj(dms):
            vj = vj.real
        return vj, vk


def periodic(mf, jk):
    '''Turn a molecular socutils spinor SCF object on a cell into the Gamma-point periodic one with SplitDirectJK
    Coulomb interactions (the density-matrix path: no MO-based K from stored vectors).'''
    from socutils.pbc import spinor_scf
    from socutils.scf.spinor_hf import spinor2sph, sph2spinor
    spinor_scf.check_nuclear_model(mf.mol, jk)
    if not getattr(jk, '_built', getattr(jk, 'rsjk', None) is not None):
        jk.build()

    class _SplitDirectSCF(spinor_scf._PeriodicSpinorSCF):
        # incremental Fock build: V(D) = V(D_last) + G(D - D_last); the direct SR part screens with the density, so
        # a small change skips most quartets.  A full build every rebuild_cycle calls bounds the accumulated error.
        # The (D, V) pair of the previous call is kept here (pyscf's SCF kernel may not pass dm_last/vhf_last);
        # V is linear in D, so the update is exact for any D.
        incremental = True
        rebuild_cycle = 8
        _n_incremental = 0
        _veff_last = None

        def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
            if mol is None:
                mol = self.mol
            if dm is None:
                dm = self.make_rdm1()
            dm = self._natural_orbitals(dm)                           # untagged (guess) densities -> MO path
            if self._use_mo(dm):                                      # occ-RI: full builds
                self._veff_last = None
                return spinor_scf._PeriodicSpinorSCF.get_veff(self, mol, dm, hermi=hermi)
            dm = np.asarray(dm)
            last = self._veff_last
            if (self.incremental and hermi and last is not None and last[0].shape == dm.shape
                    and self._n_incremental < self.rebuild_cycle - 1):
                self._n_incremental += 1
                ddm = dm - last[0]
                vj, vk = self.get_jk(mol, ddm, hermi=hermi)
                dvj, dvk = self.g0_terms(ddm)
                v = last[1] + vj + dvj - (vk + dvk)
            else:
                self._n_incremental = 0
                v = np.asarray(spinor_scf._PeriodicSpinorSCF.get_veff(self, mol, dm, hermi=hermi))
            if self.incremental and hermi:
                self._veff_last = (dm.copy(), v.copy())
            return v

        def _natural_orbitals(self, dm, tol=1e-8):
            '''Tag an untagged density with its natural orbitals (dm = C n C^H, C^H S C = 1) when its occupations
            are >= -tol, so that a guess density takes the batched occupied-orbital J/K path too.'''
            if (not getattr(self, 'guess_to_mo', True) or getattr(dm, 'mo_coeff', None) is not None
                    or np.ndim(dm) != 2 or not hasattr(self.with_df, 'get_jk_mo')):
                return dm
            dm = np.asarray(dm)
            if abs(dm - dm.conj().T).max() > 1e-10:
                return dm
            S = self.get_ovlp()
            e, v = np.linalg.eigh(S)
            s12 = (v * np.sqrt(e)) @ v.conj().T
            sm12 = (v / np.sqrt(e)) @ v.conj().T
            n, U = np.linalg.eigh(s12 @ dm @ s12)
            if n.min() < -tol:
                return dm
            keep = n > tol
            C = sm12 @ U[:, keep]
            return lib.tag_array(dm, mo_coeff=C, mo_occ=np.maximum(n[keep], 0.))

        def _use_mo(self, dm):
            return (hasattr(self.with_df, 'get_jk_mo') and getattr(dm, 'mo_coeff', None) is not None
                    and getattr(dm, 'mo_occ', None) is not None and np.ndim(dm) == 2
                    and np.min(dm.mo_occ) >= -1e-12)

        def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
            if mol is None:
                mol = self.mol
            if dm is None:
                dm = self.make_rdm1()
            t0 = (logger.process_clock(), logger.perf_counter())
            if self._use_mo(dm):
                vj, vk = self.with_df.get_jk_mo(mol, dm.mo_coeff, dm.mo_occ)
                logger.timer(self, 'vj and vk (occ-RI)', *t0)
                return (vj if with_j else None), (vk if with_k else None)

            def jkbuild(mol, dms, hermi, with_j, with_k, omega=None):
                dms = np.asarray(dms)
                shp = dms.shape
                vj, vk = self.with_df.get_jk(dms.reshape(-1, shp[-2], shp[-1]), hermi, with_j, with_k)
                return (None if vj is None else vj.reshape(shp)), (None if vk is None else vk.reshape(shp))

            dm_sph = spinor2sph(mol, np.asarray(dm))
            vj, vk = ghf.get_jk(mol, dm_sph, hermi, with_j, with_k, jkbuild, omega)
            vj = None if vj is None else sph2spinor(mol, vj)
            vk = None if vk is None else sph2spinor(mol, vk)
            logger.timer(self, 'vj and vk (split direct)', *t0)
            return vj, vk

    dfmf = _SplitDirectSCF(mf, jk, False)
    dfmf.direct_scf = False
    return lib.set_class(dfmf, (_SplitDirectSCF, mf.__class__))


def SpinorSplitDirect(cell, jk):
    '''Gamma-point j-adapted spinor HF of a cell (socutils SpinorSCF + SplitDirectJK).'''
    from socutils.scf import spinor_hf
    return periodic(spinor_hf.SpinorSCF(cell), jk)
