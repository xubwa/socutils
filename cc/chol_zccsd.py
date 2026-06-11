#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
# Density-fitting (Cholesky) two-component CCSD.
#
# The three-center integrals are held in a small density-fitting object
# (:class:`DFIntegrals`, exposed as ``cc.with_df``) that stores the *scalar*
# AO Cholesky factors and transforms them to the complex spinor MO basis on
# demand.  The MO factors are kept on the eris as the blocks ``Loo, Lov, Lvo,
# Lvv`` (pyscf style), and every term that would otherwise need the O(v^4)
# ``vvvv`` block (the particle-particle ladder, Fvv, Wovvo, and the (T)
# correction) is contracted on the fly from these factors so ``vvvv`` is never
# formed.
#

import numpy as np
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import gccsd
from pyscf.cc import gintermediates as imd
from pyscf import __config__

from socutils.cc import chol_zccsd_t
from socutils.mcscf import zmc_ao2mo

einsum = lib.einsum


class DFIntegrals:
    '''Density-fitting / Cholesky three-center integrals for a (two-component)
    spinor reference.

    Holds the *scalar* AO Cholesky factors ``L_ao`` of shape ``(naux, nao,
    nao)`` with ``(uv|ls)_ao = sum_P L_ao[P,u,v] L_ao[P,l,s]`` and transforms
    them to the complex spinor MO basis on demand::

        Lpq = with_df.ao2mo(mo_coeff)          # (naux, nmo, nmo)
        (pq|rs) = sum_P Lpq[P,p,q] Lpq[P,r,s]
    '''

    def __init__(self, mol, cderi=None, max_error=1e-5):
        self.mol = mol
        nao = mol.nao_nr()
        if cderi is None:
            cderi = zmc_ao2mo.chunked_cholesky(mol, max_error=max_error,
                                               verbose=False)
        self._cderi = np.asarray(cderi).reshape(-1, nao, nao)
        self.mf_type = 'g'

    @property
    def naux(self):
        return self._cderi.shape[0]

    def ao2mo(self, mo_coeff, mf_type=None):
        '''Transform the AO Cholesky factors to the spinor MO basis,
        ``Lpq`` of shape ``(naux, nmo, nmo)``.'''
        if mf_type is None:
            mf_type = self.mf_type
        nao = self.mol.nao_nr()
        orbs = mo_coeff
        if mf_type == 'j':
            orbs = np.vstack(self.mol.sph2spinor_coeff()).dot(orbs)
        L = self._cderi
        Lpq  = einsum('Lib,bj->Lij', einsum('Lab,ai->Lib', L, orbs[:nao].conj()), orbs[:nao])
        Lpq += einsum('Lib,bj->Lij', einsum('Lab,ai->Lib', L, orbs[nao:].conj()), orbs[nao:])
        return Lpq


# ---------------------------------------------------------------------------
# DF intermediates (contract the three-center factors on the fly, no vvvv)
# ---------------------------------------------------------------------------

def _cc_vvvv_df(t1, t2, eris):
    '''Particle-particle ladder ``0.5 sum_ef <ab||ef> tau_ij^ef`` from the DF
    factors (never forms vvvv).'''
    nocc, nvir = t1.shape
    tau = imd.make_tau(t2, t1, t1)
    oovv = np.asarray(eris.oovv)

    tmp = einsum('ijcd,klcd->ijkl', tau, oovv)
    t2new = einsum('ijkl,klab->ijab', tmp, tau) / 8.

    Lvv = eris.Lvv
    L2 = einsum('Pkc,ka->Pac', eris.Lov, t1)
    T2asym = tau - tau.transpose(0, 1, 3, 2)
    for a in range(nvir):
        int2a = einsum('Pc,Pbd->cbd', Lvv[:, a, :] - L2[:, a, :], Lvv)
        t2new[:, :, a, :] += 0.5*einsum('ijcd,cbd->ijb', T2asym, int2a)
        int2a = einsum('Pbc,Pd->bcd', L2, Lvv[:, a, :])
        t2new[:, :, a, :] += 0.5*einsum('ijcd,bcd->ijb', T2asym, int2a)
    return t2new


def _cc_Fvv_df(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc, nocc:]
    fvv = eris.fock[nocc:, nocc:]
    tau_tilde = imd.make_tau(t2, t1, t1, fac=0.5)
    Fae = fvv - 0.5*einsum('me,ma->ae', fov, t1)
    L = einsum('Lic,ic->L', eris.Lov, t1)
    Fae += einsum('L,Lab->ab', L, eris.Lvv)
    Lai = einsum('Lac,ic->Lai', eris.Lvv, t1)
    Fae -= einsum('Lib,Lai->ab', eris.Lov, Lai)
    Fae -= 0.5*einsum('mnaf,mnef->ae', tau_tilde, eris.oovv)
    return Fae


def _cc_Wovvo_df(t1, t2, eris):
    eris_ovvo = -np.asarray(eris.ovov).transpose(0, 1, 3, 2)
    eris_oovo = -np.asarray(eris.ooov).transpose(0, 1, 3, 2)
    tmp1 = einsum('ia,Pba->Pbi', t1, eris.Lvv)
    Wmbej = einsum('Pjc,Pbi->jbci', eris.Lov, tmp1)
    tmp1 = einsum('ia,Pja->Pji', t1, eris.Lov)
    Wmbej -= einsum('Pji,Pbc->jbci', tmp1, eris.Lvv)
    Wmbej -= einsum('nb,mnej->mbej', t1, eris_oovo)
    Wmbej -= 0.5*einsum('jnfb,mnef->mbej', t2, eris.oovv)
    Wmbej -= einsum('jf,nb,mnef->mbej', t1, t1, eris.oovv)
    Wmbej += eris_ovvo
    return Wmbej


def update_amps(cc, t1, t2, eris):
    assert isinstance(eris, _PhysicistsERIs)
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc, nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    tau = imd.make_tau(t2, t1, t1)

    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)
    Woooo = imd.cc_Woooo(t1, t2, eris)
    Fvv = _cc_Fvv_df(t1, t2, eris)            # ovvv terms from DF factors

    Fvv[np.diag_indices(nvir)] -= mo_e_v
    Foo[np.diag_indices(nocc)] -= mo_e_o

    # T1 equation
    t1new  =  einsum('ie,ae->ia', t1, Fvv)
    t1new += -einsum('ma,mi->ia', t1, Foo)
    t1new +=  einsum('imae,me->ia', t2, Fov)
    t1new += -einsum('nf,naif->ia', t1, eris.ovov)
    t1new += -0.5*einsum('mnae,mnie->ia', t2, eris.ooov)
    t1new += fov.conj()

    # T2 equation
    Ftmp = Fvv - 0.5*einsum('mb,me->be', t1, Fov)
    tmp = einsum('ijae,be->ijab', t2, Ftmp)
    t2new = tmp - tmp.transpose(0, 1, 3, 2)
    Ftmp = Foo + 0.5*einsum('je,me->mj', t1, Fov)
    tmp = einsum('imab,mj->ijab', t2, Ftmp)
    t2new -= tmp - tmp.transpose(1, 0, 2, 3)
    t2new += np.asarray(eris.oovv).conj()
    t2new += 0.5*einsum('mnab,mnij->ijab', tau, Woooo)

    # particle-particle ladder (all vvvv terms) from the DF factors
    t2new += _cc_vvvv_df(t1, t2, eris)

    Wovvo = _cc_Wovvo_df(t1, t2, eris)        # ovvv terms from DF factors

    # t2new ovvv term  (sum_e t1[ie] <je||ba>)  from DF factors
    tmp = einsum('ic,Pca->Pia', t1, eris.Lvv.conj())
    tmp1 = einsum('Pjb,Pia->ijab', eris.Lov.conj(), tmp)
    tmp = einsum('ic,Pcb->Pib', t1, eris.Lvv.conj())
    tmp1 -= einsum('Pja,Pib->ijab', eris.Lov.conj(), tmp)
    t2new += (tmp1 - tmp1.transpose(1, 0, 2, 3))

    # t1new ovvv term  (-0.5 sum_mef t2[imef] <ma||ef>)  from DF factors
    tmp = -0.5*einsum('ijab,Pja->iPb', t2, eris.Lov)
    t1new += einsum('iPb,Pcb->ic', tmp, eris.Lvv)
    tmp = 0.5*einsum('ijab,Pjb->iPa', t2, eris.Lov)
    t1new += einsum('iPa,Pca->ic', tmp, eris.Lvv)

    tmp = einsum('imae,mbej->ijab', t2, Wovvo)
    tmp -= -einsum('ie,ma,mbje->ijab', t1, t1, eris.ovov)
    tmp = tmp - tmp.transpose(1, 0, 2, 3)
    tmp = tmp - tmp.transpose(0, 1, 3, 2)
    t2new += tmp

    tmp = einsum('ma,ijmb->ijab', t1, np.asarray(eris.ooov).conj())
    t2new -= (tmp - tmp.transpose(0, 1, 3, 2))

    eia = mo_e_o[:, None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    t1new /= eia
    t2new /= eijab
    return t1new, t2new


class DFCCSD(gccsd.GCCSD):
    '''Density-fitting two-component CCSD.

    The three-center integrals live in ``self.with_df`` (a :class:`DFIntegrals`
    object, or any object with a compatible ``ao2mo`` method / ``naux``).
    '''

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None,
                 mf_type='g', cderi=None, with_df=None):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.mf_type = mf_type
        if with_df is None:
            with_df = DFIntegrals(mf.mol, cderi=cderi)
        with_df.mf_type = mf_type
        self.with_df = with_df

    update_amps = update_amps

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        return _make_df_eris(self, mo_coeff)

    def ccsd_t(self, t1=None, t2=None, eris=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None:
            eris = _make_df_eris(self, self.mo_coeff)
        return chol_zccsd_t.kernel(self, eris, t1, t2, self.verbose)

    # vvvv-free EOM-CCSD entry points (see socutils.cc.eom_zccsd_direct)
    def eomip(self, nroots=1, **kwargs):
        from socutils.cc.eom_zccsd_direct import EOMIP
        return EOMIP(self).kernel(nroots=nroots, **kwargs)

    def eomea(self, nroots=1, **kwargs):
        from socutils.cc.eom_zccsd_direct import EOMEA
        return EOMEA(self).kernel(nroots=nroots, **kwargs)

    def eomee(self, nroots=1, **kwargs):
        from socutils.cc.eom_zccsd_direct import EOMEE
        return EOMEE(self).kernel(nroots=nroots, **kwargs)

    ipccsd = eomip
    eaccsd = eomea
    eeccsd = eomee

    def lambda_ccsd_t(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        '''Lambda-CCSD(T) energy correction (CCSD(T)_Lambda): the (T) triples
        contracted with the converged Lambda amplitudes on the bra side.
        Solves the CCSD Lambda equations first if l1/l2 are not available.'''
        from socutils.cc import gccsd_t
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None:
            eris = _make_df_eris(self, self.mo_coeff)
        if l1 is None or l2 is None:
            if getattr(self, 'l1', None) is None:
                self.solve_lambda(t1, t2, eris=eris)
            l1, l2 = self.l1, self.l2
        return gccsd_t.kernel_lambda(self, eris, t1, t2, l1, l2, self.verbose)

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None,
                     with_t=False, **kwargs):
        from socutils.cc import lambda_zccsd, lambda_t_zccsd
        mod = lambda_t_zccsd if with_t else lambda_zccsd
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None:
            eris = _make_df_eris(self, self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
            mod.kernel(self, eris, t1, t2, l1, l2, **kwargs)
        return self.l1, self.l2

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, eris=None,
                  ao_repr=False, with_t=False):
        from socutils.cc import rdm_zccsd
        if eris is None:
            eris = _make_df_eris(self, self.mo_coeff)
        return rdm_zccsd.make_rdm1(self, t1, t2, l1, l2, eris, ao_repr, with_t)

    def dip_moment(self, t1=None, t2=None, l1=None, l2=None, eris=None,
                   with_t=False, unit='Debye'):
        from socutils.cc import rdm_zccsd
        if eris is None:
            eris = _make_df_eris(self, self.mo_coeff)
        return rdm_zccsd.dip_moment(self, t1, t2, l1, l2, eris, with_t, unit)


# Backward-compatible alias
ZCCSD = DFCCSD


class _PhysicistsERIs(gccsd._PhysicistsERIs):
    '''<pq||rs> = <pq|rs> - <pq|sr>, plus the DF factor blocks Loo/Lov/Lvo/Lvv.'''

    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None
        self.e_hf = None
        self.orbspin = None
        self.oooo = self.ooov = self.oovv = None
        self.ovvo = self.ovov = self.ovvv = self.vvvv = None
        self.Loo = self.Lov = self.Lvo = self.Lvv = None

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        mo_idx = ccsd.get_frozen_mask(mycc)
        mo_coeff = mo_coeff[:, mo_idx]
        self.mo_coeff = mo_coeff

        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(vhf=vhf, dm=dm)
        self.fock = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        self.e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf)
        self.nocc = mycc.nocc
        self.mol = mycc.mol
        mo_e = self.mo_energy = self.fock.diagonal().real
        gap = abs(mo_e[:self.nocc, None] - mo_e[None, self.nocc:]).min()
        if gap < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap %s too small for DF-ZCCSD', gap)
        return self


def _make_df_eris(mycc, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    assert eris.mo_coeff.dtype == complex

    Lpq = mycc.with_df.ao2mo(eris.mo_coeff, mycc.mf_type)
    eris.Loo = Lpq[:, :nocc, :nocc]
    eris.Lov = Lpq[:, :nocc, nocc:]
    eris.Lvo = Lpq[:, nocc:, :nocc]
    eris.Lvv = Lpq[:, nocc:, nocc:]

    # antisymmetrized blocks carrying at most two virtual indices (no vvvv)
    oooo = einsum('Lij,Lkl->ijkl', eris.Loo, eris.Loo)
    eris.oooo = oooo.transpose(0, 2, 1, 3) - oooo.transpose(0, 2, 3, 1)
    ooov = einsum('Lij,Lkl->ijkl', eris.Loo, eris.Lov)
    eris.ooov = ooov.transpose(0, 2, 1, 3) - ooov.transpose(2, 0, 1, 3)
    oovv = einsum('Lij,Lkl->ijkl', eris.Loo, eris.Lvv)
    ovov = einsum('Lij,Lkl->ijkl', eris.Lov, eris.Lov)
    ovvo = einsum('Lij,Lkl->ijkl', eris.Lov, eris.Lvo)
    eris.oovv = ovov.transpose(0, 2, 1, 3) - ovov.transpose(0, 2, 3, 1)
    eris.ovov = oovv.transpose(0, 2, 1, 3) - ovvo.transpose(0, 2, 3, 1)
    eris.ovvo = ovvo.transpose(0, 2, 1, 3) - oovv.transpose(0, 2, 3, 1)
    ovvv = einsum('Lia,Lbc->iabc', eris.Lov, eris.Lvv)
    eris.ovvv = ovvv.transpose(0, 2, 1, 3) - ovvv.transpose(0, 2, 3, 1)
    log.timer('DF-CCSD integral transformation', *cput0)
    return eris


if __name__ == '__main__':
    from pyscf import gto, scf, cc
    from socutils.scf import spinor_hf
    mol = gto.Mole()
    mol.atom = [[8, (0., 0., 0.)],
                [1, (0., -0.757, 0.587)],
                [1, (0., 0.757, 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.verbose = 0
    mol.build()

    mfg = scf.GHF(mol).run(conv_tol=1e-12)
    gcc = cc.GCCSD(mfg).run(conv_tol=1e-10)
    print('pyscf  GCCSD   Ecorr', gcc.e_corr, ' (T)', gcc.ccsd_t())

    mf = spinor_hf.SCF(mol).run()
    mycc = DFCCSD(mf, mf_type='j')
    ecc = mycc.kernel()[0]
    print('DF-CCSD        Ecorr', ecc, ' (T)', mycc.ccsd_t().real)
