#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
# Two-component (spinor) CCSD that never builds or stores the ``vvvv`` block.
#
# The particle-particle ladder term
#
#     0.5 * sum_ef <ab||ef> tau_ij^ef  ==  sum_ef (ae|bf) tau_ij^ef
#
# (the second equality follows from the antisymmetry of tau) is evaluated
# directly in the *scalar* AO basis.  For a two-component calculation the
# scalar spatial AO space is roughly half the dimension of the spinor MO
# space, so AO-space quantities are ~16x smaller than the spinor ``vvvv``
# tensor.  This bypasses both the O(v^4) storage and the dominant part of the
# integral transformation: only the blocks carrying at least one occupied
# index are transformed, ``vvvv`` is never formed.
#
# Two representations of the AO Coulomb operator are supported:
#   ladder='ao'   -- exact incore scalar ERIs (no approximation), and
#   ladder='chol' -- density-fitted / Cholesky factors (sub-mEh error,
#                    lower memory and faster for large bases).
#

import numpy as np
from functools import reduce

from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import gccsd
from pyscf.ao2mo import nrr_outcore
from pyscf import __config__

from socutils.cc import gintermediates as imd
from socutils.cc import zccsd as _zccsd

einsum = lib.einsum


def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i + step, end)


def _ao_spin_blocks(eris):
    '''Virtual MO coefficients in the (scalar AO x spin) basis, split into the
    two spin components.  ``eris.mo_ao_spin`` has shape ``(2*nao, nmo)`` with
    the spin-alpha scalar-AO block stacked on top of the spin-beta block.'''
    nocc = eris.nocc
    nao = eris.nao_sph
    Cv = eris.mo_ao_spin[:, nocc:]
    return [Cv[:nao], Cv[nao:]]


def _ao_contract(eris, T):
    '''Contract the AO Coulomb operator with the half back-transformed
    amplitude ``T_ij^{nu sigma}`` (the electron-2 indices), returning

        G_ij^{mu lambda} = sum_{nu sigma} (mu nu | lambda sigma) T_ij^{nu sigma}.

    Note the contracted indices ``nu`` and ``sigma`` belong to *different*
    electron pairs of the chemist's integral, so the operator is stored in the
    ``[(nu,sigma), (mu,lambda)]`` layout for a single GEMM.  ``T`` has shape
    ``(..., nao, nao)``; the leading axes are carried through.'''
    nao = eris.nao_sph
    lead = T.shape[:-2]
    Tm = T.reshape(-1, nao * nao)               # flatten (nu, sigma)
    if eris.ao_eri is not None:                 # exact dense scalar ERIs
        Gm = lib.dot(Tm, eris.ao_eri)           # ao_eri: [(nu,sigma),(mu,lambda)]
        return Gm.reshape(*lead, nao, nao)
    # density fitted / Cholesky:  (mn|ls) = sum_c L[c,m,n] L[c,l,s]
    Lc = eris.ao_chol                           # (naux, nao, nao)
    T4 = T.reshape(-1, nao, nao)                # (X, nu, sigma)
    Q = einsum('cls,Xns->Xcln', Lc, T4)         # contract sigma
    G = einsum('cmn,Xcln->Xml', Lc, Q)          # contract c and nu
    return G.reshape(*lead, nao, nao)


def update_t2_vvvv_direct(t1, t2, tau, eris):
    '''AO-direct analogue of :func:`gintermediates.update_t2_vvvv`.

    The ladder ``sum_ef <ab||ef> tau_ij^ef`` is built in the scalar AO basis
    so the spinor ``vvvv`` tensor is never required.  The remaining ``ovvv``
    and ``oovv`` contributions are identical to the in-core routine.
    '''
    nocc, nvir = t1.shape
    Cv = _ao_spin_blocks(eris)
    t2_new = np.zeros(t2.shape, dtype=t2.dtype)

    # AO-direct particle-particle ladder, blocked over the first occupied index
    # to bound the peak memory at O(nocc * nao^2).
    nao = eris.nao_sph
    blk = max(1, int(2000*1e6/16/max(1, nocc*nao*nao)))
    for i0, i1 in prange(0, nocc, blk):
        taui = tau[i0:i1]                                   # (di,no,nv,nv)
        ladder = np.zeros((i1-i0, nocc, nvir, nvir), dtype=t2.dtype)
        for C1 in Cv:
            for C2 in Cv:
                T = einsum('ne,ijef,sf->ijns', C1, taui, C2)
                G = _ao_contract(eris, T)
                ladder += einsum('ma,ijml,lb->ijab', C1.conj(), G, C2.conj())
        # 2x because <ab||ef> = (ae|bf) - (af|be) and sum_ef (af|be) tau = -sum (ae|bf) tau
        t2_new[i0:i1] += 2.0 * ladder

    eris_ovvv = np.asarray(eris.ovvv)
    tmp = einsum('maef,ijef->maij', eris_ovvv, tau)
    tmp = einsum('maij,mb->baij', tmp, t1)
    t2_new += einsum('baij->ijab', tmp) - einsum('baij->ijba', tmp)
    tmp = einsum('mnef,ijef->mnij', 0.25*np.asarray(eris.oovv), tau)
    t2_new += einsum('mnij,mnab->ijab', tmp, tau)
    return 0.5*t2_new


def update_amps(cc, t1, t2, eris):
    '''Identical to :func:`zccsd.update_amps` except the particle-particle
    ladder is evaluated AO-direct (no ``vvvv``).'''
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc, nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    tau = imd.make_tau(t2, t1, t1)

    Fvv = imd.cc_Fvv(t1, t2, eris)
    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)
    Woooo = imd.cc_Woooo(t1, t2, eris)
    Wovvo = imd.cc_Wovvo(t1, t2, eris)

    Fvv[np.diag_indices(nvir)] -= mo_e_v
    Foo[np.diag_indices(nocc)] -= mo_e_o

    # T1 equation
    t1new = einsum('ie,ae->ia', t1, Fvv)
    t1new += -einsum('ma,mi->ia', t1, Foo)
    t1new += einsum('imae,me->ia', t2, Fov)
    t1new += -einsum('nf,naif->ia', t1, eris.ovov)
    t1new += -0.5*einsum('imef,maef->ia', t2, eris.ovvv)
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
    t2new += update_t2_vvvv_direct(t1, t2, tau, eris)
    tmp = einsum('imae,mbej->ijab', t2, Wovvo)
    tmp -= -einsum('ie,ma,mbje->ijab', t1, t1, eris.ovov)
    tmp = tmp - tmp.transpose(1, 0, 2, 3)
    tmp = tmp - tmp.transpose(0, 1, 3, 2)
    t2new += tmp
    tmp = einsum('ie,jeba->ijab', t1, np.array(eris.ovvv).conj())
    t2new += (tmp - tmp.transpose(1, 0, 2, 3))
    tmp = einsum('ma,ijmb->ijab', t1, np.asarray(eris.ooov).conj())
    t2new -= (tmp - tmp.transpose(0, 1, 3, 2))

    eia = mo_e_o[:, None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    t1new /= eia
    t2new /= eijab
    return t1new, t2new


class DirectZCCSD(_zccsd.ZCCSD):
    '''Two-component CCSD with an AO-direct particle-particle ladder.

    Parameters
    ----------
    ladder : str
        ``'ao'`` uses the exact incore scalar ERIs (default), ``'chol'`` uses
        a Cholesky/density-fitted factorisation of the scalar ERIs.
    cderi : ndarray, optional
        Pre-computed AO Cholesky vectors (only used when ``ladder='chol'``).
    max_error : float
        Threshold for the on-the-fly Cholesky decomposition.
    '''

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None,
                 ladder='ao', cderi=None, max_error=1e-5):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.ladder = ladder
        self.cderi = cderi
        self.max_error = max_error
        self.eris = None

    update_amps = update_amps

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        self.eris = _make_eris_direct(self, mo_coeff)
        return self.eris

    def ccsd_t(self, t1=None, t2=None, eris=None):
        from socutils.cc import gccsd_t
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None:
            eris = self.eris if self.eris is not None else self.ao2mo()
        return gccsd_t.kernel(self, eris, t1, t2, self.verbose)


class _PhysicistsERIs(gccsd._PhysicistsERIs):
    '''<pq||rs> = <pq|rs> - <pq|sr>, with the ``vvvv`` block omitted.'''

    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None
        self.e_hf = None
        self.orbspin = None
        self.oooo = self.ooov = self.oovv = None
        self.ovvo = self.ovov = self.ovvv = None
        self.vvvv = None
        # AO-direct ladder data
        self.mo_ao_spin = None
        self.nao_sph = None
        self.ao_eri = None
        self.ao_chol = None

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
            logger.warn(mycc, 'HOMO-LUMO gap %s too small for ZCCSD', gap)
        return self


def _make_eris_direct(mycc, mo_coeff=None):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mol = mycc.mol
    nocc = eris.nocc
    nao = mol.nao_nr()
    eris.nao_sph = nao

    # Spinor MOs expressed in the (scalar AO x spin) basis.
    c2 = np.vstack(mol.sph2spinor_coeff())          # (2*nao, n2c)
    eris.mo_ao_spin = c2.dot(eris.mo_coeff)         # (2*nao, nmo)
    mog = eris.mo_ao_spin

    if mycc.ladder == 'chol':
        # Fully density-fitted: the occupied-row blocks and the ladder share
        # the same scalar Cholesky factors, so the O(nao^4) ERIs are never
        # formed in core.
        from socutils.mcscf import zmc_ao2mo
        if mycc.cderi is None:
            cderi = zmc_ao2mo.chunked_cholesky(mol, max_error=mycc.max_error,
                                               verbose=mycc.verbose > 4)
        else:
            cderi = mycc.cderi
        Lc = np.asarray(cderi).reshape(-1, nao, nao)
        eris.ao_chol = Lc
        eris.ao_eri = None
        moa, mob = mog[:nao], mog[nao:]
        # chol in the spinor MO basis (sum over the two spin components)
        chol_mo = (einsum('cmn,mp,nq->cpq', Lc, moa.conj(), moa) +
                   einsum('cmn,mp,nq->cpq', Lc, mob.conj(), mob))
        _eris_from_cholmo(eris, chol_mo, nocc)
    else:
        # Exact integrals.  Transform only the occupied-row chemist integrals
        # (orbo, mo, mo, mo) with pyscf's optimised sph->j-spinor AO2MO (C
        # code + 8-fold AO symmetry), then assemble the antisymmetrized blocks.
        # vvvv is never transformed.
        mo = eris.mo_coeff
        nmo = mo.shape[1]
        orbo = mo[:, :nocc]
        feri = lib.H5TmpFile()
        nrr_outcore.general(mol, (orbo, mo, mo, mo), feri, 'eri',
                            intor='int2e_sph', motype='j-spinor',
                            max_memory=mycc.max_memory, verbose=mycc.verbose)
        g = np.asarray(feri['eri']).reshape(nocc, nmo, nmo, nmo)  # (iq|rs) chemist
        feri = None
        no = nocc
        eris.oooo = g[:,:no,:no,:no].transpose(0,2,1,3) - g[:,:no,:no,:no].transpose(0,2,3,1)
        eris.ooov = g[:,:no,:no,no:].transpose(0,2,1,3) - g[:,no:,:no,:no].transpose(0,2,3,1)
        eris.oovv = g[:,no:,:no,no:].transpose(0,2,1,3) - g[:,no:,:no,no:].transpose(0,2,3,1)
        eris.ovov = g[:,:no,no:,no:].transpose(0,2,1,3) - g[:,no:,no:,:no].transpose(0,2,3,1)
        eris.ovvo = g[:,no:,no:,:no].transpose(0,2,1,3) - g[:,:no,no:,no:].transpose(0,2,3,1)
        eris.ovvv = g[:,no:,no:,no:].transpose(0,2,1,3) - g[:,no:,no:,no:].transpose(0,2,3,1)
        g = None
        # Scalar AO Coulomb operator for the ladder, stored as
        # [(nu,sigma), (mu,lambda)] so the contraction is a single GEMM:
        # G[mu,lambda] = sum_{nu,sigma} (mu nu | lambda sigma) T[nu,sigma].
        eri = mol.intor('int2e_sph').reshape(nao, nao, nao, nao)
        eris.ao_eri = eri.transpose(1, 3, 0, 2).reshape(nao*nao, nao*nao).copy()
        eris.ao_chol = None

    log.timer('CCSD integral transformation (vvvv-free)', *cput0)
    return eris


def _eris_from_cholmo(eris, chol_mo, nocc):
    '''Build the antisymmetrized occupied-row blocks from the Cholesky factors
    in the MO basis (mirrors :func:`chol_zccsd._make_eris_FromChol`).'''
    oo = chol_mo[:, :nocc, :nocc]
    ov = chol_mo[:, :nocc, nocc:]
    vo = chol_mo[:, nocc:, :nocc]
    vv = chol_mo[:, nocc:, nocc:]
    oooo = einsum('Lij,Lkl->ijkl', oo, oo)
    eris.oooo = oooo.transpose(0, 2, 1, 3) - oooo.transpose(0, 2, 3, 1)
    ooov = einsum('Lij,Lkl->ijkl', oo, ov)
    eris.ooov = ooov.transpose(0, 2, 1, 3) - ooov.transpose(2, 0, 1, 3)
    oovv = einsum('Lij,Lkl->ijkl', oo, vv)
    ovov = einsum('Lij,Lkl->ijkl', ov, ov)
    ovvo = einsum('Lij,Lkl->ijkl', ov, vo)
    eris.oovv = ovov.transpose(0, 2, 1, 3) - ovov.transpose(0, 2, 3, 1)
    eris.ovov = oovv.transpose(0, 2, 1, 3) - ovvo.transpose(0, 2, 3, 1)
    eris.ovvo = ovvo.transpose(0, 2, 1, 3) - oovv.transpose(0, 2, 3, 1)
    ovvv = einsum('Lia,Lbc->iabc', ov, vv)
    eris.ovvv = ovvv.transpose(0, 2, 1, 3) - ovvv.transpose(0, 2, 3, 1)


if __name__ == '__main__':
    from pyscf import gto
    from socutils.scf import spinor_hf
    from socutils.cc.zccsd import ZCCSD

    mol = gto.Mole()
    mol.atom = [[8, (0., 0., 0.)],
                [1, (0., -0.757, 0.587)],
                [1, (0., 0.757, 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.verbose = 0
    mol.build()

    mf = spinor_hf.SCF(mol)
    mf.kernel()

    e_ref = ZCCSD(mf, frozen=2).kernel()[0]
    e_ao = DirectZCCSD(mf, frozen=2, ladder='ao').kernel()[0]
    e_chol = DirectZCCSD(mf, frozen=2, ladder='chol').kernel()[0]
    print('exact  ZCCSD corr        = %.12f' % e_ref)
    print('direct ZCCSD (ao)   corr = %.12f  diff %.2e' % (e_ao, abs(e_ao-e_ref)))
    print('direct ZCCSD (chol) corr = %.12f  diff %.2e' % (e_chol, abs(e_chol-e_ref)))
