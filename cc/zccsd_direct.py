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


# ---------------------------------------------------------------------------
# AO-direct evaluation of the ovvv (<ma||ef>) contractions.
#
# When ``eris.E4`` is set (exact 'ao' mode) ``ovvv`` is never built; the five
# contractions below are evaluated as generalised J/K builds with the scalar
# AO Coulomb operator, folding the amplitudes into nao x nao densities so no
# nvir^3 intermediate ever appears.  When ``ovvv`` is stored ('chol' mode) the
# plain einsum contractions are used.
# ---------------------------------------------------------------------------

def _spin_mo(eris):
    '''Occupied/virtual/full MO coefficients in the (scalar AO x spin) basis,
    one matrix per spin component.'''
    if eris._spin_mo_cache is None:
        nocc = eris.nocc
        nao = eris.nao_sph
        Ca, Cb = eris.mo_ao_spin[:nao], eris.mo_ao_spin[nao:]
        Co = [Ca[:, :nocc], Cb[:, :nocc]]
        Cv = [Ca[:, nocc:], Cb[:, nocc:]]
        Cf = [Ca, Cb]
        eris._spin_mo_cache = (Co, Cv, Cf)
    return eris._spin_mo_cache


def _ovvv_fvv(t1, eris):
    '''sum_mf t1[m,f] <ma||fe>  ->  [a,e]   (cc_Fvv contribution).'''
    if eris.E4 is None:
        return einsum('mf,amef->ae', t1, np.asarray(eris.ovvv).transpose(1, 0, 3, 2))
    Co, Cv, Cf = _spin_mo(eris)
    E = eris.E4
    J = 0.
    for s in (0, 1):
        T1m = Co[s].conj() @ t1 @ Cv[s].T
        J = J + einsum('uv,uvls->ls', T1m, E)
    partA = 0.
    for s in (0, 1):
        partA = partA + einsum('ls,la,se->ae', J, Cv[s].conj(), Cv[s])
    partB = 0.
    for s1 in (0, 1):
        for s2 in (0, 1):
            K = Co[s1].conj() @ t1 @ Cv[s2].T
            Ex = einsum('us,unls->nl', K, E)
            partB = partB + einsum('nl,ne,la->ae', Ex, Cv[s1], Cv[s2].conj())
    return partA - partB


def _ovvv_wovvo(t1, eris):
    '''sum_f t1[j,f] <mb||ef>  ->  [m,b,e,j]   (cc_Wovvo contribution).

    The cheap legs are contracted first so neither an O(nao^4 nvir) nor an
    O(nao^2 nvir^2) intermediate appears.  For the Coulomb part (me|bf) the
    t1-dressed f index (occupied count) is contracted first; for the exchange
    part (mf|be) the occupied bra indices are contracted first.'''
    if eris.E4 is None:
        return einsum('jf,mbef->mbej', t1, np.asarray(eris.ovvv))
    Co, Cv, Cf = _spin_mo(eris)
    E = eris.E4
    res = 0.
    for s2 in (0, 1):                                  # Coulomb (me|bf)
        dt = t1 @ Cv[s2].T                             # [j,sigma]
        Ej = einsum('unls,js->unlj', E, dt)            # sigma -> j  (occ first)
        for s1 in (0, 1):
            A1 = einsum('um,unlj->mnlj', Co[s1].conj(), Ej)
            A2 = einsum('ne,mnlj->melj', Cv[s1], A1)
            res = res + einsum('melj,lb->mbej', A2, Cv[s2].conj())
    for s1 in (0, 1):                                  # exchange (mf|be)
        dt1 = t1 @ Cv[s1].T                            # [j,nu]
        Em = einsum('um,unls->mnls', Co[s1].conj(), E)  # mu -> m  (occ first)
        Emj = einsum('jn,mnls->mjls', dt1, Em)         # nu -> j
        for s2 in (0, 1):
            Eb = einsum('mjls,lb->mjbs', Emj, Cv[s2].conj())
            res = res - einsum('mjbs,se->mbej', Eb, Cv[s2])
    return res


def _ovvv_ladder(tau, eris):
    '''sum_ef <ma||ef> tau[i,j,e,f]  ->  [m,a,i,j]   (ladder ovvv term).'''
    if eris.E4 is None:
        return einsum('maef,ijef->maij', np.asarray(eris.ovvv), tau)
    Co, Cv, Cf = _spin_mo(eris)
    E = eris.E4
    res = 0.
    for s1 in (0, 1):
        for s2 in (0, 1):
            tauAO = einsum('ne,ijef,sf->ijns', Cv[s1], tau, Cv[s2])
            G = einsum('ijns,unls->ijul', tauAO, E)
            res = res + 2.0*einsum('ijul,um,la->maij', G, Co[s1].conj(), Cv[s2].conj())
    return res


def _ovvv_t1(t2, eris):
    '''sum_mef t2[i,m,e,f] <ma||ef>  ->  [i,a]   (T1 contribution).'''
    if eris.E4 is None:
        return einsum('imef,maef->ia', t2, np.asarray(eris.ovvv))
    Co, Cv, Cf = _spin_mo(eris)
    E = eris.E4
    res = 0.
    for s1 in (0, 1):
        for s2 in (0, 1):
            te = einsum('ne,imef->imnf', Cv[s1], t2)
            tef = einsum('imnf,sf->imns', te, Cv[s2])
            D = einsum('um,imns->iuns', Co[s1].conj(), tef)
            G = einsum('iuns,unls->il', D, E)
            res = res + 2.0*einsum('il,la->ia', G, Cv[s2].conj())
    return res


def _ovvv_t2(t1, eris):
    '''sum_e t1[i,e] <je||ba>.conj()  ->  [i,j,a,b]   (T2 contribution).'''
    if eris.E4 is None:
        return einsum('ie,jeba->ijab', t1, np.asarray(eris.ovvv).conj())
    Co, Cv, Cf = _spin_mo(eris)
    E = eris.E4
    A5 = 0.
    for s2 in (0, 1):
        dt = einsum('ie,le->il', t1, Cv[s2])             # [i,lambda]
        Ei = einsum('unls,il->unis', E, dt)              # lambda -> i  (occ first)
        R = einsum('unis,sa->unia', Ei, Cv[s2].conj())   # [u,n,i,a]
        for s1 in (0, 1):
            R1 = einsum('uj,unia->jnia', Co[s1], R)      # u->j
            A5 = A5 + einsum('nb,jnia->ijab', Cv[s1].conj(), R1)  # n->b
    return A5 - A5.transpose(0, 1, 3, 2)


def _cc_Fvv(t1, t2, eris):
    '''cc_Fvv with the ovvv term evaluated AO-direct (see :func:`_ovvv_fvv`).'''
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc, nocc:]
    fvv = eris.fock[nocc:, nocc:]
    tau_tilde = imd.make_tau(t2, t1, t1, fac=0.5)
    Fae = fvv - 0.5*einsum('me,ma->ae', fov, t1)
    Fae += _ovvv_fvv(t1, eris)
    Fae -= 0.5*einsum('mnaf,mnef->ae', tau_tilde, eris.oovv)
    return Fae


def _cc_Wovvo(t1, t2, eris):
    '''cc_Wovvo with the ovvv term evaluated AO-direct (see :func:`_ovvv_wovvo`).'''
    eris_ovvo = -np.asarray(eris.ovov).transpose(0, 1, 3, 2)
    eris_oovo = -np.asarray(eris.ooov).transpose(0, 1, 3, 2)
    Wmbej = _ovvv_wovvo(t1, eris)
    Wmbej -= einsum('nb,mnej->mbej', t1, eris_oovo)
    Wmbej -= 0.5*einsum('jnfb,mnef->mbej', t2, eris.oovv)
    Wmbej -= einsum('jf,nb,mnef->mbej', t1, t1, eris.oovv)
    Wmbej += eris_ovvo
    return Wmbej


def _ladder_lc(tau, eris):
    '''Particle-particle ladder ``0.5 sum_cd <ab||cd> tau_ij^cd`` evaluated
    AO-direct with the Liu-Cheng spin-block decomposition (J. Chem. Phys. 148,
    034106 (2018)).

    The same-spin (alpha-alpha, beta-beta) contributions use the antisymmetrized
    same-spin AO integral ``<mu nu||si rho>`` packed on the (mu>nu, si>rho)
    triangles -- a packed intermediate that cuts the rate limiting contraction by
    ~4.  The result is then *unpacked* and back-transformed with the usual two
    quarter transforms (which are cheaper unpacked).  The opposite-spin block has
    no permutational antisymmetry and is evaluated once, in full, reusing
    :func:`_ao_contract`.'''
    Co, Cv, Cf = _spin_mo(eris)
    nocc = eris.nocc
    nao = eris.nao_sph
    nvir = Cv[0].shape[1]
    nij = nocc*nocc
    pi, pj = eris.ao_tril
    W = eris.W_same
    res = np.zeros((nocc, nocc, nvir, nvir), dtype=tau.dtype)
    for s in (0, 1):                          # same-spin aa, bb
        T = einsum('se,ijef,rf->ijsr', Cv[s], tau, Cv[s]).reshape(nij, nao, nao)
        thatp = 2.0 * lib.dot(T[:, pi, pj], W)            # packed N^6 contraction
        that = np.zeros((nij, nao, nao), dtype=tau.dtype)
        that[:, pi, pj] = thatp
        that[:, pj, pi] = -thatp                          # unpack (antisymmetric)
        that = that.reshape(nocc, nocc, nao, nao)
        res += 0.5*einsum('ma,ijml,lb->ijab', Cv[s].conj(), that, Cv[s].conj())
    # opposite spin (alpha-beta), one full contraction + P(ab) antisymmetrization
    Tab = einsum('se,ijef,rf->ijsr', Cv[0], tau, Cv[1])
    that_ab = _ao_contract(eris, Tab)
    Iab = einsum('ma,ijmn,nb->ijab', Cv[0].conj(), that_ab, Cv[1].conj())
    res += Iab - Iab.transpose(0, 1, 3, 2)
    return res


def update_t2_vvvv_direct(t1, t2, tau, eris):
    '''AO-direct analogue of :func:`gintermediates.update_t2_vvvv`.

    The ladder ``sum_ef <ab||ef> tau_ij^ef`` is built in the scalar AO basis
    so the spinor ``vvvv`` tensor is never required.  The remaining ``ovvv``
    and ``oovv`` contributions are identical to the in-core routine.
    '''
    nocc, nvir = t1.shape
    t2_new = np.zeros(t2.shape, dtype=t2.dtype)

    if eris.W_same is not None:
        # Liu-Cheng spin-block ladder with antisymmetric AO packing.
        t2_new += 2.0 * _ladder_lc(tau, eris)
    else:
        # Plain spin-summed AO-direct ladder (e.g. chol mode), blocked over the
        # first occupied index to bound peak memory.
        Cv = _ao_spin_blocks(eris)
        nao = eris.nao_sph
        blk = max(1, int(2000*1e6/16/max(1, nocc*nao*nao)))
        for i0, i1 in prange(0, nocc, blk):
            taui = tau[i0:i1]
            ladder = np.zeros((i1-i0, nocc, nvir, nvir), dtype=t2.dtype)
            for C1 in Cv:
                for C2 in Cv:
                    T = einsum('ne,ijef,sf->ijns', C1, taui, C2)
                    G = _ao_contract(eris, T)
                    ladder += einsum('ma,ijml,lb->ijab', C1.conj(), G, C2.conj())
            t2_new[i0:i1] += 2.0 * ladder

    tmp = _ovvv_ladder(tau, eris)
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

    Fvv = _cc_Fvv(t1, t2, eris)
    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)
    Woooo = imd.cc_Woooo(t1, t2, eris)
    Wovvo = _cc_Wovvo(t1, t2, eris)

    Fvv[np.diag_indices(nvir)] -= mo_e_v
    Foo[np.diag_indices(nocc)] -= mo_e_o

    # T1 equation
    t1new = einsum('ie,ae->ia', t1, Fvv)
    t1new += -einsum('ma,mi->ia', t1, Foo)
    t1new += einsum('imae,me->ia', t2, Fov)
    t1new += -einsum('nf,naif->ia', t1, eris.ovov)
    t1new += -0.5*_ovvv_t1(t2, eris)
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
    tmp = _ovvv_t2(t1, eris)
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
        '''CCSD(T) correction.

        (T) contracts the connected <bc||ei> slice with a small t2 block for
        every (b,c) virtual pair, so its integrals enter slice by slice rather
        than as one batched contraction.  That access pattern cannot be made
        AO-direct efficiently: rebuilding each <bc||ei> from the AO integrals
        costs O(nao^4) per (b,c), i.e. O(v^2 nao^4) overall, which exceeds the
        O(o^3 v^4) (T) work itself.  (T) never needs vvvv though -- only ovvv
        (O(o v^3)) -- so here ovvv is simply reconstructed once from the AO
        integrals.  For a genuinely storage-free (T), use the Cholesky path
        (chol_zccsd_t), which rebuilds each slice cheaply from low-rank factors.
        '''
        from socutils.cc import gccsd_t
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None:
            eris = self.eris if self.eris is not None else self.ao2mo()
        if eris.ovvv is None:
            from socutils.cc.eom_zccsd_direct import get_ovvv
            eris.ovvv = get_ovvv(eris)
        return gccsd_t.kernel(self, eris, t1, t2, self.verbose)

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
        # AO-direct ladder / ovvv data
        self.mo_ao_spin = None
        self.nao_sph = None
        self.ao_eri = None
        self.ao_chol = None
        self.E4 = None              # 4-index scalar AO ERIs (exact 'ao' mode)
        self.W_same = None          # packed antisymmetric same-spin AO integral <mu nu||si rho>
        self.ao_tril = None         # (mu>nu) pair indices
        self._spin_mo_cache = None

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
        # Exact integrals, AO-direct ovvv: only blocks with at most two virtual
        # indices are transformed (oooo, ooov, oovv, ovov, ovvo).  Neither vvvv
        # nor ovvv is ever formed -- the ovvv terms are evaluated AO-direct from
        # eris.E4 during the iterations.
        E = mol.intor('int2e_sph').reshape(nao, nao, nao, nao)
        eris.E4 = E
        # ladder operator, [(nu,sigma),(mu,lambda)] layout for a single GEMM
        eris.ao_eri = E.transpose(1, 3, 0, 2).reshape(nao*nao, nao*nao).copy()
        eris.ao_chol = None
        eris.ovvv = None

        Ca, Cb = mog[:nao], mog[nao:]
        Co = [Ca[:, :nocc], Cb[:, :nocc]]
        Cv = [Ca[:, nocc:], Cb[:, nocc:]]
        Cf = [Ca, Cb]
        # Shared electron-1 half transform: H[i,q,lambda,sigma], i occ, q full.
        H = 0.
        for s in (0, 1):
            t = einsum('mi,mnls->inls', Co[s].conj(), E)
            H = H + einsum('inls,nq->iqls', t, Cf[s])

        def e2(Hx, Cr, Cs):
            '''Finish the electron-2 transform of Hx onto the (Cr, Cs) ket.'''
            out = 0.
            for s in (0, 1):
                u = einsum('iqls,lr->iqrs', Hx, Cr[s].conj())
                out = out + einsum('iqrs,st->iqrt', u, Cs[s])
            return out

        no = nocc
        g_qoo = e2(H, Co, Co)              # (i, q, o, o)
        g_qov = e2(H, Co, Cv)             # (i, q, o, v)
        g_qvo = e2(H, Cv, Co)             # (i, q, v, o)
        g_ovv = e2(H[:, :no], Cv, Cv)     # (i, o, v, v)  (q restricted to occ)
        H = None
        eris.oooo = g_qoo[:, :no].transpose(0,2,1,3) - g_qoo[:, :no].transpose(0,2,3,1)
        eris.ooov = g_qov[:, :no].transpose(0,2,1,3) - g_qoo[:, no:].transpose(0,2,3,1)
        eris.oovv = g_qov[:, no:].transpose(0,2,1,3) - g_qov[:, no:].transpose(0,2,3,1)
        eris.ovov = g_ovv.transpose(0,2,1,3) - g_qvo[:, no:].transpose(0,2,3,1)
        eris.ovvo = g_qvo[:, no:].transpose(0,2,1,3) - g_ovv.transpose(0,2,3,1)

        # Packed antisymmetrized same-spin AO integral <mu nu||si rho> on the
        # (mu>nu, si>rho) triangles, for the Liu-Cheng particle-particle ladder.
        # Amplitude independent -> built once.
        pi, pj = np.tril_indices(nao, -1)
        eris.ao_tril = (pi, pj)
        eris.W_same = (E[pi[:, None], pi[None, :], pj[:, None], pj[None, :]] -
                       E[pi[:, None], pj[None, :], pj[:, None], pi[None, :]])

    log.timer('CCSD integral transformation (vvvv/ovvv-free)', *cput0)
    return eris


def _eris_from_cholmo(eris, chol_mo, nocc):
    '''Build the antisymmetrized occupied-row blocks from the Cholesky factors
    in the MO basis (mirrors :func:`chol_zccsd._make_df_eris`).'''
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
