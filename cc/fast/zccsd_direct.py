#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
# Copied from socutils (~/apps/socutils-git, cc/zccsd_direct.py at ffbe691) by the same
# author, as the starting point of the spinor CCSD(T) of zccsdt.  Changed:
# the imports of socutils.cc, get_ovvv (for (T)) moved here from
# eom_zccsd_direct.py, the EOM entry points left out.
# Changed since: the integrals of ladder='ao' come from socutils.cc.fast.ao2mo
# (slabs of the first occupied index, their ket pass that of socutils'
# nrr_fast, within max_memory: what does not fit is mapped from a scratch file) and
# ovvv is stored by default (DirectZCCSD.ovvv_direct = True: evaluated from the
# AO integrals in every term, as in socutils).  The amplitude equations are
# those of zccsdt.zccsd.amps (packed pairs, fixed layouts, the same
# equations); DirectZCCSD.socutils_amps = True runs update_amps below.
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
from pyscf.ao2mo import restore as ao2mo_restore
from pyscf import __config__

from socutils.cc import gintermediates as imd     # identical to socutils' copy
from socutils.cc.fast import zccsd as _zccsd

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
    ovvv_direct : bool
        ``ladder='ao'`` only.  False (default): <ma||ef> is transformed and
        stored (o v^3).  True: it is never built and its five terms are
        evaluated from the AO integrals in every iteration, as in socutils.
    socutils_amps : bool
        False (default, ``ladder='ao'`` with stored <ma||ef>): the amplitude
        equations and the iterations of zccsdt.zccsd.amps (packed T2, sliced
        within max_memory).  True: update_amps of this module, as copied
        from socutils (einsums on the full arrays), on PySCF's driver.
    max_disk : float or None
        MB for the scratch files of the process.  Default None: the
        environment variable ZCCSDT_MAX_DISK, else the free space of the
        scratch directory less a reserve.  ``diis_space`` is a maximum:
        where the memory plan has the DIIS vectors in the scratch file and
        the disk does not hold them, fewer are kept (at least 3, stated in
        the log; DiskFullError before the iterations below that).  The
        other scratch files are checked against the free space when they
        are created (zccsdt.scratch, zccsdt.disk).
    three_m : bool or None
        Three real GEMMs for a complex product instead of four.  None (the
        default): in zccsdt.zccsd.amps for the two o^3 v^3 products when
        they are large and the memory is there, in (T) when the real parts
        of its operands and a whole batch fit in memory; False: ZGEMM.
    kramers : bool or None
        zccsdt.zccsd.amps / triples only.  None (the default): if the
        reference is time-reversal symmetric (a closed shell without a
        magnetic field), the spinors are rotated within their degenerate
        levels into Kramers pairs (zccsdt.zccsd.kramers), ordered
        [unbarred, barred] among the occupied and among the virtual ones,
        and the symmetry t[Kp Kq, Kr Ks] = +-t[pq, rs]* is used: (T) does
        half of the virtual triples.  The amplitudes t1 and t2 then refer to
        these spinors, eris.mo_coeff, not to mf.mo_coeff.  False: the
        spinors of the reference as they are.  True: as None, but an error
        if the reference is not symmetric.
    ao_ladder : str or None
        zccsdt.zccsd.amps only: where the two AO operators of the ladder
        (2 N^4 bytes each as square matrices for N AO functions, N^4 as
        triangles) come from.  'memory': kept; 'direct': made from the AO
        integrals in every iteration (integral-direct, about N^4/8
        integrals for every block of pairs); 'file': kept in a scratch
        file.  None (the default): 'memory' if they fit into the share of
        max_memory for stored integrals and the CCSD step fits next to
        them, else 'direct'.
    '''

    ovvv_direct = False
    socutils_amps = False
    three_m = None
    kramers = None
    ao_ladder = None
    max_disk = None

    _keys = {'max_disk', 'ladder', 'cderi', 'max_error', 'eris', 'ovvv_direct', 'socutils_amps', 'three_m',
             'kramers', 'ao_ladder', 'with_mmf', 'feri', 'ao2mo_outcore', '_t_density', 'e_t', '_cc2', 'e_cc2', 't1_cc2',
             '_l_with_t', '_t_gradient'}

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None,
                 ladder='ao', cderi=None, max_error=1e-5):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.ladder = ladder
        self.cderi = cderi
        self.max_error = max_error
        self.eris = None

    def _fixed_layouts(self):
        '''Do the amplitude equations run in zccsdt.zccsd.amps?'''
        return self.ladder == 'ao' and not (self.socutils_amps or self.ovvv_direct)

    def update_amps(self, t1, t2, eris):
        if not self._fixed_layouts():
            return update_amps(self, t1, t2, eris)
        if eris.residual is None:
            from socutils.cc.fast import amps
            eris.residual = amps.Residual(eris, self.max_memory, self.diis_space,
                                          three_m=self.three_m)
        return eris.residual.update(t1, t2, self.level_shift)

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=None):
        if mbpt2 or not self._fixed_layouts():
            return _zccsd.ZCCSD.ccsd(self, t1, t2, eris, mbpt2)
        # the iterations of zccsdt.zccsd.amps: T2 packed throughout
        from socutils.cc.fast import amps
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()
        self.e_hf = self.get_e_hf()
        if eris is None:
            eris = self.eris if self.eris is not None else self.ao2mo(self.mo_coeff)
        self._drop_t_terms()                # those of other amplitudes
        self.converged, self.e_corr, self.t1, self.t2 = amps.kernel(
            self, eris, t1, t2, max_cycle=self.max_cycle, tol=self.conv_tol,
            tolnormt=self.conv_tol_normt, verbose=self.verbose, callback=self.callback)
        self._finalize()
        return self.e_corr, self.t1, self.t2

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None, with_t=False):
        '''The Lambda equations for the amplitudes t1, t2 (default: those of
        the last CCSD): self.l1[i,a] and self.l2[i,j,a,b] as in PySCF's
        GCCSD, self.converged_lambda.  With zccsdt.zccsd.amps they are solved
        in zccsdt.zccsd.lam, on the integrals and within the memory plan of
        the CCSD; like t1 and t2 they refer to the spinors eris.mo_coeff.
        ``with_t``: for CCSD(T), with the (T) terms of the equations
        (zccsdt.zccsd.lam_t) from the batches of the (T) code
        (zccsdt.zccsd.triples.kernel with terms=True: about twice the time
        of the (T) energy); self.e_t is set on the way.  ``with_t='gradient'``:
        the densities of (T) for the orbital gradient are made in the same
        pass and kept for the relaxed density (make_rdm1 with relaxed=True
        asks for them; zccsdt.zccsd.zvector releases them).'''
        if not self._fixed_layouts():
            return _zccsd.ZCCSD.solve_lambda(self, t1, t2, l1, l2, eris)
        raise NotImplementedError('lam is not in socutils yet; use zccsdt.zccsd for it')
        if eris is None:
            eris = self.eris if self.eris is not None else self.ao2mo(self.mo_coeff)
        self._drop_t_terms()
        gradient = isinstance(with_t, str) and with_t == 'gradient'

        def t_terms(t1, t2p):
            # the (T) terms of lambda over their denominators; those of the
            # density are kept for make_rdm1
            from socutils.cc.fast import kernels as K
            terms, dens = self.t_terms(eris, t1, t2p, gradient)
            eo, ev = (np.ascontiguousarray(x, dtype=np.float64)
                      for x in (eris.mo_energy[:eris.nocc], eris.mo_energy[eris.nocc:]))
            l1t, l2t = terms[0] / (eo[:, None] - ev), K.div_packed(terms[1], eo, ev)
            if gradient:
                # the sources as the Lambda equations hold them, and the densities
                self._t_gradient = (l1t, l2t, dens)
            return l1t, l2t

        self.converged_lambda, self.l1, self.l2 = lam.kernel(
            self, eris, t1, t2, l1, l2, max_cycle=self.max_cycle, tol=self.conv_tol_normt,
            verbose=self.verbose, sources=t_terms if with_t else None)
        self._l_with_t = bool(with_t)       # which equations self.l1 and self.l2 solve
        return self.l1, self.l2

    def t_terms(self, eris, t1, t2p, gradient=False):
        '''((s1, s2p, goo, gvv, dvo), Densities or None): the (T) terms of
        the Lambda equations (before the division by their denominators) and
        of the one-particle density, with ``gradient`` the densities of the
        orbital gradient as well, from socutils.cc.fast.triples.kernel; self.e_t
        is set, and the terms of the density are kept for make_rdm1.'''
        from socutils.cc.fast import triples
        if getattr(eris, 'residual', None) is not None:
            eris.residual.close()               # as in ccsd_t: W(mbej) and the AO
        if getattr(eris, 'aoladder', None) is not None:
            eris.aoladder.close()               # operators are made again for the steps
        out = triples.kernel(self, eris, t1, t2p, self.verbose, three_m=self.three_m,
                             terms='gradient' if gradient else True)
        self.e_t, terms = out[:2]
        self._t_density = terms[2:]
        return terms, (out[2] if gradient else None)

    def _drop_t_terms(self):
        '''Forget the (T) terms of earlier amplitudes; the scratch file of
        their densities is released.'''
        kept = getattr(self, '_t_gradient', None)
        if kept is not None:
            kept[2].close()
        self._t_gradient = self._t_density = None

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, eris=None, ao_repr=False,
                  with_t=False, relaxed=False):
        '''The one-particle density of t and lambda (zccsdt.zccsd.rdm): in
        the basis of the spinors eris.mo_coeff, or in the 2c AO basis.'''
        raise NotImplementedError('rdm is not in socutils yet; use zccsdt.zccsd for it')
        return rdm.make_rdm1(self, t1, t2, l1, l2, eris, ao_repr, with_t, relaxed)

    def dip_moment(self, t1=None, t2=None, l1=None, l2=None, eris=None, with_t=False,
                   relaxed=False, unit='Debye', picture_change=True):
        '''The electric dipole moment from make_rdm1 (zccsdt.zccsd.rdm), the
        operator with the picture-change transformation of X2C.'''
        raise NotImplementedError('rdm is not in socutils yet; use zccsdt.zccsd for it')
        return rdm.dip_moment(self, t1, t2, l1, l2, eris, with_t, relaxed, unit, picture_change)

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        self.eris = _make_eris_direct(self, mo_coeff)
        return self.eris

    def ccsd_t(self, t1=None, t2=None, eris=None):
        '''CCSD(T) correction.

        With the amplitude equations of zccsdt.zccsd.amps: zccsdt.zccsd.triples
        (batches of virtual pairs on the packed <ie||bc>, within max_memory).

        Else the loops copied from socutils (gccsd_t).  (T) contracts the
        connected <bc||ei> slice with a small t2 block for every (b,c)
        virtual pair, so its integrals enter slice by slice rather than as
        one batched contraction.  That access pattern cannot be made
        AO-direct efficiently: rebuilding each <bc||ei> from the AO integrals
        costs O(nao^4) per (b,c), i.e. O(v^2 nao^4) overall, which exceeds the
        O(o^3 v^4) (T) work itself.  (T) never needs vvvv though -- only ovvv
        (O(o v^3)) -- so here ovvv is simply reconstructed once from the AO
        integrals.  For a genuinely storage-free (T), use the Cholesky path
        (chol_zccsd_t), which rebuilds each slice cheaply from low-rank factors.
        '''
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None:
            eris = self.eris if self.eris is not None else self.ao2mo()
        if self._fixed_layouts():
            from socutils.cc.fast import triples
            if getattr(eris, 'residual', None) is not None:
                eris.residual.close()               # the layouts of the CCSD step are
                eris.residual = None                # made again when they are needed
            if getattr(eris, 'aoladder', None) is not None:
                eris.aoladder.close()               # (T) has no ladder: so are its operators
            return triples.kernel(self, eris, t1, t2, self.verbose, three_m=self.three_m)
        from socutils.cc.fast import gccsd_t
        if eris.ovvv is None:
            eris.ovvv = get_ovvv(eris)              # from the packed array or the AO integrals
        return gccsd_t.kernel(self, eris, t1, t2, self.verbose)

    # ---- CC2 on the Cholesky vectors (zccsdt.zccsd.cc2; docs/ccn.md, section 8)

    def cc2_engine(self):
        '''The zccsdt.zccsd.cc2.Engine of this calculation: the active
        spinors, their Fock matrix and the Cholesky vectors of the scalar AO
        integrals - ``cderi`` if given, else decomposed to ``max_error``
        (``ladder = 'ao'``: to 1e-10, the exact integrals to that; there is
        no integral-direct CC2).  No integral of the spinors is made.'''
        raise NotImplementedError('cc2 is not in socutils yet; use zccsdt.zccsd for it')
        eris = _PhysicistsERIs()
        eris._common_init_(self, self.mo_coeff)
        mol = self.mol
        nao = mol.nao_nr()
        mog = np.vstack(mol.sph2spinor_coeff()).dot(eris.mo_coeff)
        cderi = self.cderi
        if cderi is None:
            from socutils.mcscf import zmc_ao2mo
            err = self.max_error if self.ladder == 'chol' else 1e-10
            import contextlib
            import io
            buf = io.StringIO()         # (the decomposition prints its progress whatever ``verbose``)
            with contextlib.redirect_stdout(buf):
                cderi = zmc_ao2mo.chunked_cholesky(mol, max_error=err, verbose=self.verbose > 4,
                                                   cmax=8 if self.ladder == 'chol' else max(8, (nao + 1) // 2 + 1))
            if self.verbose > 4:
                self.stdout.write(buf.getvalue())
        L = np.asarray(cderi).reshape(-1, nao, nao)
        self._cc2 = cc2.Engine(mog, eris.nocc, eris.fock, L, eris.e_hf, self.max_memory, self.verbose, self.stdout)
        logger.info(self, 'CC2 on Cholesky vectors: %s', self._cc2.describe())
        return self._cc2

    def cc2(self, t1=None):
        '''The CC2 ground state on the Cholesky vectors: (E_corr, t1); the
        doubles are never stored.  self.e_cc2, self.t1_cc2;
        conv_tol, conv_tol_normt, max_cycle, diis_space as for CCSD.'''
        eng = getattr(self, '_cc2', None) or self.cc2_engine()
        self.e_cc2, self.t1_cc2 = eng.kernel(t1, self.conv_tol, self.conv_tol_normt, self.max_cycle, self.diis_space)
        logger.note(self, 'E(CC2) = %.15g  E_corr = %.15g  (%d iterations%s)', eng.e_hf + self.e_cc2, self.e_cc2,
                    eng.cycles, '' if eng.converged else ', NOT converged')
        return self.e_cc2, self.t1_cc2

    def cc2_ee(self, nroots=3, conv_tol=1e-6, **kwargs):
        '''The ``nroots`` lowest excitation energies of CC2 (linear response;
        folded doubles, each root self-consistent in its frequency) after
        cc2(): (energies in Hartree, the Solved of zccsdt.cc.ccn.folded_roots).'''
        eng = getattr(self, '_cc2', None)
        if eng is None or eng.Y0 is None:
            self.cc2()
            eng = self._cc2
        return eng.eom_ee(nroots, conv_tol, **kwargs)

    # (the vvvv-free EOM-CCSD entry points of socutils are not copied: EOM
    # stays in socutils.cc.eom_zccsd_direct)


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
        # zccsdt.zccsd.amps
        self.kramers = False        # the spinors are Kramers pairs, [unbarred, barred]
        self.ovvv_p = None          # <ma||ef> packed over e > f
        self.oovv_p = None          # <mn||ef> packed over m > n and e > f
        self.oovv_r = None          # <mn||ef> as [m, e, n, f]
        self.ovov_r = None          # -<mb||je> as [m, e, j, b]
        self.aoladder = None        # zccsd.aoladder.AOLadder: both AO operators (amps)
        self.residual = None        # amps.Residual of these integrals

    # with the layouts of zccsdt.zccsd.amps, oovv and ovov exist only on
    # demand (for (T) and the functions of PySCF on the full arrays)
    @property
    def oovv(self):
        if self._oovv is None and self.oovv_r is not None:
            self._oovv = np.ascontiguousarray(np.asarray(self.oovv_r).transpose(0, 2, 1, 3))
        return self._oovv

    @oovv.setter
    def oovv(self, x):
        self._oovv = x

    @property
    def ovov(self):
        if self._ovov is None and self.ovov_r is not None:
            self._ovov = -np.asarray(self.ovov_r).transpose(0, 3, 2, 1)
        return self._ovov

    @ovov.setter
    def ovov(self, x):
        self._ovov = x

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
        # Exact integrals from the scalar AO integrals (zccsdt.zccsd.ao2mo): the
        # blocks with a first occupied index, slab by slab; vvvv is never
        # formed.  ovvv is stored unless ovvv_direct, in which case its terms
        # are evaluated AO-direct from eris.E4 during the iterations.  What
        # does not fit mycc.max_memory is a mapped array of a scratch file.
        from socutils.cc.fast import ao2mo as slab
        if mycc._fixed_layouts() and mycc.kramers is not False:
            mog = _kramers_pairs(mycc, eris, log)
        Ca, Cb = mog[:nao], mog[nao:]
        if mycc._fixed_layouts():       # zccsdt.zccsd.amps: its layouts, no ovvo
            names = ('oooo', 'ooov', 'oovv_p', 'oovv_r', 'ovov_r', 'ovvv_p')
        else:
            names = slab.BLOCKS[:5 + (not mycc.ovvv_direct)]
        blocks = slab.occupied_blocks(mol, Ca, Cb, nocc, max_memory=mycc.max_memory, names=names)
        eris.oooo, eris.ooov, eris.oovv = blocks['oooo'], blocks['ooov'], blocks.get('oovv')
        eris.ovov, eris.ovvo = blocks.get('ovov'), blocks.get('ovvo')
        eris.ovvv, eris.ovvv_p = blocks.get('ovvv'), blocks.get('ovvv_p')
        eris.oovv_p, eris.oovv_r, eris.ovov_r = (blocks.get(k) for k in
                                                 ('oovv_p', 'oovv_r', 'ovov_r'))
        eris.ao_chol = None
        log.info('MO integrals %s', blocks.describe())
        # the AO operators of the ladder (amplitude independent): the packed
        # antisymmetrized same-spin integral <mu nu||si rho> on the (mu>nu,
        # si>rho) triangles (Liu-Cheng ladder) and, for the alpha-beta block,
        # the packed symmetrized integral (amps) or the full matrix
        # [(nu,sigma),(mu,lambda)] (socutils)
        if mycc._fixed_layouts():
            # zccsdt.zccsd.aoladder: both operators in memory if they fit (as the
            # square matrices, or in thinner parts of their rows down to the
            # triangles), else made from the AO integrals in every iteration
            from socutils.cc.fast import memory
            from socutils.cc.fast import aoladder
            budget = memory.budget(mycc.max_memory)
            floor = aoladder.unit_floor(mol)
            nb = aoladder.blocks_that_fit(mol, slab._store_bytes(budget, floor))
            mode = mycc.ao_ladder
            if mode is None:
                mode = 'direct' if nb is None else 'memory'
            elif mode not in aoladder.MODES:
                raise ValueError('DirectZCCSD.ao_ladder = %r: None or one of %s'
                                 % (mode, ', '.join(aoladder.MODES)))
            if nb is None or mode == 'file':
                nb = aoladder.most_blocks(mol)
            work = None
            if budget is not None:      # the units while the kept rows are made
                kept = aoladder.stored_bytes(mol, nb) if mode == 'memory' else 0
                work = max(floor, (budget - kept) // 2)
            eris.aoladder = aoladder.AOLadder(mol, mode, work, nb, auto=mycc.ao_ladder is None)
            log.info('AO operators of the ladder %s', eris.aoladder.describe())
            eris._scratch = (blocks.scratch, eris.aoladder)
        else:
            aops = slab.ladder_operators(mol, mycc.max_memory, names=('ao_eri', 'W_same'))
            eris.ao_eri, eris.W_same = aops.get('ao_eri'), aops['W_same']
            eris.ao_tril = aops.idx
            log.info('AO operators of the ladder %s', aops.describe())
            eris._scratch = (blocks.scratch, aops.scratch)  # keeps the scratch files open
        # 4-index AO integrals, in memory: only for the AO-direct ovvv terms
        eris.E4 = (ao2mo_restore(1, mol.intor('int2e', aosym='s8'), nao)
                   if mycc.ovvv_direct else None)

    log.timer('CCSD integral transformation (vvvv/ovvv-free)', *cput0)
    return eris


def _kramers_pairs(mycc, eris, log):
    '''Rotate the spinors of ``eris`` within their degenerate levels into
    Kramers pairs, [unbarred, barred] among the occupied and among the
    virtual ones, if the reference is time-reversal symmetric; the
    coefficients in the (spin, AO) basis that the integrals are made from.'''
    from socutils.cc.fast import kramers
    nocc, mog, e = eris.nocc, eris.mo_ao_spin, eris.mo_energy
    s2 = np.kron(np.eye(2), mycc.mol.intor('int1e_ovlp'))
    occ = kramers.pair(mog[:, :nocc], s2, e[:nocc])
    vir = kramers.pair(mog[:, nocc:], s2, e[nocc:])
    if occ is None or vir is None:
        if mycc.kramers:
            raise RuntimeError('DirectZCCSD.kramers = True, but the spinors are not closed '
                               'under time reversal level by level')
        log.info('No Kramers pairs: the spinors are not degenerate in pairs that are closed '
                 'under time reversal (an open shell, a magnetic field, or an SCF that is '
                 'not converged: the energies of partners must agree to 1e-7)')
        return mog
    paired = np.hstack((occ[0], vir[0]))
    u = mog.conj().T @ s2 @ paired                  # block diagonal within the levels
    eris.mo_ao_spin = paired
    eris.mo_coeff = eris.mo_coeff @ u
    eris.fock = u.conj().T @ eris.fock @ u
    eris.mo_energy = eris.fock.diagonal().real
    eris.kramers = True
    off = eris.fock - np.diag(eris.fock.diagonal())
    log.info('Spinors rotated into Kramers pairs, [unbarred, barred] (largest off-diagonal '
             'Fock element %.1e)', abs(off).max())
    return paired


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


# get_ovvv: copied from socutils cc/eom_zccsd_direct.py (the EOM code itself is
# not copied), the module prefix zd. dropped.
def get_ovvv(eris):
    '''Reconstruct ``<ia||bc>`` once from the scalar AO integrals (O(o v^3)).'''
    if eris.ovvv is not None:
        return np.asarray(eris.ovvv)
    if getattr(eris, 'ovvv_p', None) is not None:      # zccsdt: from the packed array
        p = np.asarray(eris.ovvv_p)
        nocc, nvir = p.shape[:2]
        b, c = np.tril_indices(nvir, -1)
        out = np.zeros((nocc, nvir, nvir, nvir), dtype=p.dtype)
        out[:, :, b, c] = p
        out[:, :, c, b] = -p
        return out
    Co, Cv, Cf = _spin_mo(eris)
    E = eris.E4
    out = 0.
    for s1 in (0, 1):                                  # chemist (i b | a c)
        H = einsum('mi,nb,mnls->ibls', Co[s1].conj(), Cv[s1], E)
        for s2 in (0, 1):
            out = out + einsum('ibls,la,sc->ibac', H, Cv[s2].conj(), Cv[s2])
    return out.transpose(0, 2, 1, 3) - out.transpose(0, 2, 3, 1)

if __name__ == '__main__':
    from pyscf import gto
    from socutils.scf import spinor_hf
    from socutils.cc.fast.zccsd import ZCCSD

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
