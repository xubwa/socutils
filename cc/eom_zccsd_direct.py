#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
# EOM-CCSD (IP/EA/EE) on top of the vvvv/ovvv-free two-component CCSD
# (:mod:`socutils.cc.zccsd_direct`).
#
# The only places the full O(v^4) ``Wvvvv`` intermediate enters EOM are
#   (1) the ``Wvvvv . r2`` particle-particle ladder in the sigma vector and
#   (2) the ``Wvvvv`` diagonal used to build the Davidson preconditioner /
#       initial guess.
# Both are evaluated AO-direct here so ``Wvvvv`` is never stored:
#   (1) reuses the spin-block antisymmetric-packed ladder of zccsd_direct,
#       applied to the trial vector (or, for the ``Wvvvo . r1`` term, to the
#       antisymmetrized ``r1 (x) t1``);
#   (2) uses a cheap approximate diagonal (the bare ``<ab||ab>``, O(v^2)),
#       which is all the initial guess / preconditioner needs.
# ``ovvv`` (O(o v^3)) is reconstructed once from the AO integrals for the
# remaining intermediates.
#

import numpy as np
from pyscf import lib
from pyscf.lib import logger

from socutils.cc import eom_gccsd
from socutils.cc import gintermediates as imd
from socutils.cc import zccsd_direct as zd

einsum = lib.einsum


# ---------------------------------------------------------------------------
# AO-direct Wvvvv primitives
# ---------------------------------------------------------------------------

def get_ovvv(eris):
    '''Reconstruct ``<ia||bc>`` once from the scalar AO integrals (O(o v^3)).'''
    if eris.ovvv is not None:
        return np.asarray(eris.ovvv)
    Co, Cv, Cf = zd._spin_mo(eris)
    E = eris.E4
    out = 0.
    for s1 in (0, 1):                                  # chemist (i b | a c)
        H = einsum('mi,nb,mnls->ibls', Co[s1].conj(), Cv[s1], E)
        for s2 in (0, 1):
            out = out + einsum('ibls,la,sc->ibac', H, Cv[s2].conj(), Cv[s2])
    return out.transpose(0, 2, 1, 3) - out.transpose(0, 2, 3, 1)


def ladder_ao(X, eris):
    '''Bare particle-particle ladder ``0.5 sum_cd <ab||cd> X[B,cd]`` for a
    batched antisymmetric ``X`` (batch ``ij`` for EE, ``j`` for EA), via the
    Liu-Cheng spin-block antisymmetric packing.'''
    Co, Cv, Cf = zd._spin_mo(eris)
    nao = eris.nao_sph
    nvir = Cv[0].shape[1]
    pi, pj = eris.ao_tril
    W = eris.W_same
    nb = X.shape[0]
    res = np.zeros((nb, nvir, nvir), dtype=X.dtype)
    for s in (0, 1):                                   # same-spin, packed
        T = einsum('se,Bef,rf->Bsr', Cv[s], X, Cv[s]).reshape(nb, nao, nao)
        thatp = 2.0 * lib.dot(T[:, pi, pj], W)
        that = np.zeros((nb, nao, nao), dtype=X.dtype)
        that[:, pi, pj] = thatp
        that[:, pj, pi] = -thatp
        res += 0.5*einsum('ma,Bml,lb->Bab', Cv[s].conj(), that, Cv[s].conj())
    Tab = einsum('se,Bef,rf->Bsr', Cv[0], X, Cv[1])    # opposite spin
    Iab = einsum('ma,Bmn,nb->Bab', Cv[0].conj(), zd._ao_contract(eris, Tab), Cv[1].conj())
    return res + Iab - Iab.transpose(0, 2, 1)


def wvvvv_dot_x(X, t1, t2, eris, ovvv):
    '''Dressed ladder ``0.5 sum_ef Wvvvv[ab,ef] X[B,ef]`` (AO-direct), with
    ``Wvvvv = <ab||ef> - P(ab)(t1.ovvv) + 0.5 tau.oovv``.'''
    tau = imd.make_tau(t2, t1, t1)
    res = ladder_ao(X, eris)
    d1 = einsum('ma,mbef,Bef->Bab', t1, ovvv, X)
    d2 = einsum('mb,maef,Bef->Bab', t1, ovvv, X)
    res += 0.5*(-d1 + d2)
    oox = einsum('mnef,Bef->Bmn', np.asarray(eris.oovv), X)
    res += 0.25*einsum('mnab,Bmn->Bab', tau, oox)
    return res


def wvvvv_dot_xT(X, t1, t2, eris, ovvv):
    '''Transpose dressed ladder ``0.5 sum_ab Wvvvv[ab,cd] X[B,ab]`` -> [B,cd]
    (contracts the *first* virtual pair; used by the left EA sigma vector).'''
    tau = imd.make_tau(t2, t1, t1)
    res = ladder_ao(X.conj(), eris).conj()             # bare 0.5 sum_ab <ab||cd> X[ab]
    d1 = einsum('ma,mbcd,Bab->Bcd', t1, ovvv, X)
    d2 = einsum('mb,macd,Bab->Bcd', t1, ovvv, X)
    res += 0.5*(-d1 + d2)
    oox = einsum('mnab,Bab->Bmn', tau, X)
    res += 0.25*einsum('mncd,Bmn->Bcd', np.asarray(eris.oovv), oox)
    return res


def w4t1(t1, t2, eris, ovvv):
    '''``Wvvvv . t1`` over the second virtual index,
    ``sum_f Wvvvv[ab,ef] t1[i,f] -> [a,b,e,i]`` -- the vvvo piece of Wvvvo,
    built once AO-direct (O(o v^3)).'''
    Co, Cv, Cf = zd._spin_mo(eris)
    E = eris.E4
    oovv = np.asarray(eris.oovv)
    tau = imd.make_tau(t2, t1, t1)
    termA = 0.
    termB = 0.
    for s1 in (0, 1):
        for s2 in (0, 1):
            GA = einsum('mnls,is->mnli', E, t1 @ Cv[s2].T)
            termA = termA + einsum('ma,ne,lb,mnli->aebi',
                                   Cv[s1].conj(), Cv[s1], Cv[s2].conj(), GA)
            GB = einsum('mnls,lb,se->mnbe', E, Cv[s2].conj(), Cv[s2])
            termB = termB + einsum('ma,in,mnbe->aibe',
                                   Cv[s1].conj(), t1 @ Cv[s1].T, GB)
    res = termA.transpose(0, 2, 1, 3) - termB.transpose(0, 2, 3, 1)
    res += (-einsum('ma,mbef,if->abei', t1, ovvv, t1)
            + einsum('mb,maef,if->abei', t1, ovvv, t1)
            + 0.5*einsum('mnab,mnef,if->abei', tau, oovv, t1))
    return res


def vvvv_dot_v1(v1, eris):
    '''Bare ``sum_f <bc||ef> v1[f] -> [b,c,e]`` (v^3), for the EA *(star)
    contraction.  ``v1`` is a length-nvir vector.'''
    Co, Cv, Cf = zd._spin_mo(eris)
    E = eris.E4
    nv = Cv[0].shape[1]
    res = np.zeros((nv, nv, nv), dtype=np.result_type(v1, E.dtype))
    for s1 in (0, 1):
        for s2 in (0, 1):
            dt = Cv[s2].dot(v1)                         # [sigma]
            GA = einsum('mnls,s->mnl', E, dt)          # [mu,nu,lambda]
            tA = einsum('mnl,mb,ne,lc->bec', GA, Cv[s1].conj(), Cv[s1], Cv[s2].conj())
            res += tA.transpose(0, 2, 1)               # (be|cf): [b,e,c] -> [b,c,e]
            dtb = Cv[s1].dot(v1)                        # [nu]
            GB = einsum('mnls,n->mls', E, dtb)         # [mu,lambda,sigma]
            res -= einsum('mls,mb,lc,se->bce', GB, Cv[s1].conj(), Cv[s2].conj(), Cv[s2])
    return res


def wvvvv_diag(eris):
    '''Bare ``<ab||ab>`` for all (a,b), O(v^2) -- the approximate Wvvvv
    diagonal used for the preconditioner / initial guess.'''
    Co, Cv, Cf = zd._spin_mo(eris)
    E = eris.E4
    nao = eris.nao_sph
    nv = Cv[0].shape[1]
    Echem = E.reshape(nao*nao, nao*nao)
    Dtot = sum(einsum('ma,na->mna', Cv[s].conj(), Cv[s]) for s in (0, 1)).reshape(nao*nao, nv)
    JJ = lib.dot(Dtot.T, lib.dot(Echem, Dtot))         # (aa|bb)
    K = np.zeros((nv, nv), dtype=Dtot.dtype)
    for s1 in (0, 1):
        for s2 in (0, 1):
            G = einsum('nb,lb,mnls->msb', Cv[s1], Cv[s2].conj(), E)
            K += einsum('ma,sa,msb->ab', Cv[s1].conj(), Cv[s2], G)   # (ab|ba)
    return JJ - K


def _Wvvvo_partial(t1, t2, eris):
    '''imd.Wvvvo without the ``Wvvvv . t1`` term (which is folded into the
    sigma vector AO-direct).'''
    eris_ovvo = -np.asarray(eris.ovov).transpose(0, 1, 3, 2)
    eris_vvvo = -np.asarray(eris.ovvv).transpose(2, 3, 1, 0).conj()
    eris_oovo = -np.asarray(eris.ooov).transpose(0, 1, 3, 2)
    tmp1 = einsum('mbef,miaf->abei', eris.ovvv, t2)
    tmp2 = einsum('ma,mbei->abei', t1, eris_ovvo)
    tmp2 -= einsum('ma,nibf,mnef->abei', t1, t2, eris.oovv)
    FFov = imd.Fov(t1, t2, eris)
    tau = imd.make_tau(t2, t1, t1)
    Wabei = 0.5 * einsum('mnei,mnab->abei', eris_oovo, tau)
    Wabei -= einsum('me,miab->abei', FFov, t2)
    Wabei += eris_vvvo
    Wabei -= tmp1 - tmp1.transpose(1, 0, 2, 3)
    Wabei -= tmp2 - tmp2.transpose(1, 0, 2, 3)
    return Wabei


# ---------------------------------------------------------------------------
# Intermediates: everything except the stored Wvvvv
# ---------------------------------------------------------------------------

class _IMDS(eom_gccsd._IMDS):
    def __init__(self, cc, eris=None):
        eom_gccsd._IMDS.__init__(self, cc, eris)
        # ovvv is needed (once) for the non-vvvv intermediates and the dressing
        self.ovvv = get_ovvv(self.eris)
        if self.eris.ovvv is None:
            self.eris.ovvv = self.ovvv     # let the standard imd.* helpers see it

    def _make_wvvvo(self):
        t1, t2, eris = self.t1, self.t2, self.eris
        # full Wvvvo = (no-vvvv part) + Wvvvv.t1, the latter AO-direct (O(o v^3))
        return _Wvvvo_partial(t1, t2, eris) + w4t1(t1, t2, eris, self.ovvv)

    def make_ea(self):
        if not self._made_shared:
            self._make_shared()
        t1, t2, eris = self.t1, self.t2, self.eris
        self.Wvovv = imd.Wvovv(t1, t2, eris)
        self.Wvvvo = self._make_wvvvo()
        self.Wvvvv = None                            # never built (v^4)
        self.made_ea_imds = True
        return self

    def make_ee(self):
        if not self._made_shared:
            self._make_shared()
        t1, t2, eris = self.t1, self.t2, self.eris
        if not self.made_ip_imds:
            self.Woooo = imd.Woooo(t1, t2, eris)
            self.Wooov = imd.Wooov(t1, t2, eris)
            self.Wovoo = imd.Wovoo(t1, t2, eris)
        if not self.made_ea_imds:
            self.Wvovv = imd.Wvovv(t1, t2, eris)
            self.Wvvvo = self._make_wvvvo()
            self.Wvvvv = None
        self.made_ee_imds = True
        return self


# ---------------------------------------------------------------------------
# Sigma vectors (matvec): Wvvvv contractions replaced by AO-direct ladders
# ---------------------------------------------------------------------------

def eaccsd_matvec(eom, vector, imds=None, diag=None):
    if imds is None: imds = eom.make_imds()
    nocc, nmo = eom.nocc, eom.nmo
    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)
    t1, t2, eris, ovvv = imds.t1, imds.t2, imds.eris, imds.ovvv

    Hr1 = np.einsum('ac,c->a', imds.Fvv, r1)
    Hr1 += np.einsum('ld,lad->a', imds.Fov, r2)
    Hr1 += 0.5*np.einsum('alcd,lcd->a', imds.Wvovv, r2)

    Hr2 = np.einsum('abcj,c->jab', imds.Wvvvo, r1)    # full Wvvvo (incl Wvvvv.t1)
    tmp1 = lib.einsum('ac,jcb->jab', imds.Fvv, r2)
    Hr2 += tmp1 - tmp1.transpose(0, 2, 1)
    Hr2 -= lib.einsum('lj,lab->jab', imds.Foo, r2)
    tmp2 = lib.einsum('lbdj,lad->jab', imds.Wovvo, r2)
    Hr2 += tmp2 - tmp2.transpose(0, 2, 1)
    Hr2 += wvvvv_dot_x(r2, t1, t2, eris, ovvv)        # 0.5 Wvvvv.r2
    Hr2 -= 0.5*lib.einsum('klcd,lcd,kjab->jab', imds.Woovv, r2, imds.t2)
    return eom.amplitudes_to_vector(Hr1, Hr2)


def eaccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape
    abab = wvvvv_diag(imds.eris)                       # approximate <ab||ab>
    fvv = np.diagonal(imds.Fvv)
    foo = np.diagonal(imds.Foo)
    wov = einsum('jaaj->ja', imds.Wovvo)
    Hr1 = fvv.copy()
    Hr2 = (fvv[None, :, None] + fvv[None, None, :] - foo[:, None, None]
           + wov[:, None, :] + wov[:, :, None] + abab[None, :, :])
    return eom.amplitudes_to_vector(Hr1, Hr2)


def eeccsd_matvec(eom, vector, imds=None, diag=None):
    if imds is None: imds = eom.make_imds()
    nocc, nmo = eom.nocc, eom.nmo
    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)
    t1, t2, eris, ovvv = imds.t1, imds.t2, imds.eris, imds.ovvv

    Hr1 = lib.einsum('ae,ie->ia', imds.Fvv, r1)
    Hr1 -= lib.einsum('mi,ma->ia', imds.Foo, r1)
    Hr1 += lib.einsum('me,imae->ia', imds.Fov, r2)
    Hr1 += lib.einsum('maei,me->ia', imds.Wovvo, r1)
    Hr1 -= 0.5*lib.einsum('mnie,mnae->ia', imds.Wooov, r2)
    Hr1 += 0.5*lib.einsum('amef,imef->ia', imds.Wvovv, r2)

    tmpab = lib.einsum('be,ijae->ijab', imds.Fvv, r2)
    tmpab -= 0.5*lib.einsum('mnef,ijae,mnbf->ijab', imds.Woovv, imds.t2, r2)
    tmpab -= lib.einsum('mbij,ma->ijab', imds.Wovoo, r1)
    tmpab -= lib.einsum('amef,ijfb,me->ijab', imds.Wvovv, imds.t2, r1)
    tmpij = lib.einsum('mj,imab->ijab', -imds.Foo, r2)
    tmpij -= 0.5*lib.einsum('mnef,imab,jnef->ijab', imds.Woovv, imds.t2, r2)
    tmpij += lib.einsum('abej,ie->ijab', imds.Wvvvo, r1)    # full Wvvvo
    nvir = t1.shape[1]
    tmpij += lib.einsum('mnie,njab,me->ijab', imds.Wooov, imds.t2, r1)

    tmpabij = lib.einsum('mbej,imae->ijab', imds.Wovvo, r2)
    tmpabij = tmpabij - tmpabij.transpose(1, 0, 2, 3)
    tmpabij = tmpabij - tmpabij.transpose(0, 1, 3, 2)
    Hr2 = tmpabij
    Hr2 += tmpab - tmpab.transpose(0, 1, 3, 2)
    Hr2 += tmpij - tmpij.transpose(1, 0, 2, 3)
    Hr2 += 0.5*lib.einsum('mnij,mnab->ijab', imds.Woooo, r2)
    nij = nocc*nocc
    Hr2 += wvvvv_dot_x(r2.reshape(nij, t1.shape[1], t1.shape[1]),
                       t1, t2, eris, ovvv).reshape(nocc, nocc, t1.shape[1], t1.shape[1])
    return eom.amplitudes_to_vector(Hr1, Hr2)


def eeccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape
    abab = wvvvv_diag(imds.eris)
    fvv = np.diagonal(imds.Fvv)
    foo = np.diagonal(imds.Foo)
    wov = einsum('iaai->ia', imds.Wovvo)
    Hr1 = fvv[None, :] - foo[:, None] + wov
    Hr2 = (fvv[None, None, :, None] + fvv[None, None, None, :]
           - foo[:, None, None, None] - foo[None, :, None, None]
           + abab[None, None, :, :])
    return eom.amplitudes_to_vector(Hr1, Hr2)


def leaccsd_matvec(eom, vector, imds=None, diag=None):
    if imds is None: imds = eom.make_imds()
    nocc, nmo = eom.nocc, eom.nmo
    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)
    t1, t2, eris, ovvv = imds.t1, imds.t2, imds.eris, imds.ovvv

    Hr1 = lib.einsum('ac,a->c', imds.Fvv, r1)
    Hr1 += 0.5*lib.einsum('abcj,jab->c', imds.Wvvvo, r2)
    Hr2 = lib.einsum('alcd,a->lcd', imds.Wvovv, r1)
    Hr2 += lib.einsum('ld,a->lad', imds.Fov, r1)
    Hr2 -= lib.einsum('la,d->lad', imds.Fov, r1)
    tmp1 = lib.einsum('ac,jab->jcb', imds.Fvv, r2)
    Hr2 += tmp1 - tmp1.transpose(0, 2, 1)
    Hr2 -= lib.einsum('lj,jab->lab', imds.Foo, r2)
    tmp2 = lib.einsum('lbdj,jab->lad', imds.Wovvo, r2)
    Hr2 += tmp2 - tmp2.transpose(0, 2, 1)
    Hr2 += wvvvv_dot_xT(r2, t1, t2, eris, ovvv)        # 0.5 Wvvvv[ab,cd] r2[j,ab]
    Hr2 -= 0.5*lib.einsum('klcd,jab,kjab->lcd', imds.Woovv, r2, imds.t2)
    return eom.amplitudes_to_vector(Hr1, Hr2)


def eaccsd_star_contract(eom, eaccsd_evals, eaccsd_evecs, leaccsd_evecs, imds=None):
    '''EA-EOM-CCSD*(a) perturbative correction, vvvv-free (the only vvvv use,
    sum_f <bc||ef> r1[f], is evaluated AO-direct).'''
    assert eom.partition is None
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    eris = imds.eris
    fock = eris.fock
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    foo = fock[:nocc, :nocc].diagonal()
    fvv = fock[nocc:, nocc:].diagonal()
    oovv = np.asarray(eris.oovv)
    ovvv = imds.ovvv
    ovov = np.asarray(eris.ovov)
    ooov = np.asarray(eris.ooov)
    vooo = np.asarray(ooov).conj().transpose(3, 2, 1, 0)
    vvvo = np.asarray(ovvv).conj().transpose(3, 2, 1, 0)

    eabc = fvv[:, None, None] + fvv[None, :, None] + fvv[None, None, :]
    eij = foo[:, None] + foo[None, :]
    eijabc = eij[:, :, None, None, None] - eabc[None, None, :, :, :]

    def pabc(tmp):
        return tmp + tmp.transpose(0, 1, 3, 4, 2) + tmp.transpose(0, 1, 4, 2, 3)

    def pij(tmp):
        return tmp - tmp.transpose(1, 0, 2, 3, 4)

    eaccsd_evecs = np.atleast_2d(np.array(eaccsd_evecs))
    leaccsd_evecs = np.atleast_2d(np.array(leaccsd_evecs))
    eaccsd_evals = np.atleast_1d(eaccsd_evals)
    e_star = []
    for ea_eval, ea_evec, ea_levec in zip(eaccsd_evals, eaccsd_evecs, leaccsd_evecs):
        l1, l2 = eom.vector_to_amplitudes(ea_levec, nmo, nocc)
        r1, r2 = eom.vector_to_amplitudes(ea_evec, nmo, nocc)
        ldotr = np.dot(l1, r1) + 0.5 * np.dot(l2.ravel(), r2.ravel())
        if abs(ldotr) < 1e-7:
            logger.warn(eom, 'Small left-right overlap %s', ldotr)
        l1 = l1 / ldotr
        l2 = l2 / ldotr
        denom = 1. / (eijabc + ea_eval)

        tmp = lib.einsum('c,ijab->ijabc', l1, oovv)
        lijabc = -pabc(tmp)
        tmp = lib.einsum('jima,mbc->ijabc', ooov, l2)
        lijabc += -pabc(tmp)
        tmp = lib.einsum('ieab,jce->ijabc', ovvv, l2)
        tmp = pabc(tmp)
        lijabc += -pij(tmp)

        tmp = vvvv_dot_v1(r1, eris)                     # sum_f <bc||ef> r1[f]
        tmp = lib.einsum('bce,ijae->ijabc', tmp, t2)
        rijabc = -pabc(tmp)
        tmp = lib.einsum('mcje,e->mcj', ovov, r1)
        tmp = lib.einsum('mcj,imab->ijabc', tmp, t2)
        rijabc += pij(pabc(tmp))
        tmp = lib.einsum('amij,mcb->ijabc', vooo, r2)
        rijabc += pabc(tmp)
        tmp = lib.einsum('baei,jce->ijabc', vvvo, r2)
        rijabc -= pij(pabc(tmp))

        deltaE = (1. / 12) * lib.einsum('ijabc,ijabc,ijabc', lijabc, rijabc, denom)
        e_star.append(ea_eval + deltaE.real)
    return e_star


# ---------------------------------------------------------------------------
# Driver classes
# ---------------------------------------------------------------------------

class EOMIP(eom_gccsd.EOMIP):
    # IP has no particle-particle ladder (no Wvvvv); it only needs ovvv,
    # which _IMDS provides.  Standard matvec/diag are reused unchanged.
    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ip()
        return imds


class EOMEA(eom_gccsd.EOMEA):
    matvec = eaccsd_matvec
    l_matvec = leaccsd_matvec
    get_diag = eaccsd_diag
    ccsd_star_contract = eaccsd_star_contract

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ea()
        return imds


class EOMEE(eom_gccsd.EOMEE):
    matvec = eeccsd_matvec
    get_diag = eeccsd_diag

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ee()
        return imds
