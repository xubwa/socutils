#
# Brute-force, GUARANTEED-correct spin-orbital CCSDT ground truth.
#
# This module is the unambiguous oracle used to validate the production
# socutils.cc.zccsdt residual terms.  It does NOT use any hand-derived CC
# algebra.  Instead it works in the full determinant (occupation number) space
# of the active spin-orbitals and builds
#
#     Hbar = e^{-T} H_N e^{T},        T = T1 + T2 + T3
#
# as an explicit matrix in determinant space (the system is tiny:
# nocc~8, nvir~4 spinors, FCI dim ~ C(12,8) = 495), using *matrix*
# exponentials (exact, no BCH truncation).  The CCSDT residuals are then the
# *connected* projections
#
#     r1_{i}^{a}      = <Phi_i^a   | Hbar | Phi_0>
#     r2_{ij}^{ab}    = <Phi_ij^ab | Hbar | Phi_0>
#     r3_{ijk}^{abc}  = <Phi_ijk^abc | Hbar | Phi_0>
#
# Because T contains only T1,T2,T3, the singly/doubly/triply excited
# projections of e^{-T} H e^{T} |0> are automatically connected (the
# disconnected pieces would require T4+ to bring weight back into the
# <=triples manifold, which is absent), so these projections ARE the CCSDT
# amplitude equations.  This is correct by construction.
#
# Conventions match socutils.cc.zccsd: antisymmetrized physicist <pq||rs>,
# complex amplitudes, fully antisymmetric t2/t3, and the .conj() integral
# convention used by zccsd.update_amps (oovv/ooov/ovvv/fov carry .conj()).
#
# Slow but unambiguous.  Use only for validation / small systems.
#

import numpy as np
import itertools
from pyscf import lib

einsum = lib.einsum


def build_g(eris, nocc, nmo):
    '''Reconstruct the full antisymmetrized integral tensor g[p,q,r,s]=<pq||rs>
    over all spin-orbitals from the 7 stored eris blocks, using the
    permutational/Hermitian symmetries
        <pq||rs> = -<qp||rs> = -<pq||sr> = <rs||pq>*.
    '''
    g = np.zeros((nmo, nmo, nmo, nmo), dtype=complex)
    o = slice(0, nocc)
    v = slice(nocc, nmo)
    oooo = np.asarray(eris.oooo)
    ooov = np.asarray(eris.ooov)
    oovv = np.asarray(eris.oovv)
    ovov = np.asarray(eris.ovov)
    ovvo = np.asarray(eris.ovvo)
    ovvv = np.asarray(eris.ovvv)
    vvvv = np.asarray(eris.vvvv)

    # direct blocks
    g[o, o, o, o] = oooo
    g[o, o, o, v] = ooov                       # <ij||ka>
    g[o, o, v, v] = oovv                       # <ij||ab>
    g[o, v, o, v] = ovov                       # <ia||jb>
    g[o, v, v, o] = ovvo                       # <ia||bj>
    g[o, v, v, v] = ovvv                       # <ia||bc>
    g[v, v, v, v] = vvvv

    # antisymmetry within bra (pq) and ket (rs):
    # ooov -> oovo  (swap r,s): <ij||ak> = -<ij||ka>
    g[o, o, v, o] = -ooov.transpose(0, 1, 3, 2)
    # ovov/ovvo give voov, vovo etc via swapping p,q
    g[v, o, o, v] = -ovov.transpose(1, 0, 2, 3)   # <ai||jb> = -<ia||jb>
    g[v, o, v, o] = -ovvo.transpose(1, 0, 2, 3)   # <ai||bj> = -<ia||bj>

    # ovvv -> vovv : <ai||bc> = -<ia||bc>
    g[v, o, v, v] = -ovvv.transpose(1, 0, 2, 3)
    # ooov -> ovoo / vooo via Hermiticity: <ka||ij> = <ij||ka>* (swap bra<->ket)
    g[o, v, o, o] = ooov.transpose(2, 3, 0, 1).conj()   # <ka||ij>
    g[v, o, o, o] = -g[o, v, o, o].transpose(1, 0, 2, 3)  # <ak||ij>
    # oovv -> vvoo : <ab||ij> = <ij||ab>*
    g[v, v, o, o] = oovv.transpose(2, 3, 0, 1).conj()
    # ovvv -> vvov / vvvo : <bc||ia> = <ia||bc>* (Hermiticity)
    g[v, v, o, v] = ovvv.transpose(2, 3, 0, 1).conj()   # <bc||ia>
    g[v, v, v, o] = -g[v, v, o, v].transpose(0, 1, 3, 2)  # <bc||ai>
    return g


# --------------------------------------------------------------------------
# Determinant (occupation-number) machinery over the active spin-orbitals.
# A determinant is an integer bitmask; bit p set => spin-orbital p occupied.
# Orbitals 0..nocc-1 are occupied in the HF reference; nocc..nmo-1 virtual.
# --------------------------------------------------------------------------

def _popcount(x):
    return bin(x).count('1')


def _build_dets(nmo, nelec):
    '''All determinants (bitmasks) with `nelec` electrons in `nmo` spin-orbitals,
    plus a dict mapping bitmask -> index.'''
    dets = []
    for occ in itertools.combinations(range(nmo), nelec):
        mask = 0
        for p in occ:
            mask |= (1 << p)
        dets.append(mask)
    dets.sort()
    index = {m: i for i, m in enumerate(dets)}
    return dets, index


def _orbs(mask, nmo):
    return [p for p in range(nmo) if (mask >> p) & 1]


def _phase_remove(mask, p):
    '''Fermion sign for annihilating orbital p from `mask` (must be occupied).
    Sign = (-1)^(number of occupied orbitals with index < p).'''
    lower = mask & ((1 << p) - 1)
    return -1.0 if (_popcount(lower) & 1) else 1.0


def _apply_excitation(mask, holes, parts, nmo):
    '''Apply  a_{parts[0]}^+ ... a_{holes[0]} ...  (de-excitation/excitation
    string) to a determinant.

    We apply, in order, annihilation of each `hole` orbital then creation of
    each `part` orbital, i.e. operator  prod_creations prod_annihilations.
    Returns (new_mask, sign) or (None, 0) if it annihilates the determinant.

    Convention: the operator is  c^+_{p1} c^+_{p2} ... c_{h2} c_{h1}
    applied right-to-left.  We pass holes=[h1,h2,...] (annihilated first->...),
    parts=[p1,p2,...].  Annihilations applied first (h1 then h2 ...), then
    creations (last listed first to mirror normal ordering is handled by the
    caller via consistent ordering of indices in both bra and operator).
    '''
    sign = 1.0
    m = mask
    # annihilate holes in given order
    for h in holes:
        if not ((m >> h) & 1):
            return None, 0.0
        sign *= _phase_remove(m, h)
        m &= ~(1 << h)
    # create parts in given order
    for p in parts:
        if (m >> p) & 1:
            return None, 0.0
        sign *= _phase_remove(m, p)  # same lower-bit counting for creation
        m |= (1 << p)
    return m, sign


# --------------------------------------------------------------------------
# Build the electronic Hamiltonian matrix in determinant space.
#   H = sum_pq h_pq c^+_p c_q + 1/4 sum_pqrs <pq||rs> c^+_p c^+_q c_s c_r
# where h is the one-body (Fock minus the mean-field potential is NOT used;
# we use the full Slater-Condon construction directly from f and <pq||rs>).
#
# To keep this purely diagrammatic-free we build H by Slater-Condon rules
# using the antisymmetrized two-electron integrals g[p,q,r,s] = <pq||rs> and
# a one-body matrix h1[p,q] such that the diagonal reproduces HF energy.
#
# We reconstruct the *bare* one-electron h from the Fock matrix:
#   f_pq = h_pq + sum_{i occ} <pi||qi>     (sum over occupied spin-orbitals)
# so  h_pq = f_pq - sum_{i occ} <pi||qi>.
# Then E_HF = sum_i h_ii + 1/2 sum_ij <ij||ij>  (relative energy; constant).
# We work with energies relative to E_HF, so the constant cancels in Hbar
# projections onto excited determinants.
# --------------------------------------------------------------------------

class _Hbuilder:
    def __init__(self, h1, g, nmo, nocc):
        self.h1 = h1
        self.g = g
        self.nmo = nmo
        self.nocc = nocc

    def diag(self, occ):
        h1 = self.h1
        g = self.g
        e = 0.0
        for p in occ:
            e += h1[p, p]
        for a in range(len(occ)):
            for b in range(a + 1, len(occ)):
                p, q = occ[a], occ[b]
                e += g[p, q, p, q].real if False else g[p, q, p, q]
        return e


def _build_h1_from_fock(fock, g, nocc):
    nmo = fock.shape[0]
    h1 = fock.copy()
    # subtract mean-field: f_pq = h_pq + sum_{i occ} <pi||qi>
    for i in range(nocc):
        h1[:, :] -= g[:, i, :, i]
    return h1


def build_H(fock, g, nmo, nocc, nelec):
    '''Full electronic Hamiltonian matrix in the determinant space (complex).'''
    dets, index = _build_dets(nmo, nelec)
    ndet = len(dets)
    h1 = _build_h1_from_fock(fock, g, nocc)
    H = np.zeros((ndet, ndet), dtype=complex)

    for I, mask in enumerate(dets):
        occ = _orbs(mask, nmo)
        # diagonal: one-body
        diag = 0.0
        for p in occ:
            diag += h1[p, p]
        for a in range(len(occ)):
            for b in range(a + 1, len(occ)):
                p, q = occ[a], occ[b]
                diag += g[p, q, p, q]
        H[I, I] += diag

        # one-body off-diagonal: c^+_p c_q, p in occ-complement, q in occ
        vir = [p for p in range(nmo) if not ((mask >> p) & 1)]
        for q in occ:
            for p in vir:
                newm, sgn = _apply_excitation(mask, [q], [p], nmo)
                if newm is None:
                    continue
                J = index[newm]
                val = h1[p, q]
                # plus the contraction with occupied: sum_{i in occ\{q}} <pi||qi>
                for i in occ:
                    if i == q:
                        continue
                    val += g[p, i, q, i]
                H[J, I] += sgn * val

        # two-body: double excitations c^+_p c^+_r c_s c_q  (p,r in vir; q,s in occ)
        for qi in range(len(occ)):
            for si in range(qi + 1, len(occ)):
                q, s = occ[qi], occ[si]
                for pi in range(len(vir)):
                    for ri in range(pi + 1, len(vir)):
                        p, r = vir[pi], vir[ri]
                        # operator a^+_p a^+_r a_s a_q :
                        newm, sgn = _apply_excitation(mask, [q, s], [r, p], nmo)
                        if newm is None:
                            continue
                        J = index[newm]
                        # coefficient is <pr||qs> (antisymmetrized)
                        H[J, I] += sgn * g[p, r, q, s]
    return dets, index, H


# --------------------------------------------------------------------------
# Build a cluster-operator matrix in determinant space from amplitude tensors.
#   T1 = sum_{ia} t1[i,a] a^+_a a_i
#   T2 = 1/4 sum t2[ijab] a^+_a a^+_b a_j a_i   (t2 fully antisym)
#   T3 = 1/36 sum t3[ijkabc] a^+_a a^+_b a^+_c a_k a_j a_i
# Indices i,j,k label occupied spin-orbitals (0..nocc-1), a,b,c virtual
# (offset by nocc).
# --------------------------------------------------------------------------

def build_T(t1, t2, t3, dets, index, nocc, nmo):
    ndet = len(dets)
    T = np.zeros((ndet, ndet), dtype=complex)
    nvir = nmo - nocc

    for I, mask in enumerate(dets):
        # T1
        for i in range(nocc):
            if not ((mask >> i) & 1):
                continue
            for a in range(nvir):
                A = nocc + a
                if (mask >> A) & 1:
                    continue
                newm, sgn = _apply_excitation(mask, [i], [A], nmo)
                if newm is None:
                    continue
                J = index[newm]
                T[J, I] += sgn * t1[i, a]
        # T2: 1/4 sum_{ijab} t2 a^+_a a^+_b a_j a_i ; antisym -> pick i<j,a<b *... but
        # to be safe just sum all and rely on 1/4 with full antisym tensor.
        occ_o = [i for i in range(nocc) if (mask >> i) & 1]
        vir_v = [a for a in range(nvir) if not ((mask >> (nocc + a)) & 1)]
        for i in occ_o:
            for j in occ_o:
                if j == i:
                    continue
                for a in vir_v:
                    for b in vir_v:
                        if b == a:
                            continue
                        A, B = nocc + a, nocc + b
                        # operator a^+_A a^+_B a_j a_i
                        newm, sgn = _apply_excitation(mask, [i, j], [B, A], nmo)
                        if newm is None:
                            continue
                        J = index[newm]
                        T[J, I] += 0.25 * sgn * t2[i, j, a, b]
        # T3: 1/36 sum t3 a^+_A a^+_B a^+_C a_k a_j a_i
        if t3 is not None:
            for i in occ_o:
                for j in occ_o:
                    if j == i:
                        continue
                    for k in occ_o:
                        if k == i or k == j:
                            continue
                        for a in vir_v:
                            for b in vir_v:
                                if b == a:
                                    continue
                                for c in vir_v:
                                    if c == a or c == b:
                                        continue
                                    A, B, C = nocc + a, nocc + b, nocc + c
                                    newm, sgn = _apply_excitation(
                                        mask, [i, j, k], [C, B, A], nmo)
                                    if newm is None:
                                        continue
                                    J = index[newm]
                                    T[J, I] += (1.0 / 36.0) * sgn * t3[i, j, k, a, b, c]
    return T


# --------------------------------------------------------------------------
# Reference determinant and excited-determinant bookkeeping
# --------------------------------------------------------------------------

def _ref_mask(nocc):
    m = 0
    for i in range(nocc):
        m |= (1 << i)
    return m


def _excited_mask(nocc, nmo, occ_holes, vir_parts):
    '''Determinant obtained from the HF reference by replacing the listed
    occupied orbitals (occ_holes) by the listed virtual orbitals (vir_parts),
    together with the fermion sign of  a^+_{parts} a_{holes} |HF> relative to
    the *sorted* occupation of the resulting determinant.

    We return (mask, sign) where sign relates the operator-string-ordered
    excited state to the canonical (index-sorted) determinant stored in `dets`.
    '''
    ref = _ref_mask(nocc)
    parts = [nmo and (p) for p in vir_parts]
    newm, sgn = _apply_excitation(ref, list(occ_holes), list(vir_parts), nmo)
    return newm, sgn


# --------------------------------------------------------------------------
# CCSDT residuals via Hbar in determinant space.
# --------------------------------------------------------------------------

def ccsdt_residuals(fock, g, nocc, nmo, t1, t2, t3):
    '''Return (r1, r2, r3) connected CCSDT residuals (NOT divided by
    denominators) for the given amplitudes, computed by the determinant-space
    Hbar = e^{-T} H e^{T}.

    g[p,q,r,s] = <pq||rs> antisymmetrized physicist integrals.
    fock = MO Fock matrix.
    '''
    nelec = nocc
    nvir = nmo - nocc
    dets, index, H = build_H(fock, g, nmo, nocc, nelec)
    # Shift H by -E_ref * I so its entries are O(1); this leaves all excited
    # projections <mu|Hbar|0> unchanged (constant * I commutes with T and
    # <mu|0>=0) but greatly improves the numerical conditioning of the matrix
    # exponentials below.
    ref0 = _ref_mask(nocc)
    eref = H[index[ref0], index[ref0]]
    H = H - eref * np.eye(H.shape[0])
    T = build_T(t1, t2, t3, dets, index, nocc, nmo)

    # Hbar|0> = e^{-T} H e^{T} |0> = (H + [H,T] + 1/2[[H,T],T] + ...) |0>.
    # T is a pure excitation operator (nilpotent), so the BCH series
    # terminates.  We need the projection of Hbar|0> onto the <=triples
    # manifold; the connected commutator series truncated at 4th order (the
    # highest order that can return to <=triples from H+T1+T2+T3) is exact.
    # Evaluating it as nested commutators applied to the single vector |0>
    # (matrix-vector products) is far better conditioned than forming the
    # dense e^{-T}He^{T}.
    ref = _ref_mask(nocc)
    I0 = index[ref]
    ndet = H.shape[0]
    v0 = np.zeros(ndet, dtype=complex)
    v0[I0] = 1.0

    # Hbar|0> = e^{-T} (H (e^{T} |0>)), all evaluated as matrix-VECTOR products
    # via the finite (nilpotent T) exponential series.  This avoids forming the
    # dense e^{-T} H e^{T} product and is numerically well conditioned.
    def exp_apply(Tmat, vec):
        out = vec.copy()
        term = vec.copy()
        n = 0
        while True:
            n += 1
            term = (Tmat @ term) / n
            if np.abs(term).max() < 1e-16 or n > 30:
                break
            out = out + term
        return out

    w = exp_apply(T, v0)        # e^{T} |0>
    w = H @ w                   # H e^{T} |0>
    col = exp_apply(-T, w)      # e^{-T} H e^{T} |0>

    # singles
    r1 = np.zeros((nocc, nvir), dtype=complex)
    for i in range(nocc):
        for a in range(nvir):
            A = nocc + a
            newm, sgn = _apply_excitation(ref, [i], [A], nmo)
            if newm is None:
                continue
            r1[i, a] = sgn * col[index[newm]]

    # doubles  (canonical i<j, a<b then antisymmetrize)
    r2 = np.zeros((nocc, nocc, nvir, nvir), dtype=complex)
    for i in range(nocc):
        for j in range(nocc):
            if i == j:
                continue
            for a in range(nvir):
                for b in range(nvir):
                    if a == b:
                        continue
                    A, B = nocc + a, nocc + b
                    # bra <Phi_ij^ab| corresponds to operator a^+_A a^+_B a_j a_i
                    newm, sgn = _apply_excitation(ref, [i, j], [B, A], nmo)
                    if newm is None:
                        continue
                    r2[i, j, a, b] = sgn * col[index[newm]]

    # triples
    r3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=complex)
    for i in range(nocc):
        for j in range(nocc):
            if j == i:
                continue
            for k in range(nocc):
                if k == i or k == j:
                    continue
                for a in range(nvir):
                    for b in range(nvir):
                        if b == a:
                            continue
                        for c in range(nvir):
                            if c == a or c == b:
                                continue
                            A, B, C = nocc + a, nocc + b, nocc + c
                            newm, sgn = _apply_excitation(
                                ref, [i, j, k], [C, B, A], nmo)
                            if newm is None:
                                continue
                            r3[i, j, k, a, b, c] = sgn * col[index[newm]]

    return r1, r2, r3


def kernel(fock, g, nocc, nmo, mo_energy, conv_tol=1e-11, max_cycle=300,
           verbose=False):
    '''Iterate the brute-force CCSDT residual equations to convergence.

    Returns (e_corr, t1, t2, t3).
    '''
    nvir = nmo - nocc
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    eijab = (eia[:, None, :, None] + eia[None, :, None, :])
    eijkabc = (eia[:, None, None, :, None, None]
               + eia[None, :, None, None, :, None]
               + eia[None, None, :, None, None, :])

    # MP1 start (zccsd .conj() convention for the driving integrals)
    fov = fock[:nocc, nocc:]
    oovv = g[:nocc, :nocc, nocc:, nocc:]
    t1 = fov.conj() / eia
    t2 = oovv.conj() / eijab
    t3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=complex)

    def energy(t1, t2):
        e = einsum('ia,ia', fov, t1)
        e += 0.25 * einsum('ijab,ijab', t2, oovv)
        e += 0.5 * einsum('ia,jb,ijab', t1, t1, oovv)
        return e.real

    e_old = energy(t1, t2)
    adiis = lib.diis.DIIS()
    adiis.space = 8
    for it in range(max_cycle):
        r1, r2, r3 = ccsdt_residuals(fock, g, nocc, nmo, t1, t2, t3)
        t1n = t1 + r1 / eia
        t2n = t2 + r2 / eijab
        t3n = t3 + r3 / eijkabc
        vec = np.hstack((t1n.ravel(), t2n.ravel(), t3n.ravel()))
        vec = adiis.update(vec)
        n1 = nocc * nvir
        n2 = nocc * nocc * nvir * nvir
        t1 = vec[:n1].reshape(nocc, nvir)
        t2 = vec[n1:n1 + n2].reshape(nocc, nocc, nvir, nvir)
        t3 = vec[n1 + n2:].reshape(nocc, nocc, nocc, nvir, nvir, nvir)
        e_new = energy(t1, t2)
        de = e_new - e_old
        if verbose:
            print('  bruteforce cycle %d  E=%.12f  dE=%.3e' % (it + 1, e_new, de))
        e_old = e_new
        if abs(de) < conv_tol:
            break
    return e_old, t1, t2, t3
