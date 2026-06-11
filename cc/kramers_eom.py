#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
# Kramers-augmented Davidson for IP/EA-EOM-CCSD on a (closed-shell,
# Kramers-symmetric) spinor reference.
#
# The similarity-transformed Hamiltonian Hbar commutes with time reversal K,
# and for an odd-electron target (IP/EA) K^2 = -1, so every root is an exact
# Kramers doublet {R, K R} with K R orthogonal to R, and the partner's matvec
# is free:  Hbar (K R) = K (Hbar R).  This solver matvecs only one component of
# each doublet and generates the partner by applying K, yielding **exactly
# degenerate, properly Kramers-paired** roots -- the clean way to identify and
# track IP/EA doublets instead of the abstract "follow by energy".
#
# K acting on an EOM amplitude is built from the MO time-reversal matrix
# T[q,p] = <phi_q | K phi_p> = (C^dag S P C^*)[q,p], where P is the signed AO
# Kramers permutation from mol.time_reversal_map().  Verified on H2O/cc-pVDZ
# (spinor IP-EOM): K R is a degenerate eigenvector of Hbar to the EOM residual
# tolerance, ||T^2+I|| ~ 1e-13, <R|K R> ~ 1e-16, and the energies match the
# plain (Kramers-unaware) eom kernel to ~2e-8.
#
# NOTE ON COST: the "free partner" does NOT, by itself, speed up the solve over
# pyscf's well-tuned non-symmetric Davidson.  That Davidson already resolves a
# degenerate pair without paying ~2x (it finds 6 degenerate states in about the
# same matvecs as 6 non-degenerate ones), so there is little redundant work for
# the free partner to remove; with the buffer-root overhead this hand-rolled
# solver is in fact comparable-to-slightly-slower (e.g. 66 vs 54 matvecs for 3
# doublets).  The value here is the *structure*: guaranteed clean Kramers pairs
# and the reusable time-reversal-on-amplitudes machinery (also useful for
# enforcing exact degeneracy, properties, and symmetry analysis).  A genuine
# speedup needs the cost of each sigma-vector reduced via Kramers/quaternion-
# adapted intermediates (a CC-layer rewrite), not the eigensolver.
#

import numpy as np
import scipy.linalg
from pyscf.cc import ccsd
from pyscf.lib import logger


def time_reversal_mo(mycc, mf=None):
    '''Active-space MO time-reversal blocks (To over occ, Tv over vir).

    Returns (To, Tv, Tmo) with Tmo[q,p] = <phi_q | K phi_p> for the active
    spinor MOs; K phi_p = sum_q phi_q Tmo[q,p], Tmo^2 = -I.'''
    if mf is None:
        mf = mycc._scf
    mol = mf.mol
    moidx = ccsd.get_frozen_mask(mycc)
    C = mf.mo_coeff[:, moidx]
    S = mf.get_ovlp()
    tao = np.asarray(mol.time_reversal_map())
    idx = np.abs(tao) - 1
    sgn = np.where(tao < 0, -1.0, 1.0)
    n2c = tao.size
    P = np.zeros((n2c, n2c))
    P[idx, np.arange(n2c)] = sgn                 # K f_i = sgn_i f_{idx_i}
    Tmo = C.conj().T @ S @ P @ C.conj()
    nocc = mycc.nocc
    return Tmo[:nocc, :nocc], Tmo[nocc:, nocc:], Tmo


def make_kbar_ip(eom, To, Tv):
    '''Time-reversal operator on IP amplitudes (r1[i], r2[i,j,a]).'''
    nmo, nocc = eom.nmo, eom.nocc
    def kbar(vec):
        r1, r2 = eom.vector_to_amplitudes(vec, nmo, nocc)
        r1b = (To @ r1).conj()
        r2b = np.einsum('ki,lj,ba,ija->klb', To, To, Tv.conj(), r2).conj()
        return eom.amplitudes_to_vector(r1b, r2b)
    return kbar


def make_kbar_ea(eom, To, Tv):
    '''Time-reversal operator on EA amplitudes (r1[a], r2[i,a,b]).'''
    nmo, nocc = eom.nmo, eom.nocc
    def kbar(vec):
        r1, r2 = eom.vector_to_amplitudes(vec, nmo, nocc)
        r1b = (Tv @ r1).conj()
        # r2[i,a,b]: i occ (annihilation), a,b vir (creation)
        r2b = np.einsum('ki,ca,db,iab->kcd', To.conj(), Tv, Tv, r2).conj()
        return eom.amplitudes_to_vector(r1b, r2b)
    return kbar


def _orth_add(V, W, x, Hx, lindep=1e-9):
    '''Append x (with known Hx = Hbar x) to the orthonormal basis V, keeping
    W = Hbar V consistent under the Gram-Schmidt transform (no extra matvec).'''
    for _ in range(2):
        if len(V):
            c = V.conj().dot(x)
            x = x - c.dot(V)
            Hx = Hx - c.dot(W)
    nrm = np.linalg.norm(x)
    if nrm < lindep:
        return V, W, False
    return (np.vstack([V, x/nrm]), np.vstack([W, Hx/nrm]), True)


def kr_davidson(matvec, kbar, hdiag, npair, conv_tol=1e-7, max_cycle=60,
                max_space=None, verbose=0):
    '''Kramers-augmented non-symmetric Davidson for the lowest `npair` IP/EA
    doublets.  Returns (conv, e, rs, ks, nmatvec): distinct eigenvalues e
    (ascending real part), one representative eigenvector rs[k] per doublet and
    its free Kramers partner ks[k] = K rs[k].  Only one matvec per doublet per
    new direction (the partner sigma is free).'''
    n = hdiag.size
    hdiag = np.asarray(hdiag)
    # track a few buffer doublets beyond npair so a root found late (or after a
    # collapse) cannot be silently displaced by a spurious higher one.
    ntrack = min(npair + 2, n // 2)
    if max_space is None:
        max_space = max(8*ntrack, 24)

    # init guess: lowest-diagonal unit vectors (primaries)
    ng = min(ntrack, n)
    g_idx = np.argsort(hdiag.real)[:ng]
    prim = [np.eye(n, dtype=complex)[i] for i in g_idx]

    V = np.zeros((0, n), dtype=complex)
    W = np.zeros((0, n), dtype=complex)
    nmatvec = 0
    conv = np.zeros(npair, dtype=bool)
    e = np.zeros(ntrack); rs = np.zeros((ntrack, n), complex); ks = np.zeros((ntrack, n), complex)

    for icyc in range(max_cycle):
        # expand subspace: each primary needs one matvec; its partner is free
        for p in prim:
            Hp = matvec([p])[0]; nmatvec += 1
            V, W, ok = _orth_add(V, W, p, Hp)
            q = kbar(p); Hq = kbar(Hp)            # free partner (Hbar K p = K Hbar p)
            V, W, ok = _orth_add(V, W, q, Hq)
        m = len(V)

        Hred = V.conj().dot(W.T)                  # non-symmetric reduced Hbar
        w, vr = scipy.linalg.eig(Hred)
        order = np.argsort(w.real)
        w, vr = w[order], vr[:, order]

        # pick the lowest `ntrack` DISTINCT eigenvalues (each doublet x2)
        sel = []
        for i in range(len(w)):
            if all(abs(w[i]-w[j]) > 1e-6 for j in sel):
                sel.append(i)
            if len(sel) == ntrack:
                break
        sel = np.asarray(sel)

        new_prim = []
        rnorm = np.zeros(len(sel))
        for t, i in enumerate(sel):
            y = vr[:, i]
            x = V.T.dot(y); Hx = W.T.dot(y)
            wk = w[i]
            r = Hx - wk*x
            rnorm[t] = np.linalg.norm(r)
            e[t] = wk.real; rs[t] = x; ks[t] = kbar(x)
            if rnorm[t] >= conv_tol:
                den = hdiag - wk
                den = np.where(np.abs(den) > 1e-3, den, 1e-3)
                new_prim.append(r/den)
        nret = min(npair, len(sel))
        conv[:nret] = rnorm[:nret] < conv_tol
        if verbose:
            print('kr-davidson %2d  m=%3d  e=%s  |r|max=%.2e  nmv=%d'
                  % (icyc+1, m, np.array2string(e[:len(sel)], precision=6),
                     rnorm.max(), nmatvec))
        # converge the requested npair, but also keep the buffer roots small so
        # they cannot later displace a "converged" one
        if (conv[:nret].all() and len(sel) >= npair
                and (rnorm[npair:] < max(conv_tol*1e3, 1e-4)).all()):
            break
        if not new_prim:
            break
        if m + 2*len(new_prim) > max_space:       # collapse to current Ritz reps
            V = np.zeros((0, n), dtype=complex); W = np.zeros((0, n), dtype=complex)
            prim = [rs[t] for t in range(len(sel))] + new_prim
        else:
            prim = new_prim
    return conv, e[:npair], rs[:npair], ks[:npair], nmatvec
