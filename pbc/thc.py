'''Level 3 of the compact/smooth exchange ladder (HANDOFF Open item 2): Gamma-point exchange with the smooth-smooth
(SS) pairs by THC/ISDF on a uniform grid and the pairs with a compact function (CX) by the level-2 local fit
(pbc/bench/hybrid_k.py).

Basis: split at alpha_c (sr_locality.split_basis), chi_full = chi_split T; every function is compact (all
primitives >= alpha_c) or smooth.  Kernel convention (pbc/rsdf.py, lr='gspace'): the RSDF vectors carry
    (p|q)_vec = (p|q)_SR,erfc + (p|q)_LR,G!=0,          (p|q)' = (p|q)_vec - c0 S_p S_q,
and in G space (rho(G) = int_cell rho(r) e^{-iGr}, (a|b) = 1/V sum_G w(G) a(G)^* b(G)) this is
    w_vec(G) = 4 pi / G^2 (G != 0),   w_vec(0) = pi / w^2      (= c0 V; the G -> 0 limit of the erfc kernel),
    w_SR(G)  = 4 pi / G^2 (1 - exp(-G^2 / 4 w^2)),  w_SR(0) = pi / w^2.
THCExchange.get_k returns K in this vector convention (before the -c0 S D S term), split basis:

  SS-SS   THC:  rho_mn(r) ~= sum_mu xi_mu(r) phi_m(r_mu) phi_n(r_mu)   (Lu-Ying least-squares xi, points from a
          pivoted Cholesky of the grid Gram matrix), V_mu,nu = (xi_mu|xi_nu) with w_vec by FFT, and
          K_ab = sum_mu,nu phi_a(r_mu) [Phi D Phi^T]_mu,nu V_mu,nu phi_b(r_nu)         (no SR/LR split for SS);
  CX      SR: local fit C_p = M_DD^-1 J_D,p on the pair's two atoms (Bloch sums); one-sided C_p^T J_q against
          SS pairs q with the SS side from THC, J_P,q ~= sum_mu A_P,mu phi_m(r_mu) phi_n(r_mu),
          A_P,mu = (P|xi_mu)_SR in G space (xi is band-limited to the mesh, so the G sum over the mesh with the
          analytic FT of the Gaussian P is exact for the representation; no aux function on the grid);
          robust C_p^T J_q + J_p^T C_q - C_p^T M C_q between CX pairs (exact three-centre J);
          LR: the exact RSDF LR rows restricted to the CX-involving blocks.

This is a validation prototype on small cells: the SR three-centre integrals, metric and LR rows come from an
RSDF build with keep_raw (the CX columns are what a production code would compute locally).'''
import os, time
import numpy as np
import scipy.linalg
from scipy.linalg import blas
import scipy.fft
from pyscf import lib
from pyscf.gto.ft_ao import ft_ao
from pyscf.gto.eval_gto import BLKSIZE
from pyscf.df import addons as df_addons
from pyscf.pbc.gto.cell import estimate_ke_cutoff
from socutils.pbc.bench.hybrid_k import _inv_metric, k_from_factors
from socutils.pbc.rsdf import RSDF

OMEGA = 0.5
OMEGA_ROUTE1 = 0.35


def pivots(phi, kmax):
    '''Interpolation points: pivoted Cholesky of G(r, r') = (Phi Phi^T)(r, r')^2 on the grid, column by column.
    Returns the pivot order and the factor L (k, ngrid), G ~= L^T L; prefixes give any N_mu <= k.'''
    ng = len(phi)
    L = np.zeros((kmax, ng))
    res = np.einsum('ri,ri->r', phi, phi) ** 2
    piv = []
    for k in range(kmax):
        p = int(np.argmax(res))
        if res[p] <= 0:
            break
        piv.append(p)
        col = (phi @ phi[p]) ** 2
        col -= L[:k].T @ L[:k, p]
        L[k] = col / np.sqrt(res[p])
        res -= L[k] ** 2
        np.maximum(res, 0, out=res)
    return np.array(piv), L[:len(piv)]


def pivots_blocked(phi, kmax, b=64, over=4, tol=1e-14):
    '''Pivoted Cholesky of G = (Phi Phi^T)^2 on the grid in blocks (BLAS-3): per block the over*b grid points with
    the largest residual diagonal are candidates, their residual columns come from one GEMM, and an exact pivoted
    Cholesky among the candidates picks up to b pivots.  Same output as pivots() (pivot order, factor L); the pivot
    choice is restricted to the candidates of the block only.'''
    ng = len(phi)
    L = np.zeros((kmax, ng))
    res = np.einsum('ri,ri->r', phi, phi) ** 2
    dmax = res.max()
    piv = []
    k = 0
    while k < kmax:
        m = min(over * b, ng)
        cand = np.argpartition(-res, m - 1)[:m]
        if res[cand].max() <= tol * dmax:
            break
        Cc = (phi @ phi[cand].T) ** 2                                  # (ng, m)
        if k:
            Cc -= L[:k].T @ L[:k][:, cand]
        rc = res[cand].copy()
        nb = min(b, kmax - k)
        new = []
        for j in range(nb):
            q = int(np.argmax(rc))
            if rc[q] <= tol * dmax:
                break
            row = Cc[:, q].copy()
            for r_ in new:
                row -= r_ * r_[cand[q]]
            row /= np.sqrt(rc[q])
            new.append(row)
            piv.append(int(cand[q]))
            rc -= row[cand] ** 2
            rc[q] = 0.
        if not new:
            break
        L[k:k + len(new)] = new
        res -= (np.asarray(new) ** 2).sum(axis=0)
        np.maximum(res, 0, out=res)
        res[piv[-len(new):]] = 0.
        k += len(new)
    return np.array(piv), L[:k]


def _malloc_trim():
    '''glibc malloc_trim(0): return the free heap of every arena (C code allocating in OpenMP threads leaves
    freed buffers in per-thread arenas, counted as resident) to the system.'''
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


def space_group_ops(cell, symprec=1e-3):
    '''Space group of the cell (spglib): rotations (n, 3, 3) on fractional coordinates, translations (n, 3).'''
    try:
        import spglib
    except ImportError:                                                # no spglib: P1 (no symmetry used)
        return np.eye(3, dtype=int)[None], np.zeros((1, 3)), 'P1 (no spglib)'
    a = np.asarray(cell.lattice_vectors())
    frac = cell.atom_coords() @ np.linalg.inv(a)
    nums = [cell.atom_charge(i) * 1000 + sum(map(ord, cell.atom_symbol(i))) for i in range(cell.natm)]
    ds = spglib.get_symmetry_dataset((a, frac, nums), symprec=symprec)  # basis labels distinguish (Eu vs Eu1)
    if ds is None:
        return np.eye(3, dtype=int)[None], np.zeros((1, 3)), 'P1 (spglib failed)'
    return np.asarray(ds.rotations), np.asarray(ds.translations) % 1, ds.international


def symmetric_mesh(mesh, ops, factor=1):
    '''The smallest mesh >= mesh mapped onto itself by every space group operation (t * mesh integer, axes that a
    rotation couples equal), every entry a multiple of factor (then mesh // factor is such a mesh as well).'''
    R, T = ops
    m = np.array(mesh, dtype=int)
    den = np.ones(3, dtype=int)
    for t in T:
        for i in range(3):
            for d in range(1, 49):
                if abs(t[i] * d - round(t[i] * d)) < 1e-6:
                    den[i] = np.lcm(den[i], d)
                    break
    step = den * factor
    for _ in range(10):
        m = -(-m // step) * step
        m0 = m.copy()
        for r in R:
            for i in range(3):
                for j in range(3):
                    if r[i, j] != 0:
                        m[i] = m[j] = max(m[i], m[j])
        if (m == m0).all():
            break
    return m


def grid_orbits(mesh, ops):
    '''orbit[p]: index of the orbit of grid point p under the space group (the grid must be symmetric_mesh), and the
    member list of every orbit.'''
    R, T = ops
    mesh = np.asarray(mesh)
    ng = int(np.prod(mesh))
    ijk = np.stack(np.unravel_index(np.arange(ng), mesh), -1)
    imgs = np.empty((len(R), ng), dtype=np.int64)
    for k, (r, t) in enumerate(zip(R, T)):
        f = ijk @ r.T + np.rint(t * mesh).astype(int)
        imgs[k] = np.ravel_multi_index((f % mesh).T, mesh)
    rep = imgs.min(axis=0)                                           # orbit label: smallest member
    _, orbit = np.unique(rep, return_inverse=True)
    order = np.argsort(orbit, kind='stable')
    bounds = np.searchsorted(orbit[order], np.arange(orbit.max() + 2))
    members = [order[bounds[o]:bounds[o + 1]] for o in range(len(bounds) - 1)]
    return orbit, members


def pivots_orbits(phi, kmax, orbit, members, b=64, over=4, tol=1e-14, orbit_dep=1e-1):
    '''pivots_blocked with whole symmetry orbits: the Gram kernel is invariant under the space group (phi spans
    symmetry-closed shells), so a pivot set made of complete orbits keeps the residual symmetric and the point set
    (and the ISDF fit on the symmetric grid) has the symmetry of the crystal.  Up to kmax points (whole orbits).'''
    ng = len(phi)
    L = np.zeros((kmax + 64, ng))
    res = np.einsum('ri,ri->r', phi, phi) ** 2
    dmax = res.max()
    piv = []
    k = 0
    while k < kmax:
        m = min(over * b, ng)
        top = np.argpartition(-res, m - 1)[:m]
        top = top[np.argsort(-res[top])]
        if res[top[0]] <= tol * dmax:
            break
        _, first = np.unique(orbit[top], return_index=True)
        orbs = orbit[top[np.sort(first)]]                            # candidate orbits, by residual
        cand = np.concatenate([members[o] for o in orbs])
        Cc = (phi @ phi[cand].T) ** 2
        if k:
            Cc -= L[:k].T @ L[:k][:, cand]
        rc = res[cand].copy()
        pos = {int(c): i for i, c in enumerate(cand)}
        new = []
        dropped = []
        for o in orbs:
            mem = members[o]
            if k + len(new) + len(mem) > kmax + 63 or len(new) >= b:
                break
            if max(rc[pos[int(q)]] for q in mem) <= tol * dmax:
                continue
            # the whole orbit or nothing: a member whose residual is far below the current largest one (a
            # near-duplicate image close to a symmetry element, or nearly dependent on the orbits taken in this
            # block) would make the Gram matrix singular (quartz: smallest eigenvalue 1e-19, E erratic by 1e-4);
            # the greedy selection never takes such a point, so such orbits are rejected
            rows, rc_try, ok = [], rc.copy(), True
            top = rc.max()                                            # as the greedy choice: no pivot far below it
            for q in mem:
                iq = pos[int(q)]
                d = rc_try[iq]
                if d <= orbit_dep * top or d <= tol * dmax:
                    ok = False
                    break
                row = Cc[:, iq].copy()
                for r_ in new + rows:
                    row -= r_ * r_[q]
                row /= np.sqrt(d)
                rows.append(row)
                rc_try -= row[cand] ** 2
                rc_try[iq] = 0.
            if not ok:
                dropped.extend(int(q) for q in mem)
                continue
            new.extend(rows)
            piv.extend(int(q) for q in mem)
            rc = rc_try
            if k + len(new) >= kmax:
                break
        res[dropped] = 0.                                             # not candidates again
        if not new:
            if dropped:
                continue
            break
        L[k:k + len(new)] = new
        res -= (np.asarray(new) ** 2).sum(axis=0)
        np.maximum(res, 0, out=res)
        res[piv[-len(new):]] = 0.
        k += len(new)
    return np.array(piv), L[:k]


def pivots_blocked_dist(phi_loc, r0, ng, kmax, comm, b=64, over=4, tol=1e-14):
    '''pivots_blocked over MPI ranks: this rank holds the grid rows r0:r0 + len(phi_loc) (phi and its columns of the
    factor L, kmax x own rows); per block the global candidates come from an allgather of every rank's best ones,
    their phi rows and L columns from two Allreduce, and the small candidate Cholesky runs on every rank (same
    pivots on all of them).  Returns the pivot order (global grid indices); the factor is not kept.'''
    from mpi4py import MPI
    nl, ns = phi_loc.shape
    L = np.zeros((kmax + 64, nl))
    res = np.einsum('ri,ri->r', phi_loc, phi_loc) ** 2
    dmax = comm.allreduce(res.max() if nl else 0., op=MPI.MAX)
    piv = []
    k = 0
    while k < kmax:
        m = min(over * b, ng)
        mm = min(m, nl)
        top = np.argpartition(-res, mm - 1)[:mm] if mm else np.zeros(0, dtype=int)
        vals = np.concatenate(comm.allgather(res[top]))
        gidx = np.concatenate(comm.allgather(top + r0))
        o = np.argsort(-vals, kind='stable')[:m]
        cand, rc = gidx[o], vals[o].copy()
        if rc[0] <= tol * dmax:
            break
        mine = (cand >= r0) & (cand < r0 + nl)
        PC = np.zeros((len(cand), ns))
        PC[mine] = phi_loc[cand[mine] - r0]
        comm.Allreduce(MPI.IN_PLACE, PC, op=MPI.SUM)
        Cc = (phi_loc @ PC.T) ** 2                                    # (own rows, m)
        Ccc = (PC @ PC.T) ** 2                                        # (m, m)
        if k:
            Lc = np.zeros((k, len(cand)))
            Lc[:, mine] = L[:k, cand[mine] - r0]
            comm.Allreduce(MPI.IN_PLACE, Lc, op=MPI.SUM)
            Cc -= L[:k].T @ Lc
            Ccc -= Lc.T @ Lc
        nb = min(b, kmax - k)
        new, newc = [], []
        for j in range(nb):
            q = int(np.argmax(rc))
            if rc[q] <= tol * dmax:
                break
            row, rowc = Cc[:, q].copy(), Ccc[:, q].copy()
            for r_, c_ in zip(new, newc):
                row -= r_ * c_[q]
                rowc -= c_ * c_[q]
            sq = np.sqrt(rc[q])
            row /= sq
            rowc /= sq
            new.append(row)
            newc.append(rowc)
            piv.append(int(cand[q]))
            rc -= rowc ** 2
            rc[q] = 0.
        if not new:
            break
        L[k:k + len(new)] = new
        res -= (np.asarray(new) ** 2).sum(axis=0)
        np.maximum(res, 0, out=res)
        p_ = np.asarray(piv[-len(new):])
        p_ = p_[(p_ >= r0) & (p_ < r0 + nl)] - r0
        res[p_] = 0.
        k += len(new)
    return np.array(piv)


def isdf_xi(L, piv, nmu):
    '''Lu-Ying least-squares interpolation vectors for the first nmu pivots, xi (nmu, ngrid):
    xi = G[:, P] G[P, P]^-1 = L^T L[:, P]^-T with the Cholesky factor (L[:, P] is upper triangular in pivot
    order), i.e. xi^T = L[:, P]^-1 L -- no explicit inverse of the ill-conditioned G[P, P].'''
    Lk = L[:nmu]
    U = Lk[:, piv[:nmu]]
    return scipy.linalg.solve_triangular(U, Lk, lower=False, check_finite=False), nmu


def kmeans_points(cell, coords, weight, k, iters=40, tol=1e-4, seed=0):
    '''Interpolation points by weighted K-means on the grid (ISDF-K-means, Dong, Hu and Lin, JCTC 14, 1311 (2018)):
    Lloyd iterations with periodic (minimum-image) distances, weight w(r) = G(r, r) = (sum_m phi_m(r)^2)^2, then the
    grid point nearest to each centroid.  O(ngrid log k) per iteration (k-d tree over the 27 images of the centroids).'''
    from scipy.spatial import cKDTree
    A = np.asarray(cell.lattice_vectors())
    shifts = np.array([[i, j, l] for i in (-1, 0, 1) for j in (-1, 0, 1) for l in (-1, 0, 1)]) @ A
    rng = np.random.default_rng(seed)
    w = weight / weight.sum()
    cen = coords[rng.choice(len(coords), k, replace=False, p=w)]
    for it in range(iters):
        img = (cen[None] + shifts[:, None]).reshape(-1, 3)
        _, idx = cKDTree(img).query(coords, workers=-1)
        lab = idx % k
        d = coords - img[idx]                                  # displacement from the nearest centroid image
        ws = np.bincount(lab, weights=weight, minlength=k)
        empty = ws <= 0
        shift = np.stack([np.bincount(lab, weights=weight * d[:, x], minlength=k) for x in range(3)], -1)
        shift[~empty] /= ws[~empty, None]
        cen = cen + shift
        if empty.any():                                        # re-seed empty clusters at heavy points
            cen[empty] = coords[rng.choice(len(coords), empty.sum(), replace=False, p=w)]
        if np.sqrt((shift[~empty] ** 2).sum(1)).max() < tol:
            break
    img = (cen[None] + shifts[:, None]).reshape(-1, 3)
    dist, idx = cKDTree(img).query(coords, workers=-1)
    lab = idx % k
    order = np.lexsort((dist, lab))                            # per cluster, the nearest grid point first
    first = np.ones(len(order), dtype=bool)
    first[1:] = lab[order][1:] != lab[order][:-1]
    pts = np.unique(order[first])
    return pts, it + 1


def candidate_cholesky_points(phi, cand, k):
    '''The first k pivots of a pivoted Cholesky (LAPACK dpstrf, blocked) of the Gram matrix G = (Phi Phi^T)^2
    restricted to the candidate grid points cand.'''
    from scipy.linalg import lapack
    Pc = phi[cand]
    G = (Pc @ Pc.T) ** 2
    d = np.sqrt(G.diagonal())
    _, piv, rank, info = lapack.dpstrf(G / d[:, None] / d[None], lower=1, tol=1e-14)   # equilibrated
    return cand[piv[:min(k, rank)] - 1], rank


def isdf_xi_ls(phi, pts, cut=1e-13, blk=200000):
    '''Lu-Ying least-squares interpolation vectors for given points, xi (npts, ngrid) = (G[:, P] G[P, P]^+)^T with
    G = (Phi Phi^T)^2 (Hadamard square); the pseudo-inverse on a relative eigenvalue cut, GEMMs in grid blocks.'''
    PP = phi[pts]
    e, v = scipy.linalg.eigh((PP @ PP.T) ** 2)
    keep = e > cut * e.max()
    Ginv = (v[:, keep] / e[keep]) @ v[:, keep].T
    xi = np.empty((len(pts), len(phi)))
    for g0 in range(0, len(phi), blk):
        g1 = min(g0 + blk, len(phi))
        xi[:, g0:g1] = Ginv @ ((PP @ phi[g0:g1].T) ** 2)
    return xi, int(keep.sum())


_THCLIB = []


def _libthc():
    '''lib/libsocthc.so (OpenMP kernels of the route-1 exchange), or None if it is not built.'''
    if not _THCLIB:
        try:
            from socutils.lib import load_library
            _THCLIB.append(load_library('libsocthc', 'SOCUTILS_THC_LIBRARY'))
        except OSError:
            _THCLIB.append(None)
    return _THCLIB[0]


def _ptr(a):
    import ctypes
    return a.ctypes.data_as(ctypes.c_void_p)


def pair_accum(H, B, E, pi, pj, off):
    '''H[pi_p] += B[p] * E[pj_p], and H[pj_p] += B[p] * E[pi_p] where off_p (H, B, E real C-contiguous).'''
    lib_ = _libthc()
    if lib_ is None:
        np.add.at(H, pi, B * E[pj])
        np.add.at(H, pj[off], B[off] * E[pi[off]])
        return H
    import ctypes
    pi, pj = np.ascontiguousarray(pi, dtype=np.int32), np.ascontiguousarray(pj, dtype=np.int32)
    off = np.ascontiguousarray(off, dtype=np.int8)
    B = np.ascontiguousarray(B)
    lib_.SOCTHCpair_accum(_ptr(H), _ptr(B), _ptr(E), _ptr(pi), _ptr(pj), _ptr(off),
                          ctypes.c_int(len(pi)), ctypes.c_int(B.shape[1]))
    return H


def unpack_sym(V, ii, jj, n):
    '''(nb, npair) packed pair values -> (nb, n, n) symmetric matrices (zero elsewhere).'''
    V = np.ascontiguousarray(V, dtype=float)
    nb = V.shape[0]
    lib_ = _libthc()
    W = np.empty((nb, n, n)) if lib_ is not None else np.zeros((nb, n, n))
    if lib_ is None:
        W[:, ii, jj] = V
        W[:, jj, ii] = V
        return W
    import ctypes
    ii, jj = np.ascontiguousarray(ii, dtype=np.int32), np.ascontiguousarray(jj, dtype=np.int32)
    lib_.SOCTHCunpack_sym(_ptr(W), _ptr(V), _ptr(ii), _ptr(jj), ctypes.c_int(nb), ctypes.c_int(len(ii)),
                          ctypes.c_int(n))
    return W


def embed(W, S, idx):
    '''W[k][idx, idx] = S[k] in place (W (nb, n, n), S (nb, ns, ns)).'''
    lib_ = _libthc()
    if lib_ is None:
        for k in range(len(W)):
            W[k][np.ix_(idx, idx)] = S[k]
        return W
    import ctypes
    S = np.ascontiguousarray(S, dtype=float)
    idx = np.ascontiguousarray(idx, dtype=np.int32)
    lib_.SOCTHCembed(_ptr(W), _ptr(S), _ptr(idx), ctypes.c_int(len(W)), ctypes.c_int(len(idx)),
                     ctypes.c_int(W.shape[1]))
    return W


def add_cols(M, S, cols):
    '''M[:, cols] += S in place (C kernel; distinct cols).'''
    lib_ = _libthc()
    if lib_ is None:
        M[:, cols] += S
        return M
    import ctypes
    S = np.ascontiguousarray(S, dtype=float)
    cols = np.ascontiguousarray(cols, dtype=np.int32)
    lib_.SOCTHCadd_cols(_ptr(M), _ptr(S), _ptr(cols), ctypes.c_int(M.shape[0]), ctypes.c_int(len(cols)),
                        ctypes.c_int(M.shape[1]))
    return M


def add_rows(M, S, rows):
    '''M[rows] += S in place (C kernel; distinct rows).'''
    lib_ = _libthc()
    if lib_ is None or not hasattr(lib_, 'SOCTHCadd_rows'):
        M[rows] += S
        return M
    import ctypes
    S = np.ascontiguousarray(S, dtype=float)
    rows = np.ascontiguousarray(rows, dtype=np.int32)
    lib_.SOCTHCadd_rows(_ptr(M), _ptr(S), _ptr(rows), ctypes.c_int(len(rows)), ctypes.c_int(M.shape[1]))
    return M


def swap02(src):
    '''(n0, n1, n2, L) -> (n2, n1, n0, L), C order (rows of L copied in parallel by the C kernel).'''
    n0, n1, n2, L = src.shape
    lib_ = _libthc()
    if lib_ is None:
        return np.ascontiguousarray(src.transpose(2, 1, 0, 3))
    import ctypes
    src = np.ascontiguousarray(src, dtype=float)
    dst = np.empty((n2, n1, n0, L))
    lib_.SOCTHCswap02(_ptr(dst), _ptr(src), ctypes.c_int(n0), ctypes.c_int(n1), ctypes.c_int(n2), ctypes.c_int(L))
    return dst


def ft_post(Pg, w):
    '''(2 nG, n) real: rows 2i, 2i+1 = w_i (Re, Im) Pg[i] (OpenMP kernel for an F-order Pg, else numpy).'''
    lib_ = _libthc()
    g, n = Pg.shape
    if lib_ is None or not hasattr(lib_, 'SOCTHCft_post') or not Pg.flags['F_CONTIGUOUS'] or Pg.dtype != complex:
        Pg = np.ascontiguousarray(Pg * w[:, None])
        return np.ascontiguousarray(Pg.view(float).reshape(g, n, 2).transpose(0, 2, 1)).reshape(2 * g, n)
    import ctypes
    out = np.empty((2 * g, n))
    w = np.ascontiguousarray(w, dtype=float)
    lib_.SOCTHCft_post(_ptr(out), _ptr(Pg), _ptr(w), ctypes.c_int(g), ctypes.c_int(n))
    return out


def gram_cols(PP, phi_blk):
    '''(PP phi_blk^T)^2 elementwise (the Gram columns of the ISDF points), squared in place by the C kernel.'''
    G = PP @ phi_blk.T
    lib_ = _libthc()
    if lib_ is None or not hasattr(lib_, 'SOCTHCsquare') or not G.flags['C_CONTIGUOUS']:
        return np.square(G, out=G)
    import ctypes
    lib_.SOCTHCsquare(_ptr(G), ctypes.c_size_t(G.size))
    return G


def _mm_split(A, B):
    '''A @ B with a real A and a possibly complex B, without promoting A.'''
    if np.iscomplexobj(B):
        return A @ np.ascontiguousarray(B.real) + 1j * (A @ np.ascontiguousarray(B.imag))
    return A @ B


def _lr_batch(L, D):
    '''sum_g L_g D L_g for a stack of real symmetric L (gb, n, n) and a (possibly complex) D, in two GEMMs.'''
    gb, n, _ = L.shape
    L2 = L.reshape(gb * n, n)
    T = _mm_split(L2, D)                                        # (gb n, n): rows of L_g D
    T = T.reshape(gb, n, n).transpose(1, 0, 2).reshape(n, gb * n)   # [L_1 D ... L_gb D]
    return _mm_split(L2.T, T.T).T                               # [L_1 D ... ] [L_1; ...; L_gb]


def half_gv(cell, mesh):
    '''G vectors of the rfftn half spectrum, (nG, 3), and the multiplicity f (1 or 2) of each.'''
    b = cell.reciprocal_vectors()
    k0 = np.fft.fftfreq(mesh[0]) * mesh[0]
    k1 = np.fft.fftfreq(mesh[1]) * mesh[1]
    k2 = np.arange(mesh[2] // 2 + 1)
    K = np.stack(np.meshgrid(k0, k1, k2, indexing='ij'), -1).reshape(-1, 3)
    f = np.full(len(K), 2.)
    f[K[:, 2] == 0] = 1.
    if mesh[2] % 2 == 0:
        f[K[:, 2] == mesh[2] // 2] = 1.
    return K @ b, f


def nyquist_mask(mesh):
    '''The Nyquist bins of the rfftn half spectrum of an even mesh (k0 = -m0/2, k1 = -m1/2, k2 = m2/2).'''
    k0 = np.fft.fftfreq(mesh[0]) * mesh[0]
    k1 = np.fft.fftfreq(mesh[1]) * mesh[1]
    k2 = np.arange(mesh[2] // 2 + 1)
    K = np.stack(np.meshgrid(k0, k1, k2, indexing='ij'), -1).reshape(-1, 3)
    m = np.zeros(len(K), dtype=bool)
    for i, n in enumerate(mesh):
        if n % 2 == 0:
            m |= np.abs(K[:, i]) == n // 2
    return m


def stc_factor(G, rc):
    '''v_TC(G) / (4 pi / G^2) = 1 - cos(|G| R_c) for the spherically truncated Coulomb kernel (1 at G = 0: the
    G = 0 term is handled separately).'''
    g = np.sqrt(np.einsum('gx,gx->g', G, G))
    f = 1 - np.cos(g * rc)
    f[g < 1e-6] = 1.
    return f


def kernels(Gv, omega):
    G2 = np.einsum('gx,gx->g', Gv, Gv)
    w_vec = np.empty(len(G2))
    w_sr = np.empty(len(G2))
    nz = G2 > 1e-12
    w_vec[nz] = 4 * np.pi / G2[nz]
    w_sr[nz] = w_vec[nz] * (1 - np.exp(-G2[nz] / (4 * omega ** 2)))
    w_vec[~nz] = w_sr[~nz] = np.pi / omega ** 2
    return w_vec, w_sr


class THCExchange:
    def __init__(self, cell, aux, alpha_c, omega=OMEGA, verbose=True, selector='cholesky', nuc_zeta=1e12,
                 local_rcut=None):
        '''selector: 'cholesky' (column-by-column pivoted Cholesky of the grid Gram matrix, O(k^2 ngrid), reference),
        'coarse' (the same on the grid coarsened by self.coarsen per direction, least-squares xi on the full grid),
        'cholesky_blocked' (blocked candidates, BLAS-3; worse points, not faster), 'kmeans'
        (weighted K-means on the grid + least-squares xi, BLAS-3; fast but poor points for all-electron smooth
        functions) or 'kmeans+chol' (K-means candidates, cand_factor per point, then the pivots of a blocked
        pivoted Cholesky among them; least-squares xi).'''
        self.cell, self.aux, self.ac, self.omega = cell, aux, alpha_c, omega
        self.selector = selector
        self.local_rcut = local_rcut   # route 1 (Angstrom): no global 3c tensor (local CX fit, GAPW J, CX LR rows)
        self.cand_factor = 4.        # kmeans+chol: K-means candidates per point
        self.coarsen = None          # coarse: grid coarsening per direction for the point selection; None: 2 if the
                                     # Cholesky factor fits half the budget, else 3 (4: EuOCl fails)
        self.blocked_pivots = True   # coarse: blocked (BLAS-3) pivoted Cholesky
        self.nuc_zeta = nuc_zeta
        self.verbose = verbose
        self.timings = {}

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def distribute(self):
        '''MPI (self.comm): after the build, keep only what this rank uses: the rows of the uniform grid it integrates
        in J (phi) and its contiguous range of LR rows (the exact CX rows; the SS rows are then built for it only).'''
        comm = getattr(self, 'comm', None)
        if comm is None or getattr(self, '_distributed', False):
            return
        rank, size = comm.Get_rank(), comm.Get_size()
        r0, r1 = self.ng * rank // size, self.ng * (rank + 1) // size
        if getattr(self, '_phi_rows', None) != (r0, r1):              # (the grid-split build has them already)
            self.phi = np.ascontiguousarray(self.phi_at(slice(r0, r1)))
        self._phi_rows = (r0, r1)
        if getattr(self, 'Llr_cx', None) is not None:
            nlr = self.Llr_cx.shape[0]
            g0, g1 = nlr * rank // size, nlr * (rank + 1) // size
            self.Llr_cx = np.ascontiguousarray(self.Llr_cx[g0:g1])
            self._lr_range = (g0, g1)
        self._distributed = True

    def phi_at(self, idx):
        '''Smooth functions at the uniform-grid points idx (a slice or an index array): from the stored phi, or
        evaluated (phi on demand).'''
        if self.phi is not None and not getattr(self, '_distributed', False) and not getattr(self, '_phi_local', False):
            return self.phi[idx]
        # pbc_eval_gto screens the lattice sum per block of BLKSIZE consecutive points, so a value depends (~1e-12)
        # on the block it falls in; the ISDF pseudo-inverse amplifies that to ~1e-6 Eh in K.  Evaluate whole blocks
        # of the full grid: values bit-identical to the full-grid evaluation, whatever the chunking.
        B, ng = BLKSIZE, self.ng
        if isinstance(idx, slice):
            g0, g1, _ = idx.indices(ng)
            a0, a1 = g0 // B * B, min(-(-g1 // B) * B, ng)
            v = self._smooth_eval(self.coords[a0:a1])
            return np.ascontiguousarray(v[g0 - a0:g1 - a0])
        idx = np.asarray(idx)
        rows = (np.unique(idx // B)[:, None] * B + np.arange(B)).ravel()
        rows = rows[rows < ng]                     # whole blocks (sorted; only the last one can be short)
        v = self._smooth_eval(self.coords[rows])
        return np.ascontiguousarray(v[np.searchsorted(rows, idx)])

    def _mem_mark(self, stage):
        '''Peak resident size (getrusage) and current size after a build stage, max over the ranks, logged with the
        budget (cell.max_memory) -- where the build exceeds it.'''
        peak = 0.                                                     # VmHWM: this process's high-water mark
        try:                                                          # (getrusage's ru_maxrss survives exec:
            for line in open('/proc/self/status'):                    # it showed the launcher's 32 GB)
                if line.startswith('VmHWM:'):
                    peak = float(line.split()[1]) / 1e3                   # MB
        except OSError:
            pass
        before = lib.current_memory()[0]
        _malloc_trim()                                                # freed C buffers back to the system
        cur = lib.current_memory()[0]
        self.log('   (%s: %.1f GB resident before malloc_trim, %.1f GB after)' % (stage, before / 1e3, cur / 1e3))
        comm = getattr(self, 'comm', None)
        if comm is not None:
            from mpi4py import MPI
            peak, cur = comm.allreduce(peak, op=MPI.MAX), comm.allreduce(cur, op=MPI.MAX)
        self.log('   memory after %s: peak %.1f GB, now %.1f GB (max over ranks; budget %.1f GB)'
                 % (stage, peak / 1e3, cur / 1e3, self.cell.max_memory / 1e3))

    def _smooth_eval(self, coords):
        '''The smooth functions at coords, (n, nao_s): evaluated on a cell of the smooth shells only (all AOs, then
        the smooth columns, held 2.4x the memory: EuOCl +6.8 GB on its uniform grid); same values per shell.'''
        if getattr(self, '_sm_cell', None) is None:
            self._sm_cell = _subcell(self.cell, self.smooth)
            assert self._sm_cell.nao == len(self.smooth)
        return self._sm_cell.pbc_eval_gto('GTOval_sph', coords)

    def avail_bytes(self):
        '''Memory left in the budget (cell.max_memory, MB) beyond the current resident size of the process.'''
        return max(0., self.cell.max_memory - lib.current_memory()[0]) * 1e6

    def fits(self, nbytes, frac=0.5):
        '''A cache of nbytes fits when it takes at most frac of the memory left in the budget.'''
        return nbytes <= frac * self.avail_bytes()

    # ---- pieces independent of N_mu
    def build_common(self, kmax_factor=20):
        cell, ac = self.cell, self.ac
        t = time.time()
        nao = self.nao = cell.nao
        aohard = np.repeat([cell.bas_exp(i).min() >= ac for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
        local = getattr(self, 'local_rcut', None)
        if local:
            # route 1: no global three-index tensor.  Nuclear terms, c0 and the exact LR rows of the CX pairs only
            # from RSDF (sr_fit False, one s aux per atom); SR metric by pbc_intor; CX pairs fitted pair-atom locally
            ti_, tj_ = np.tril_indices(nao)
            cxc = np.where(aohard[ti_] | aohard[tj_])[0]
            r = RSDF(cell, {l: [[0, [1., 1.]]] for l in cell._basis}, omega=self.omega, precision=1e-10,
                     nuc_zeta=self.nuc_zeta, lr='gspace')
            r.sr_fit = False
            r.lr_columns = cxc
            r.verbose = 0
            if getattr(self, 'exx_trunc', None):                      # LR rows: K only (J by GAPW)
                rc_ = self.exx_trunc
                r.lr_row_factor = lambda G: stc_factor(G, rc_)
            self.r = r.build()
            self.J = None
            self.Llr_cx, self.lr_cols = r.cderi, cxc                   # (nlr, nCX)
            self.nkept = 0
            auxcell = df_addons.make_auxmol(cell, self.aux)
            with auxcell.with_range_coulomb(-self.omega):
                self.M = auxcell.pbc_intor('int2c2e', hermi=1)
        else:
            r = RSDF(cell, self.aux, omega=self.omega, precision=1e-10, nuc_zeta=self.nuc_zeta, lr='gspace')
            r.keep_raw = True
            r.cderi_in_memory = True
            r.verbose = 0
            self.r = r.build()
            self.J = r.int3c_sr
            self.M = r.metric_sr
            self.nkept = r.cderi.shape[0] - r.nlr
            self.Llr = r.cderi[self.nkept:]
        self.timings['rsdf'] = time.time() - t
        self._mem_mark('rsdf')
        self.aoat = np.repeat([cell.bas_atom(i) for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
        self.smooth = np.where(~aohard)[0]
        ti, tj = np.tril_indices(nao)
        self.ti, self.tj = ti, tj
        self.cx = aohard[ti] | aohard[tj]
        self.ss = np.where(~self.cx)[0]
        self.auxmol = df_addons.make_auxmol(cell, self.aux)
        # local fit of the CX pairs (level 2)
        t = time.time()
        if local:
            from socutils.pbc.thc_local import local_cx_fit
            comm_ = getattr(self, 'comm', None)
            dist_c = comm_ is not None and comm_.Get_size() > 1 and getattr(self, 'distribute_c', True)
            self.Cs, st_ = local_cx_fit(cell, self.aux, aohard, omega=self.omega, r_cut=local, verbose=self.verbose,
                                        sparse=True, far=getattr(self, 'far_fit', True),   # far: r_cut-converged
                                        comm=comm_, rows_to_owner=dist_c)
            # MPI: C held by aux-atom owner (its rows only on a rank); the (M C) of CX-CX by broadcast blocks
            self._row_owner = st_.get('row_owner') if dist_c else None
            self.C = self.Cs.toarray() if getattr(self, 'dense_c', True) else None
            self.nonrobust = True
        else:
            self._bloch_fit()
        if self.C is not None:
            ew, U = scipy.linalg.eigh(self.M)
            self.Y = ((U * np.sqrt(np.maximum(ew, 0))) @ U.T) @ self.C
        self.timings['local fit'] = time.time() - t
        self._mem_mark('local fit')
        self._grid_and_points(kmax_factor)
        return self

    def _bloch_fit(self):
        cell, ti, tj = self.cell, self.ti, self.tj
        pat = np.repeat([self.auxmol.bas_atom(i) for i in range(self.auxmol.nbas)], np.diff(self.auxmol.ao_loc_nr()))
        C = np.zeros_like(self.J)
        pa, pb = self.aoat[ti], self.aoat[tj]
        for A in range(cell.natm):
            for B in range(A + 1):
                cols = np.where(self.cx & (((pa == A) & (pb == B)) | ((pa == B) & (pb == A))))[0]
                if len(cols) == 0:
                    continue
                D = np.where((pat == A) | (pat == B))[0]
                vd, ed = _inv_metric(self.M[np.ix_(D, D)])
                C[np.ix_(D, cols)] = (vd / ed) @ (vd.T @ self.J[np.ix_(D, cols)])
        self.C = C

    def _grid_and_points(self, kmax_factor):
        cell, nao = self.cell, self.nao
        # smooth functions on the uniform grid
        t = time.time()
        sm_cell = _subcell(cell, self.smooth)
        # the grid carries smooth PAIR densities: products of high-l shells near alpha_c (Eu: h, exponent 2.4;
        # products up to l = 10) need more than the single-function estimate at 1e-10 (EuOCl: +4.6e-4 Eh at
        # 1e-10, mesh 85x85x143; 1.4e-5 vs RSDF on 90x90x144)
        self.ke = estimate_ke_cutoff(sm_cell, getattr(self, 'ke_precision', 1e-12))
        self.mesh = cell.cutoff_to_mesh(self.ke)
        if getattr(self, 'mesh_override', None) is not None:          # tests: an explicit uniform grid
            self.mesh = np.asarray(self.mesh_override, dtype=int)
        if self.selector == 'coarse' and self.coarsen is None:
            # the coarse grid must resolve the smooth functions itself: EuOCl near its cutoff, coarsened 3x (29 points
            # along a) gave pair-fit errors 1e-6..1e-5 jumping with the mesh (E +-2e-4, one mesh off by 2.7e-2 Eh);
            # 2x gives 2.6e-8..3.6e-8 for every mesh.  The selection holds phi and the pivoted-Cholesky factor
            # (kmax rows) on the coarse points, per rank its share under MPI (all of them for the orbit selection):
            # the finest coarsening whose share fits half of every rank's budget (cell.max_memory)
            comm_ = getattr(self, 'comm', None)
            size_ = comm_.Get_size() if comm_ is not None else 1
            avail = self.avail_bytes()
            if comm_ is not None:
                avail = comm_.allreduce(avail, op=__import__('mpi4py').MPI.MIN)
            kmx = int(getattr(self, 'kmax_factor', kmax_factor) * len(self.smooth))
            if getattr(self, 'symm_grid', None) is None:               # decided here: it sets the selection mode
                self._ops = space_group_ops(cell)
                self.symm_grid = len(self._ops[0]) > 1
            orbit_ = self.symm_grid and getattr(self, 'symm_pivots', True)
            serial = size_ == 1 or orbit_ or not self.blocked_pivots    # orbits: rank 0 alone
            # whole-orbit selection: validated on the 3x grid (quartz -4e-6 vs exact; on the 2x grid +2.7e-4), and
            # the symmetric mesh makes the coarse grid an exact sublattice anyway.  Plain selection: 2x (EuOCl
            # meshes 85/88/91: E -4.9/-3.4/-5.0e-5 vs RSDF; 3x: +4.6e-4/-2.7e-2/+1.7e-4)
            for c in ((3, 4) if orbit_ else (2, 3, 4)):
                npts = np.prod(-(-np.asarray(self.mesh) // c))
                need = (kmx + 64 + len(self.smooth)) * (npts if serial else -(-npts // size_)) * 8
                if need < 0.5 * avail:
                    break
            else:
                raise MemoryError('THC point selection: %.0f GB per rank even on the 4x coarser grid, budget %.0f GB '
                                  '(cell.max_memory); use more MPI ranks or a larger max_memory'
                                  % (need / 1e9, avail / 1e9))
            self.coarsen = c
            self.log('   point selection on the grid coarsened %dx (%.1f GB per rank)' % (c, need / 1e9))
        if getattr(self, 'symm_grid', None) is None:                   # default: on when the cell has symmetry
            self._ops = space_group_ops(cell)
            self.symm_grid = len(self._ops[0]) > 1
        if self.symm_grid:
            # a uniform grid (and its coarse grid) mapped onto itself by the space group, pivots in whole orbits:
            # the THC factors keep the crystal symmetry (otherwise broken by the grid and the greedy point choice;
            # EuOCl prim: W_y 128.6 -> 0.46 = RSDF's 0.45, W_z -2975 -> -2959 vs RSDF -2957)
            if getattr(self, '_ops', None) is None:
                self._ops = space_group_ops(cell)
            m0 = self.mesh
            self.mesh = symmetric_mesh(m0, self._ops[:2], self.coarsen if self.selector == 'coarse' else 1)
            self.log('   space group %s (%d ops): mesh %s -> %s' % (self._ops[2], len(self._ops[0]), list(m0),
                                                                    list(self.mesh)))
        coords = self.coords = cell.get_uniform_grids(self.mesh)
        self.ng = len(coords)
        # the smooth functions on the uniform grid: stored, or evaluated per grid block when they do not fit the
        # budget (phi_ondemand True / False forces; then only the streamed THC build is available)
        ondemand = getattr(self, 'phi_ondemand', None)
        comm = getattr(self, 'comm', None)
        if ondemand is None:
            # MPI + streamed build: always distributed by grid rows (no full copy per rank); serial: if it fits
            ondemand = ((comm is not None and comm.Get_size() > 1 and getattr(self, 'stream', False))
                        or self.ng * len(self.smooth) * 8 > 0.3 * self.avail_bytes())
        self.phi_ondemand = bool(ondemand)
        if self.phi_ondemand:
            self.phi = None
            self.log('   phi on demand (%.1f GB not stored)' % (self.ng * len(self.smooth) * 8 / 1e9))
        else:
            self.phi = self._smooth_eval(coords)                      # (ng, nao_s)
        self.timings['grid'] = time.time() - t
        self._mem_mark('grid')
        t = time.time()
        if self.selector == 'coarse':
            # pivoted Cholesky on a grid coarsened by coarsen per direction, pivots snapped to the nearest point of
            # the full grid (the smooth pair densities are band-limited; xi is fitted on the full grid)
            mc = np.maximum(-(-np.asarray(self.mesh) // self.coarsen), 1)   # rounded up: never coarser
            cc = cell.get_uniform_grids(mc)
            comm = getattr(self, 'comm', None)
            rank = comm.Get_rank() if comm is not None else 0
            orbit_sel = getattr(self, 'symm_grid', False) and getattr(self, 'symm_pivots', True)
            dist = comm is not None and comm.Get_size() > 1 and self.blocked_pivots and not orbit_sel
            # serial selection: phi on the whole coarse grid on rank 0 only; distributed: per rank below
            phic = self._smooth_eval(cc) if (not dist and rank == 0) else None
            f = pivots_blocked if self.blocked_pivots else pivots
            if getattr(self, 'symm_grid', False) and getattr(self, 'symm_pivots', True):
                orb, mem = grid_orbits(mc, self._ops[:2])
                f = lambda ph, k_: pivots_orbits(ph, k_, orb, mem, orbit_dep=getattr(self, "orbit_dep", 1e-1))
            kmax = min(len(cc), int(getattr(self, 'kmax_factor', kmax_factor) * len(self.smooth)))
            pf = getattr(self, 'piv_file', None)                      # cache of the coarse pivots (restarts)
            comm = getattr(self, 'comm', None)
            rank = comm.Get_rank() if comm is not None else 0
            if pf and os.path.exists(pf) and len(np.load(pf)) >= kmax:
                pc = np.load(pf)[:kmax]
            elif dist:                                                # rows of the coarse grid per rank
                ncg = len(cc)
                q0, q1 = ncg * rank // comm.Get_size(), ncg * (rank + 1) // comm.Get_size()
                phic = None
                ph_loc = self._smooth_eval(cc[q0:q1])
                pc = pivots_blocked_dist(ph_loc, q0, ncg, kmax, comm)
                ph_loc = None
                if pf and rank == 0:
                    np.save(pf, pc)
            elif rank == 0:
                pc, _ = f(phic, kmax)
                if pf:
                    np.save(pf, pc)
            if comm is not None and not dist:                         # one point set for every rank
                pc = comm.bcast(pc if rank == 0 else None, root=0)
            ijk = np.stack(np.unravel_index(pc, mc), -1)
            fine = np.rint(ijk * (np.asarray(self.mesh) / mc)).astype(int) % np.asarray(self.mesh)
            fidx = np.ravel_multi_index(fine.T, self.mesh)
            _, first = np.unique(fidx, return_index=True)
            self.piv, self.L = fidx[np.sort(first)], None
            self.coarse_mesh = mc
        elif not self.selector.startswith('cholesky'):
            self.piv, self.L = np.zeros(0, dtype=int), None
        else:
            f = pivots_blocked if self.selector == 'cholesky_blocked' else pivots
            self.piv, self.L = f(self.phi, min(self.ng, kmax_factor * len(self.smooth)))
        self.timings['pivots'] = time.time() - t
        self._mem_mark('pivots')
        self.Gv, self.fG = half_gv(cell, self.mesh)
        self.w_vec, self.w_sr = kernels(self.Gv, self.omega)
        # exchange kernel w_k (V, A_lr, the LR rows): Coulomb (w_vec), or with exx_trunc = R_c (bohr) the smoothed
        # truncated Coulomb  w_SR + v_TC(G) exp(-G^2 / 4 w^2),  v_TC = 4 pi (1 - cos G R_c) / G^2  (sphere; Yang,
        # Bakry, Ye, arXiv:2609.00203): the LR weights times 1 - cos(G R_c) at G != 0; G = 0 as w_vec, its TC part
        # 2 pi R_c^2 / V S D S added by the mean field (spinor_scf exxdiv 'stc').  J keeps w_vec.
        self.w_k = self.w_vec
        if getattr(self, 'exx_trunc', None):
            self.w_k = self.w_sr + (self.w_vec - self.w_sr) * stc_factor(self.Gv, self.exx_trunc)
        if getattr(self, 'g_sphere', False):
            # the smooth pair densities live inside |G|^2 <= 2 ke; beyond it (the corners of the mesh box) the
            # ISDF vectors carry only fit error, which the analytic FT of a compact aux function does not damp:
            # A (and V) summed over the sphere only (EuOCl: A-terms of E_K moved 8e-3 Eh between meshes 88, 89)
            out = np.einsum('gx,gx->g', self.Gv, self.Gv) > 2 * self.ke * (1 + 1e-12)
            self.fG = np.where(out, 0., self.fG)
            self.w_k = np.where(out, 0., self.w_k)
            self.log('   G sphere: %d of %d half-spectrum G beyond 2 ke = %.0f dropped from A, V'
                     % (out.sum(), len(out), 2 * self.ke))
        if getattr(self, 'drop_nyquist', False):
            # even mesh: the k = -m/2 bin stands for +-G_N; pairing it with the analytic FT of a compact aux
            # function at -G_N only is ambiguous, so the Nyquist rows / planes are left out of A and V
            nyq = nyquist_mask(self.mesh)
            self.fG = np.where(nyq, 0., self.fG)
            self.w_k = np.where(nyq, 0., self.w_k)
        self.log('   THC common: nao %d (smooth %d), CX pairs %d of %d, ke %.0f Eh, mesh %s = %d points, %d pivots; '
                 'RSDF %.0f s, local fit %.0f s, grid %.0f s, pivots %.0f s'
                 % (nao, len(self.smooth), self.cx.sum(), len(self.cx), self.ke, self.mesh, self.ng, len(self.piv),
                    self.timings['rsdf'], self.timings['local fit'], self.timings['grid'], self.timings['pivots']))

    # ---- THC factors for N_mu points
    def _build_thc_stream(self, pts, cut=None, nb=None, gchunk=None, rchunk=None):
        '''V, A, A_lr and the LR rows Xi of the ISDF vectors without storing xi (N_mu x ngrid). Blocks b of mu:
        xi_b = G+[b] Gc with the Gram columns Gc[P, r] = (phi(r_P) . phi(r))^2 regenerated in grid chunks, its FFT
        gives A[:, b], A_lr[:, b] and Xi[:, b] as in the full build, and V[b] = int (v * xi_b) Gc^T G+ (one G+ on
        the right of an explicitly formed xi_b; V = G+ Z G+ from Z = Gc v Gc^T loses ~1e-3 Eh to the conditioning).'''
        vol, ng, mesh = self.cell.vol, self.ng, self.mesh
        hshape = (mesh[0], mesh[1], mesh[2] // 2 + 1)
        PP = self.phi_at(pts)
        nmu = len(pts)
        if cut is None:                                               # relative eigenvalue cut of the Gram pseudo-inverse
            cut = getattr(self, 'isdf_cut', 1e-10)                    # 1e-13 amplifies BLAS rounding (EuOCl: 7e-5 Eh)
        comm = getattr(self, 'comm', None)                            # MPI: mu blocks round-robin, Allreduce
        rank, size = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
        e, v = scipy.linalg.eigh((PP @ PP.T) ** 2)
        keep = e > cut * e.max()
        Ginv = (v[:, keep] / e[keep]) @ v[:, keep].T
        if comm is not None:                                          # rank 0's pseudo-inverse on every rank
            Ginv = np.ascontiguousarray(Ginv)
            comm.Bcast(Ginv, root=0)
            keep = comm.bcast(keep, root=0)
        # MPI with phi not stored: phi distributed by grid rows (this rank's rows evaluated once); every mu block is
        # shared by all ranks, xi_b formed on the own rows, transposed (Alltoallv) to own mu rows on the full grid
        # for the FFT / A, and the potential transposed back for V
        grid_split = comm is not None and size > 1 and self.phi is None
        if grid_split:
            r0, r1 = ng * rank // size, ng * (rank + 1) // size
            ta = time.time()
            phi_loc = self.phi_at(slice(r0, r1))
            self.phi, self._phi_rows, self._phi_local = phi_loc, (r0, r1), True
            phi_loc = None
            self.log('   streamed THC: phi distributed by grid rows (%.1f GB per rank, %.0f s)'
                     % (self.phi.nbytes / 1e9, time.time() - ta))
        # sizes from the memory budget (cell.max_memory): grid chunks of the Gram columns (nmu, rchunk), G chunks
        # of the aux FT (complex + its real layout, 32 gchunk naux bytes), mu blocks (two (nb, ng) arrays live;
        # grid split: ~6 (nb, ng / size) arrays)
        avail = self.avail_bytes()
        naux = self.auxmol.nao
        if rchunk is None:
            rchunk = int(min(100000, max(2000, 0.08 * avail / (8 * nmu * 2))))
        if gchunk is None:
            gchunk = int(min(20000, max(500, 0.08 * avail / (32 * naux))))
        if nb is None:
            if getattr(self, 'stream_gb', None):
                nb = int(self.stream_gb * 1e9 / (8 * ng * 3))
            elif grid_split:
                # live at once per rank: 3 arrays of nb x (ng / size) doubles (xl / rbuf / xi, then Xh / sbuf / Xq,
                # then vxi / sbuf / vl), 4 with margin; the aux FT cache takes 0.2 of the budget apart
                nb = int((0.65 * avail - 8 * nmu * rchunk * 2 - 32 * gchunk * naux) * size / (8 * ng * 4))
            else:
                # serial stream: xi, its spectrum and the potential with the FFT work space, measured ~3.3
                # (nb, ng) doubles at once (EuOCl: 16.7 GB above the start for a planned 11.3 GB at 2.2)
                nb = int((0.7 * avail - 8 * nmu * rchunk * 2 - 32 * gchunk * naux) / (8 * ng * 3.3))
            if grid_split:                                            # Alltoallv counts / displacements < 2^31
                nb = min(nb, size * ((2 ** 31 - 1) // ng))
            nb = max(32, min(nmu, nb))
        if comm is not None:                                          # the same blocks on every rank, >= 1 per rank
            nb, rchunk, gchunk = (comm.allreduce(x, op=__import__('mpi4py').MPI.MIN) for x in (nb, rchunk, gchunk))
            if not grid_split:
                nb = min(nb, -(-nmu // size))
        self.log('   streamed THC: %d mu blocks of %d (budget %.0f GB, available %.0f GB), grid chunk %d, G chunk %d'
                 % (-(-nmu // nb), nb, self.cell.max_memory / 1e3, avail / 1e9, rchunk, gchunk))
        V = np.zeros((nmu, nmu))
        A = np.zeros((self.auxmol.nao, nmu))
        A_lr = np.zeros((self.auxmol.nao, nmu))
        sw = self.fG * self.w_sr / vol
        swl = self.fG * (self.w_k - self.w_sr) / vol
        lrg = np.where(np.abs(swl) > 1e-14 * np.abs(swl).max())[0]       # the LR kernel lives at small G only
        Pl = ft_ao(self.auxmol, self.Gv[lrg]) * swl[lrg, None]           # (nG_lr, naux)
        Plr, Pli = np.ascontiguousarray(Pl.real.T), np.ascontiguousarray(Pl.imag.T)
        Pl = None
        local = getattr(self, 'local_rcut', None)
        if local:
            Xi = np.zeros((2 * len(self.r.lr_G), nmu))                 # zeros: summed over the ranks
        # G chunks in the rfft order (contiguous columns of the block spectra); aux shells pruned per chunk by its
        # smallest |G|^2
        G2 = np.einsum('gx,gx->g', self.Gv, self.Gv)
        g2c = np.array([G2[g0:g0 + gchunk].min() for g0 in range(0, len(G2), gchunk)])
        ctx = (sw, g2c, gchunk, lrg, Plr, Pli)
        t0 = time.time()
        tt = dict.fromkeys(('xi', 'FFT', 'ft_ao', 'A GEMM', 'V'), 0.)
        if grid_split:
            self._stream_grid_blocks(PP, Ginv, nb, rchunk, ctx, V, A, A_lr, Xi if local else None, tt, t0)
        for ib, b0 in enumerate(range(0, nmu if not grid_split else 0, nb)):
            if ib >= getattr(self, 'stream_max_blocks', 1 << 30):     # timing runs: stop after a few blocks
                self.log('   streamed THC: stopped after %d blocks (timing run; V, A incomplete)' % ib)
                break
            if ib % size != rank:
                continue
            b1 = min(b0 + nb, nmu)
            ta = time.time()
            xi = np.empty((b1 - b0, ng))
            for g0 in range(0, ng, rchunk):
                g1 = min(g0 + rchunk, ng)
                xi[:, g0:g1] = Ginv[b0:b1] @ gram_cols(PP, self.phi_at(slice(g0, g1)))
            tt['xi'] += time.time() - ta
            xi = self._block_spectra([xi], b0, b1, ctx, A, A_lr, Xi if local else None, tt)
            ta = time.time()
            Vb = np.zeros((b1 - b0, nmu))
            for g0 in range(0, ng, rchunk):
                g1 = min(g0 + rchunk, ng)
                Vb += xi[:, g0:g1] @ gram_cols(PP, self.phi_at(slice(g0, g1))).T
            V[b0:b1] = Vb @ Ginv                                        # int (v * xi_b) xi^T (Parseval)
            tt['V'] += time.time() - ta
            xi = Vb = None
            self.log('   streamed THC: mu block %d-%d of %d, %.0f s (%s)'
                     % (b0, b1, nmu, time.time() - t0, ', '.join('%s %.0f' % kv for kv in tt.items())))
        if comm is not None:                                          # blocks of every rank -> full V, A, A_lr, Xi
            from mpi4py import MPI
            ta = time.time()
            arrs = [V, A, A_lr] + ([Xi] if local else [])
            for X_ in arrs:
                comm.Allreduce(MPI.IN_PLACE, X_, op=MPI.SUM)
            self.log('   streamed THC: Allreduce of V, A, A_lr, Xi %.0f s' % (time.time() - ta))
        self.V = .5 * (V + V.T)
        self.A, self.A_lr = A, A_lr
        if local:
            self.Xi_lr = Xi
        return int(keep.sum())

    def _stream_grid_blocks(self, PP, Ginv, nb, rchunk, ctx, V, A, A_lr, Xi, tt, t0):
        '''Streamed V, A, A_lr, Xi with phi distributed by grid rows (self.phi: rows self._phi_rows): all ranks take
        every mu block b together; xi_b on the own grid rows -> Alltoallv -> own mu rows of b on the full grid ->
        FFT, A, A_lr, Xi columns, potential -> Alltoallv back -> own-row part of V[b] (summed by the caller).'''
        from mpi4py import MPI
        comm = self.comm
        rank, size = comm.Get_rank(), comm.Get_size()
        ng, nmu = self.ng, len(PP)
        gs = [ng * p // size for p in range(size + 1)]                # grid rows of every rank
        nr = gs[rank + 1] - gs[rank]
        phi = self.phi
        vol, mesh = self.cell.vol, self.mesh
        hshape = (mesh[0], mesh[1], mesh[2] // 2 + 1)
        sw, g2c, gchunk, lrg, Plr, Pli = ctx
        nG = len(self.Gv)
        nch = -(-nG // gchunk)
        # G chunks of every rank: contiguous ranges balanced by the cost of the A GEMM / aux FT of a chunk
        # (aux functions kept at its |G| x its G points); contiguous columns pack by memcpy (round-robin chunks
        # needed a fancy-index gather of the spectra: EuOCl np4 37 of 56 s per block)
        cost = np.empty(nch)
        for c in range(nch):
            sub_, nk_ = self._aux_upto(g2c[c])
            cost[c] = max(nk_, 1) * (min((c + 1) * gchunk, nG) - c * gchunk)
        cum = np.concatenate([[0.], np.cumsum(cost)])
        cuts = np.searchsorted(cum, cum[-1] * np.arange(1, size) / size)
        cuts = np.concatenate([[0], np.clip(cuts, 0, nch), [nch]]).astype(int)
        cuts = np.maximum.accumulate(cuts)
        own = list(range(cuts[rank], cuts[rank + 1]))
        gsl = [(cuts[q] * gchunk, min(cuts[q + 1] * gchunk, nG)) for q in range(size)]   # G columns of rank q
        ncol = [b_ - a_ for a_, b_ in gsl]
        p2cache = {}                                                  # aux FT of the own G chunks, kept if it fits
        p2_left = 0.2 * self.avail_bytes()                            # its share of the budget (blocks: 0.65)
        ta_comm = 0.
        tw = dict(wait=0., a2a=0., bytes=0.)                          # transpose: waiting, Alltoallv, bytes moved
        for b0 in range(0, nmu, nb):
            b1 = min(b0 + nb, nmu)
            nbk = b1 - b0
            ms = [nbk * q // size for q in range(size + 1)]           # mu rows (within the block) of every rank
            m0, m1 = ms[rank], ms[rank + 1]
            ta = time.time()
            xl = np.empty((nbk, nr))
            for g0 in range(0, nr, rchunk):
                g1 = min(g0 + rchunk, nr)
                xl[:, g0:g1] = Ginv[b0:b1] @ gram_cols(PP, phi[g0:g1])
            tt['xi'] += time.time() - ta
            # forward: to rank q the rows ms[q]:ms[q+1] of xl (contiguous); from rank p its (m1 - m0, n_p) piece
            ta = time.time()
            comm.Barrier()                                            # wait for the slowest rank (timed apart)
            tw['wait'] += time.time() - ta
            ta = time.time()
            sc = np.array([(ms[q + 1] - ms[q]) * nr for q in range(size)])
            rc = np.array([(m1 - m0) * (gs[p + 1] - gs[p]) for p in range(size)])
            rbuf = np.empty(rc.sum())
            tb = time.time()
            comm.Alltoallv([xl, (sc, np.r_[0, np.cumsum(sc)[:-1]]), MPI.DOUBLE],
                           [rbuf, (rc, np.r_[0, np.cumsum(rc)[:-1]]), MPI.DOUBLE])
            tw['a2a'] += time.time() - tb
            tw['bytes'] += 8. * (sc.sum() + rc.sum())
            xl = None
            xi = np.empty((m1 - m0, ng))
            off = 0
            for p in range(size):
                n_p = gs[p + 1] - gs[p]
                xi[:, gs[p]:gs[p + 1]] = rbuf[off:off + (m1 - m0) * n_p].reshape(m1 - m0, n_p)
                off += (m1 - m0) * n_p
            rbuf = None
            ta_comm += time.time() - ta
            # FFT of the own mu rows; A_lr, Xi (small G) from them
            ta = time.time()
            if m1 > m0:
                Xh = scipy.fft.rfftn(xi.reshape(m1 - m0, *mesh), axes=(1, 2, 3), workers=-1).reshape(m1 - m0, -1)
            else:                                                     # no mu row here (a short last block)
                Xh = np.zeros((0, nG), dtype=complex)
            xi = None
            Xh *= vol / ng
            tt['FFT'] += time.time() - ta
            if m1 > m0:
                Xc = Xh[:, lrg]
                A_lr[:, b0 + m0:b0 + m1] = Plr @ Xc.real.T + Pli @ Xc.imag.T
                Xc = None
                if Xi is not None:
                    Xi[:, b0 + m0:b0 + m1] = self._lr_rows_of_xi(Xh)
            # A: transpose the spectra to all mu rows of the block on the own G chunks, A[:, b] += Re(P^* X) there
            ta = time.time()
            comm.Barrier()
            tw['wait'] += time.time() - ta
            ta = time.time()
            sg = np.array([2 * (m1 - m0) * ncol[q] for q in range(size)])
            rg = np.array([2 * (ms[p + 1] - ms[p]) * ncol[rank] for p in range(size)])
            sbuf = np.empty(sg.sum())
            o_ = 0
            for q in range(size):                                     # contiguous column slices: memcpy
                a_, b_ = gsl[q]
                sbuf[o_:o_ + sg[q]].view(complex).reshape(m1 - m0, ncol[q])[:] = Xh[:, a_:b_]
                o_ += sg[q]
            Xq = np.empty((nbk, ncol[rank]), dtype=complex)
            tb = time.time()
            comm.Alltoallv([sbuf, (sg, np.r_[0, np.cumsum(sg)[:-1]]), MPI.DOUBLE],
                           [Xq.view(float), (rg, np.r_[0, np.cumsum(rg)[:-1]]), MPI.DOUBLE])
            tw['a2a'] += time.time() - tb
            tw['bytes'] += 8. * (sg.sum() + rg.sum())
            sbuf = None
            ta_comm += time.time() - ta
            As = np.zeros((self.auxmol.nao, nbk))                     # aux-cutoff order
            off = 0
            for c in own:
                g0, g1 = c * gchunk, min((c + 1) * gchunk, nG)
                ta = time.time()
                sub, nk = self._aux_upto(g2c[c])
                if sub is None:
                    off += g1 - g0
                    continue
                P2 = p2cache.get(c)
                if P2 is None:
                    P2 = ft_post(ft_ao(sub, self.Gv[g0:g1]), sw[g0:g1])     # (2 nG, nk)
                    if P2.nbytes < p2_left and b1 < nmu:                # reused by the later blocks
                        p2cache[c] = P2
                        p2_left -= P2.nbytes
                tt['ft_ao'] += time.time() - ta
                ta = time.time()
                X2 = np.ascontiguousarray(Xq[:, off:off + g1 - g0]).view(float)
                cc = As[:nk].T
                r = blas.dgemm(1., X2.T, P2.T, beta=1., c=cc, trans_a=1, trans_b=1, overwrite_c=1)
                if not np.shares_memory(r, As):
                    As[:nk] = r.T
                off += g1 - g0
                tt['A GEMM'] += time.time() - ta
            A[self._aux_perm, b0:b1] += As
            As = Xq = X2 = P2 = None
            # potential of the own mu rows on the full grid
            ta = time.time()
            Xh *= self.w_k[None]
            if m1 > m0:
                vxi = scipy.fft.irfftn(Xh.reshape((m1 - m0,) + hshape), s=tuple(mesh), axes=(1, 2, 3),
                                       workers=-1, overwrite_x=True).reshape(m1 - m0, -1)
            else:
                vxi = np.zeros((0, ng))
            Xh = None
            tt['V'] += time.time() - ta
            # back: to rank p the (m1 - m0, n_p) columns of its rows; from rank q its mu rows on the own grid rows
            ta = time.time()
            comm.Barrier()
            tw['wait'] += time.time() - ta
            ta = time.time()
            sbuf = np.empty(rc.sum())
            o_ = 0
            for p in range(size):                                     # one copy into the send buffer
                n_p = gs[p + 1] - gs[p]
                sbuf[o_:o_ + rc[p]].reshape(m1 - m0, n_p)[:] = vxi[:, gs[p]:gs[p + 1]]
                o_ += rc[p]
            vxi = None
            vl = np.empty((nbk, nr))
            tb = time.time()
            comm.Alltoallv([sbuf, (rc, np.r_[0, np.cumsum(rc)[:-1]]), MPI.DOUBLE],
                           [vl, (sc, np.r_[0, np.cumsum(sc)[:-1]]), MPI.DOUBLE])
            tw['a2a'] += time.time() - tb
            tw['bytes'] += 8. * (sc.sum() + rc.sum())
            sbuf = None
            ta_comm += time.time() - ta
            ta = time.time()
            Vb = np.zeros((nbk, nmu))
            for g0 in range(0, nr, rchunk):
                g1 = min(g0 + rchunk, nr)
                Vb += vl[:, g0:g1] @ gram_cols(PP, phi[g0:g1]).T
            V[b0:b1] += Vb @ Ginv                                       # own-row part of int (v * xi_b) xi^T
            vl = Vb = None
            tt['V'] += time.time() - ta
            self.log('   streamed THC (grid split): mu block %d-%d of %d, %.0f s (%s, transpose %.0f = wait %.0f + '
                     'Alltoallv %.0f (%.2f GB/s) + pack; aux FT cached for %d of %d own G chunks)'
                     % (b0, b1, nmu, time.time() - t0, ', '.join('%s %.0f' % kv for kv in tt.items()), ta_comm,
                        tw['wait'], tw['a2a'], tw['bytes'] / max(tw['a2a'], 1e-9) / 1e9, len(p2cache), len(own)))
        self._log_rank_spread(tt, tw)
        self._mem_mark('V, A stream')

    def _log_rank_spread(self, tt, tw):
        '''min / max over the ranks of the stream timings (load balance: the transpose waits for the slowest).'''
        comm = self.comm
        keys = list(tt) + ['wait', 'a2a']
        loc = np.array([tt[k] for k in tt] + [tw['wait'], tw['a2a']])
        allv = np.array(comm.allgather(loc))
        import socket
        hosts = comm.allgather(socket.gethostname())
        self.log('   streamed THC (grid split), per rank min / max: ' + ', '.join(
            '%s %.0f / %.0f' % (k, allv[:, i].min(), allv[:, i].max()) for i, k in enumerate(keys)))
        for r, (h, v) in enumerate(zip(hosts, allv)):                 # slow node or imbalance
            self.log('     rank %d (%s): %s' % (r, h, ', '.join('%s %.0f' % (k, x) for k, x in zip(keys, v))))

    def _block_spectra(self, xs, b0, b1, ctx, A, A_lr, Xi, tt):
        '''xs = [xi], xi (b1 - b0, ng): ISDF vectors b0:b1 on the full grid (taken out of the list, so that it is
        freed after its FFT).  Their FFT gives the columns b0:b1 of A, A_lr (and of the LR rows Xi); returns the
        potential (v * xi)(r) (b1 - b0, ng).'''
        vol, ng, mesh = self.cell.vol, self.ng, self.mesh
        hshape = (mesh[0], mesh[1], mesh[2] // 2 + 1)
        sw, g2c, gchunk, lrg, Plr, Pli = ctx
        n = b1 - b0
        ta = time.time()
        xi = xs.pop()
        As = np.zeros((self.auxmol.nao, n))                           # A[:, b] in aux-cutoff order
        Xh = scipy.fft.rfftn(xi.reshape(n, *mesh), axes=(1, 2, 3), workers=-1).reshape(n, -1)
        xi = None
        Xh *= vol / ng                                                  # xi(G) = int_cell xi(r) e^{-iGr}
        tt['FFT'] += time.time() - ta
        for g0 in range(0, len(self.Gv), gchunk):
            g1 = min(g0 + gchunk, len(self.Gv))
            ta = time.time()
            sub, nk = self._aux_upto(g2c[g0 // gchunk])                   # the first nk aux functions (cutoff order)
            if sub is None:
                continue
            # Re(P^* X) = Pr Xr + Pi Xi as one real GEMM over (G, re/im), accumulated in place (beta = 1) into
            # the aux-cutoff-ordered A[:, b] (naux, nb), C order: its first nk rows are contiguous; the weighted
            # re/im layout of the F-order ft_ao block in one pass (C kernel)
            P2 = ft_post(ft_ao(sub, self.Gv[g0:g1]), sw[g0:g1])              # (2 nG, nk)
            tt['ft_ao'] += time.time() - ta
            ta = time.time()
            X2 = np.ascontiguousarray(Xh[:, g0:g1]).view(float)         # (nb, 2 nG), contiguous copy of the chunk
            c = As[:nk].T                                               # column-major (nb, nk) view
            r = blas.dgemm(1., X2.T, P2.T, beta=1., c=c, trans_a=1, trans_b=1, overwrite_c=1)
            if not np.shares_memory(r, As):
                As[:nk] = r.T
            tt['A GEMM'] += time.time() - ta
        A[self._aux_perm, b0:b1] = As
        As = None
        Xc = Xh[:, lrg]
        A_lr[:, b0:b1] = Plr @ Xc.real.T + Pli @ Xc.imag.T
        Xc = None
        if Xi is not None:
            Xi[:, b0:b1] = self._lr_rows_of_xi(Xh)
        # potential of the block in real space, (v * xi_b)(r) = irfft(w_vec xi_b(G)) ng / vol
        ta = time.time()
        Xh *= self.w_k[None]
        vxi = scipy.fft.irfftn(Xh.reshape((n,) + hshape), s=tuple(mesh), axes=(1, 2, 3),
                               workers=-1, overwrite_x=True).reshape(n, -1)
        tt['V'] += time.time() - ta
        return vxi

    def _aux_upto(self, g2min, eps=1e-15):
        '''The aux shells whose FT is not negligible at |G|^2 >= g2min (exp(-G^2 / 4 alpha) > eps for the smallest
        exponent of the shell): a prefix of the shells sorted by that cutoff.  Returns the sub-molecule of the prefix
        and its number of functions nk; the functions of the sorted shells are self._aux_perm (aux AO indices), so
        the sub-molecule's functions are aux AOs _aux_perm[:nk].'''
        am = self.auxmol
        if getattr(self, '_aux_cut', None) is None:
            amin = np.array([am.bas_exp(i).min() for i in range(am.nbas)])
            cut = 4 * amin * np.log(1 / eps)                            # |G|^2 beyond which shell i is negligible
            order = np.argsort(-cut, kind='stable')
            loc = am.ao_loc_nr()
            self._aux_perm = np.concatenate([np.arange(loc[i], loc[i + 1]) for i in order])
            self._aux_cut = (cut[order], order, {})
        cut, order, cache = self._aux_cut
        if not getattr(self, 'ft_prune', True):
            k = am.nbas
        else:
            k = int(np.searchsorted(-cut, -g2min, side='right'))       # shells with cut >= g2min (a prefix)
        if k == 0:
            return None, 0
        if k not in cache:
            sub = am.copy(deep=False)
            sub._bas = am._bas[order[:k]]                               # shells in cutoff order
            cache.clear()                                               # keep one (consecutive chunks share it)
            cache[k] = (sub, int(sub.nao_nr()))
        return cache[k]

    def build_thc(self, nmu):
        if getattr(self, 'stream', False):
            t = time.time()
            if getattr(self, 'symm_grid', False):                    # whole orbits (the selection ends on one)
                nmu = max(nmu, len(self.piv)) if len(self.piv) <= nmu + 64 else nmu
            pts = self.piv[:nmu]
            rank = self._build_thc_stream(pts)
            self.pts = pts
            self.phimu = self.phi_at(pts)
            self.pair_err = (np.nan, np.nan)
            self.timings.update({'V, A (streamed)': time.time() - t})
            return rank
        t = time.time()
        if self.selector in ('kmeans', 'kmeans+chol'):
            weight = np.einsum('ri,ri->r', self.phi, self.phi) ** 2
            ncand = nmu if self.selector == 'kmeans' else min(self.ng, int(self.cand_factor * nmu))
            pts, nit = kmeans_points(self.cell, self.coords, weight, ncand)
            self.timings['kmeans (%d it)' % nit] = time.time() - t
            if self.selector == 'kmeans+chol':
                t = time.time()
                pts, crank = candidate_cholesky_points(self.phi, pts, nmu)
                self.timings['candidate Cholesky (%d candidates, rank %d)' % (ncand, crank)] = time.time() - t
            t = time.time()
            xi, rank = isdf_xi_ls(self.phi, pts, cut=getattr(self, 'isdf_cut', 1e-10))
            nmu = len(pts)
        elif self.selector == 'coarse':
            pts = self.piv[:nmu]
            xi, rank = isdf_xi_ls(self.phi, pts, cut=getattr(self, 'isdf_cut', 1e-10))
        else:
            pts = self.piv[:nmu]
            xi, rank = isdf_xi(self.L, self.piv, nmu)
        t_xi = time.time() - t
        # per-pair interpolation error |rho - sum_mu xi_mu rho(r_mu)| / |rho| on the grid, for all pairs of 16
        # sampled first functions, dropping pairs below 1e-3 of the largest pair sharing that first function
        rng = np.random.default_rng(0)
        errs = []
        for a in rng.choice(len(self.smooth), min(16, len(self.smooth)), replace=False):
            rho = self.phi[:, a:a + 1] * self.phi                  # (ng, nao_s)
            nr = np.linalg.norm(rho, axis=0)
            keep = nr >= 1e-3 * nr.max()
            e = np.linalg.norm(rho[:, keep] - xi.T @ rho[pts][:, keep], axis=0) / nr[keep]
            errs.append(e)
        errs = np.concatenate(errs)
        self.pair_err = (errs.max(), np.median(errs))
        rho = None
        t = time.time()
        vol, ng = self.cell.vol, self.ng
        Xh = scipy.fft.rfftn(xi.reshape(nmu, *self.mesh), axes=(1, 2, 3), workers=-1).reshape(nmu, -1)
        Xh *= vol / ng                                              # xi(G) = int_cell xi(r) e^{-iGr}
        xi = None
        s = np.sqrt(self.fG * self.w_k / vol)
        W = Xh * s
        V = W.real @ W.real.T + W.imag @ W.imag.T                   # (xi_mu | xi_nu) with w_vec
        W = None
        # A_P,mu = (P|xi_mu)_SR, analytic FT of the aux functions, G blocks
        A = np.zeros((self.auxmol.nao, nmu))
        A_lr = np.zeros((self.auxmol.nao, nmu))
        sw = self.fG * self.w_sr / vol
        swl = self.fG * (self.w_k - self.w_sr) / vol                  # LR (G != 0; the G = 0 terms cancel)
        for g0 in range(0, len(self.Gv), 20000):
            g1 = min(g0 + 20000, len(self.Gv))
            Pg = ft_ao(self.auxmol, self.Gv[g0:g1])                 # (nG, naux)
            A += ((Pg.conj() * sw[g0:g1, None]).T @ Xh[:, g0:g1].T).real
            A_lr += ((Pg.conj() * swl[g0:g1, None]).T @ Xh[:, g0:g1].T).real
        self.A_lr = A_lr                                            # (P|xi_mu)_LR
        if getattr(self, 'local_rcut', None):
            self.Xi_lr = self._lr_rows_of_xi(Xh)                       # (nlr, N_mu): LR rows of the SS pairs = Xi X
        t_v = time.time() - t
        self.pts, self.V, self.A = pts, V, A
        self.phimu = self.phi[pts]                                  # (N_mu, nao_s)
        self.timings.update({'xi': t_xi, 'V, A (FFT)': t_v})
        return rank

    def _lr_rows_of_xi(self, Xh):
        '''The RSDF LR rows (sqrt(v_G / V) Re/Im rho(G) at RSDF's G vectors, rows 2g / 2g+1) of the interpolation
        vectors xi_mu, from their FFT (Xh: (N_mu, half spectrum), xi(G) = int xi e^{-iGr}).'''
        r, mesh = self.r, np.asarray(self.mesh)
        b = self.cell.reciprocal_vectors()
        m = np.rint(r.lr_G @ np.linalg.inv(b)).astype(int)
        flip = m[:, 2] < 0
        m[flip] *= -1
        assert (np.abs(m) <= mesh // 2).all(), 'LR G vectors beyond the THC mesh'
        i = np.mod(m, mesh)
        flat = (i[:, 0] * mesh[1] + i[:, 1]) * (mesh[2] // 2 + 1) + i[:, 2]
        xg = Xh[:, flat].T                                            # (nG, N_mu)
        xg[flip] = xg[flip].conj()
        sq = np.sqrt(getattr(r, 'lr_vG_rows', r.lr_vG) / self.cell.vol)[:, None]
        rows = np.empty((2 * len(flat), Xh.shape[0]))
        rows[0::2] = sq * xg.real
        rows[1::2] = sq * xg.imag
        return rows

    def _lr_blocks(self, X, blk):
        '''Row blocks (F_cx, F_ss) of the LR rows of the CX pairs (exact, RSDF) and of the SS pairs (THC), packed.'''
        npair = len(self.ti)
        nlr = self.Llr_cx.shape[0]
        for p0 in range(0, nlr, blk):
            p1 = min(p0 + blk, nlr)
            Fcx = np.zeros((p1 - p0, npair))
            Fcx[:, self.lr_cols] = self.Llr_cx[p0:p1]
            Fss = np.zeros((p1 - p0, npair))
            Fss[:, self.ss] = self.Xi_lr[p0:p1] @ X
            yield Fcx, Fss

    def get_k_sparse_v1(self, dm, pblk=20000, kblk=64):
        '''K (vector convention, split basis) for one density (any, complex allowed) with the sparse local CX fit
        (route 1, self.Cs scipy.sparse (naux, npair)) and no N^3 intermediate:
          SS-SS   THC;
          CX-SS   sum_k F_k D G_k^T + G_k D F_k^T, F_k = C_k (SR aux) or the exact LR rows of the CX pairs, G_k = B_k X
                  (B = A = (P|xi)_SR, or the LR rows Xi of xi): with E = D_{:,SS} Phi_mu^T and B_p = (F^T B)_p (built
                  per block of CX pairs), H_a = sum_{p = (a, c)} B_p E_c and K = H Phi_mu (cost nnz(F) N_mu);
          CX-CX   sum_kl M_kl C_k D C_l^T (non-robust SR) + sum_g L_g D L_g^T (LR), as sum_k (C_k D)(M C)_k^T.'''
        import scipy.sparse as sps
        t = time.time()
        nao, sm = self.nao, self.smooth
        dt = np.result_type(dm, float)
        K = np.zeros((nao, nao), dtype=dt)
        Pm = self.phimu
        K[np.ix_(sm, sm)] = Pm.T @ ((Pm @ dm[np.ix_(sm, sm)] @ Pm.T) * self.V) @ Pm
        t_ss = time.time() - t
        t = time.time()
        cxc = self.lr_cols                                            # packed CX pair indices
        ii, jj = self.ti[cxc], self.tj[cxc]
        off = ii != jj
        Ccx = self.Cs[:, cxc].tocsc()                                 # (naux, nCX)
        Lcx = self.Llr_cx                                             # (nlr, nCX) dense
        Bsr, Blr = self.A, self.Xi_lr                                 # (naux, N_mu), (nlr, N_mu)
        lr_cross = getattr(self, 'k_lr', 'dense') == 'cross'          # old path: LR of CX-SS through Lcx^T Xi
        blocks = self._cx_atom_blocks(Ccx)
        ii32, jj32 = ii.astype(np.int32), jj.astype(np.int32)

        def cross(D):
            E = D[:, sm] @ Pm.T                                       # (nao, N_mu)
            cplx = np.iscomplexobj(E)
            Es = [np.ascontiguousarray(E.real)] + ([np.ascontiguousarray(E.imag)] if cplx else [])
            Hs = [np.zeros((nao, len(Pm))) for _ in Es]
            E = None
            for p0 in range(0, len(cxc), pblk):
                p1 = min(p0 + pblk, len(cxc))
                Bp = np.zeros((p1 - p0, Bsr.shape[1]))                # (np, N_mu) = (C^T A)_p, per aux atom GEMMs
                for rows, cols, Cd in blocks:
                    a, b = np.searchsorted(cols, (p0, p1))
                    if b > a:
                        Bp[cols[a:b] - p0] += Cd[:, a:b].T @ Bsr[rows]
                if lr_cross:
                    Bp += Lcx[:, p0:p1].T @ Blr
                for H, E in zip(Hs, Es):                              # H[i] += Bp E[j] (+ H[j] += Bp E[i]), C kernel
                    pair_accum(H, Bp, E, ii32[p0:p1], jj32[p0:p1], off[p0:p1])
            H = Hs[0] @ Pm
            return H + 1j * (Hs[1] @ Pm) if cplx else H                # (nao, nao_s)
        K1 = cross(dm)
        K[:, sm] += K1
        K[sm, :] += cross(dm.T).T                                     # G D F^T = (F D^T G^T)^T
        t_cross = time.time() - t
        t = time.time()

        def unpack(v):                                                # CX-pair vector -> sparse symmetric (nao, nao)
            return sps.csr_matrix((np.concatenate([v, v[off]]), (np.concatenate([ii, jj[off]]),
                                                                np.concatenate([jj, ii[off]]))), shape=(nao, nao))
        # CX-CX, SR (non-robust: C^T M C) and LR (exact rows of the CX pairs): per factor k, dense blocks on the
        # support U_k of C_k and V_k of (M C)_k:  K[U, V] += (C_k[U, U] D[U, :])[:, V] (M C)_k[V, V]^T
        def dense(v, thr=0.):
            nz = np.where(np.abs(v) > thr)[0]
            U = np.unique(np.concatenate([ii[nz], jj[nz]]))
            pos = np.full(nao, -1)
            pos[U] = np.arange(len(U))
            Mk = np.zeros((len(U), len(U)), dtype=v.dtype)
            Mk[pos[ii[nz]], pos[jj[nz]]] = v[nz]
            Mk[pos[jj[nz]], pos[ii[nz]]] = v[nz]
            return U, Mk
        Cr = self.Cs[:, cxc].tocsr()
        naux = Cr.shape[0]
        mthr = getattr(self, 'mc_threshold', 1e-12)
        if getattr(self, 'k_cc', 'blocks') == 'blocks':
            # per aux atom X and chunk of its aux functions P: C_P dense on S_X x S_X (S_X: AOs of the CX pairs
            # fitted on X), W_P = unpack((M C)_P) (nao, nao) from per-atom GEMMs, K[S_X, :] += sum_P (C_P D) W_P
            nCX = len(cxc)
            for rows, cols, Cd in blocks:
                S = np.unique(np.concatenate([ii[cols], jj[cols]]))
                pos = np.full(nao, -1)
                pos[S] = np.arange(len(S))
                li, lj = pos[ii[cols]], pos[jj[cols]]
                DS = dm[S]
                for c0 in range(0, len(rows), kblk):
                    c1 = min(c0 + kblk, len(rows))
                    nb = c1 - c0
                    MC = np.zeros((nb, nCX))
                    for rows2, cols2, Cd2 in blocks:
                        MC[:, cols2] += self.M[np.ix_(rows[c0:c1], rows2)] @ Cd2
                    W = unpack_sym(MC, ii32, jj32, nao).reshape(nb * nao, nao)   # C kernels
                    MC = None
                    Cf = unpack_sym(Cd[c0:c1], li, lj, len(S))
                    CD = _mm_split(Cf, DS)                            # (nb, |S|, nao)
                    CD = CD.transpose(1, 0, 2).reshape(len(S), nb * nao)
                    K[S] += _mm_split(W.T, CD.T).T                    # CD W (W real)
                    W = CD = Cf = None
        for k0 in range(0, naux if getattr(self, 'k_cc', 'blocks') != 'blocks' else 0, kblk):
            k1 = min(k0 + kblk, naux)
            MC = np.asarray(Cr.T @ self.M[k0:k1].T).T                 # (kb, nCX) = (M C)_k
            for k in range(k0, k1):
                row = Cr[k]
                if row.nnz == 0:
                    continue
                ck = np.zeros(len(cxc))
                ck[row.indices] = row.data
                U, Ck = dense(ck)
                V, Mk = dense(MC[k - k0], mthr * abs(MC[k - k0]).max())
                Q = Ck @ dm[U]                                         # (u, nao) = (C_k D)[U, :]
                K[np.ix_(U, V)] += Q[:, V] @ Mk.T
        t_cc = time.time() - t
        t = time.time()
        if lr_cross:
            for g in range(len(Lcx)):
                V, Lg = dense(Lcx[g])
                K[np.ix_(V, V)] += Lg @ dm[np.ix_(V, V)] @ Lg.T
        else:
            # LR of every block through the full LR rows L_g (nao, nao): exact for the CX pairs, THC for the SS
            # pairs (Phi_mu^T diag(Xi_g) Phi_mu); the SS-SS LR is in V already, so its L^SS D L^SS is taken out
            Dss = dm[np.ix_(sm, sm)]
            sm32 = np.asarray(sm, dtype=np.int32)
            gb = getattr(self, 'lr_batch', 16)
            for g0 in range(0, len(Lcx), gb):
                g1 = min(g0 + gb, len(Lcx))
                nb_ = g1 - g0
                Ls = np.stack([(Pm.T * Blr[g]) @ Pm for g in range(g0, g1)])   # (gb, nao_s, nao_s)
                Lg = embed(unpack_sym(Lcx[g0:g1], ii32, jj32, nao), Ls, sm32)   # C kernels
                # sum_g Lg D Lg as two GEMMs per batch: T = [Lg_1 D, ...] stacked, then [T_1 ... T_gb] [Lg_1; ...]
                K += _lr_batch(Lg, dm)
                K[np.ix_(sm, sm)] -= _lr_batch(Ls, Dss)
                Lg = Ls = None
        self.timings.update({'K SS (THC)': t_ss, 'K CX-SS (SR)': t_cross, 'K CX-CX (SR)': t_cc,
                             'K LR': time.time() - t})
        return K

    def get_k_sparse(self, dm):
        '''K of one density (any, complex allowed): get_k_multi on its real and imaginary parts.'''
        if getattr(self, 'k_lr', 'dense') != 'dense' or getattr(self, 'k_cc', 'blocks') != 'blocks':
            return self.get_k_sparse_v1(dm)
        if np.iscomplexobj(dm):
            Kr, Ki = self.get_k_multi([dm.real, dm.imag])
            return Kr + 1j * Ki
        return self.get_k_multi([dm])[0]

    def get_k_multi(self, dms, sym=None, pblk=20000, kblk=None, lr_factors=None):
        '''K (vector convention, split basis) of several real densities at once (route 1).  The density-independent
        intermediates -- B_p = (C^T A)_p of CX-SS, (M C)_P and C_P of CX-CX, the LR rows L_g -- are built once and
        applied to all densities stacked.  sym[n] = +1 / -1 marks a symmetric / antisymmetric density (its CX-SS
        transpose term follows from the direct one), 0 a general one.  Terms as in get_k_sparse_v1.
        lr_factors = (F, terms): the densities as sums of real rank-r products, dms[n] = sum_(c, x, y) c F[x] F[y]^T
        over terms[n]; the LR term then needs sum_g (L_g F[x]) (L_g F[y])^T per product only.'''
        nD = len(dms)
        comm = getattr(self, 'comm', None)                            # MPI (mpi4py): round-robin work split, Allreduce
        rank, size = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
        split = getattr(self, 'mpi_split', ('cs', 'cc', 'lr'))        # parts split over the ranks (others: rank 0)

        def mine(part, i):
            return (i % size == rank) if part in split else rank == 0
        if kblk is None:
            kblk = getattr(self, '_kblk_fixed', None)                 # once other atoms' C dropped: the cached chunks
        if kblk is None:                                              # aux functions per CX-CX chunk (2x1x1: 64 -> 256,
            kblk = getattr(self, 'kblk', None)                        # -12%); W, Y, Z ~ 4 kblk nao^2 doubles
            if kblk is None:
                kblk = int(min(256, max(16, 0.1 * self.avail_bytes() / (4 * 8 * self.nao ** 2))))
        if comm is not None:                                          # the same chunking on every rank
            from mpi4py import MPI
            kblk = comm.allreduce(kblk, op=MPI.MIN)
        dms = [np.ascontiguousarray(d, dtype=float) for d in dms]
        sym = [0] * nD if sym is None else list(sym)
        t = time.time()
        nao, sm = self.nao, self.smooth
        Pm = self.phimu
        Kall = np.zeros((nD, nao, nao))                               # summed over the ranks at the end
        Ks = list(Kall)
        if rank == 0:
            for K, d in zip(Ks, dms):
                K[np.ix_(sm, sm)] = Pm.T @ ((Pm @ d[np.ix_(sm, sm)] @ Pm.T) * self.V) @ Pm
        t_ss = time.time() - t
        t = time.time()
        cxc = self.lr_cols
        ii, jj = self.ti[cxc], self.tj[cxc]
        off = ii != jj
        ii32, jj32 = ii.astype(np.int32), jj.astype(np.int32)
        blocks = self._cx_atom_blocks(None if getattr(self, '_cxb', None) is not None
                                      else self.Cs[:, cxc].tocsc())
        Bsr, Blr, Lcx = self.A, self.Xi_lr, self.Llr_cx
        xown = self._x_owner(blocks, size)                            # aux atom -> rank (greedy by work)

        dist_c = getattr(self, '_row_owner', None) is not None and comm is not None
        if dist_c and getattr(self, 'Cs', None) is not None:          # the own dense blocks hold it now
            self.Cs = None

        def mine_x(part, X):
            if dist_c and part in ('cs', 'cc'):                       # distributed C: only its owner has the block
                return xown[X] == rank
            return (xown[X] == rank) if part in split else rank == 0
        # CX-SS: cross(D) = H(D) Pm with H[a] = sum_{p=(a,c)} B_p E_c, E = D[:, SS] Pm^T; for each density the
        # direct term and (unless sym) the transposed one
        Es, owner = [], []
        for n, d in enumerate(dms):
            Es.append(np.ascontiguousarray(d[:, sm] @ Pm.T)); owner.append((n, 0))
            if sym[n] == 0:
                Es.append(np.ascontiguousarray(d.T[:, sm] @ Pm.T)); owner.append((n, 1))
        Hs = [np.zeros((nao, len(Pm))) for _ in Es]
        ts = dict.fromkeys(('E', 'Bp', 'accum', 'HPm'), 0.)
        ts['E'] = time.time() - t
        # per aux atom X (its rank only): B_p restricted to X's aux, H += sum_{p in cols_X} B^X_p E (linear in B)
        for X, (rows, cols, Cd) in enumerate(blocks):
            if not mine_x('cs', X):
                continue
            for a in range(0, len(cols), pblk):
                ta = time.time()
                bcol = cols[a:a + pblk]
                Bp = Cd[:, a:a + pblk].T @ Bsr[rows[0]:rows[-1] + 1]      # (np, N_mu); rows: a range
                ts['Bp'] += time.time() - ta; ta = time.time()
                for H, E in zip(Hs, Es):
                    pair_accum(H, Bp, E, ii32[bcol], jj32[bcol], off[bcol])
                ts['accum'] += time.time() - ta
        Bp = Es = None
        ta = time.time()
        cr = {}
        for (n, tr), H in zip(owner, Hs):
            cr[n, tr] = H @ Pm                                        # (nao, nao_s)
        Hs = None
        for n in range(nD):
            Ks[n][:, sm] += cr[n, 0]
            Ks[n][sm, :] += (cr[n, 1] if sym[n] == 0 else sym[n] * cr[n, 0]).T
        cr = None
        ts['HPm'] = time.time() - ta
        self.cs_timings = ts
        t_cross = time.time() - t
        t = time.time()
        # CX-CX (SR, non-robust): K[S_X] += sum_P (C_P D) W_P per aux atom X and chunk of P, all densities stacked
        nCX = len(cxc)
        # (M C)_P is density independent: kept across calls when naux nCX doubles fit in k_cache_gb (GB)
        if getattr(self, '_mc_cache', None) is None:
            budget = getattr(self, 'k_cache_gb', None)               # GB; default: half of what is left
            nrows_mc = (sum(len(blocks[X][0]) for X in range(len(blocks)) if mine_x('cc', X)) if dist_c
                        else self.auxmol.nao)                         # distributed C: the own aux rows only
            nbytes = nrows_mc * nCX * 8
            fits = nbytes <= budget * 1e9 if budget is not None else self.fits(nbytes)
            self._mc_cache = {} if fits else False
        mc_cache = self._mc_cache if self._mc_cache is not False else None
        tc = dict.fromkeys(('MC', 'unpack', 'CD', 'swap', 'GEMM', 'add'), 0.)
        aoat = self.aoat
        # distributed C: the (M C)_P of the own aux atoms from every atom's C block broadcast by its owner, for
        # groups of own atoms whose (M C) fit the budget (all at once, and kept, when they fit k_cache_gb);
        # collective: every rank takes as many groups as the busiest one
        own_cc = [X for X in range(len(blocks)) if mine_x('cc', X)]
        groups, mc_group, done = [own_cc], {}, [0]
        if dist_c:
            keys_of = lambda X: [(X, c0, min(c0 + kblk, len(blocks[X][0]))) for c0 in range(0, len(blocks[X][0]), kblk)]
            if mc_cache is None:
                per = max(1., 0.3 * self.avail_bytes() / 8. / max(nCX, 1))      # (M C) rows per group
                groups, cur, nr = [], [], 0
                for X in own_cc:
                    if cur and nr + len(blocks[X][0]) > per:
                        groups.append(cur); cur, nr = [], 0
                    cur.append(X); nr += len(blocks[X][0])
                groups.append(cur)
            ngroups = comm.allreduce(len(groups), op=MPI.MAX)
            groups += [[] for _ in range(ngroups - len(groups))]
        first_of = {g[0]: gi for gi, g in enumerate(groups) if g}

        def mc_pass(gX):
            keys = [k for X in gX for k in keys_of(X) if mc_cache is None or k not in mc_cache]
            if comm.allreduce(len(keys), op=MPI.MAX):
                ta = time.time()
                got = self._mc_broadcast(blocks, keys, nCX, xown)
                if mc_cache is not None:
                    mc_cache.update(got)
                    got = {}
                tc['MC'] += time.time() - ta
                return got
            return {}
        for X in [X for g in groups for X in g]:
            rows, cols, Cd = blocks[X]
            if dist_c and first_of.get(X) == done[0]:
                mc_group = mc_pass(groups[done[0]])
                done[0] += 1
            S = np.unique(np.concatenate([ii[cols], jj[cols]]))
            # local fit: the pairs fitted on X's aux have an AO on atom X (cluster (X, B)), so C_P = [[C_xx, C_xn],
            # [C_nx, 0]] with x the AOs of X: S ordered x first
            Xat = self.auxmol.bas_atom(int(np.searchsorted(self.auxmol.ao_loc_nr(), rows[0], side='right') - 1))
            onx = aoat[S] == Xat
            S = np.concatenate([S[onx], S[~onx]])
            nx = int(onx.sum())
            pos = np.full(nao, -1)
            pos[S] = np.arange(len(S))
            li, lj = pos[ii[cols]], pos[jj[cols]]
            DSh = np.hstack([d[S] for d in dms])                      # (|S|, nD nao)
            DxT = np.ascontiguousarray(np.vstack([d[S[:nx]] for d in dms]).T)   # (nao, nD nx)
            nS = len(S)
            for c0 in range(0, len(rows), kblk):
                c1 = min(c0 + kblk, len(rows))
                nb = c1 - c0
                ta = time.time()
                MC = mc_cache.get((X, c0, c1)) if mc_cache is not None else None
                if MC is None and dist_c:
                    MC = mc_group.pop((X, c0, c1))
                if MC is None:
                    MC = np.zeros((nb, nCX))
                    for rows2, cols2, Cd2 in blocks:
                        assert Cd2 is not None, 'CX-CX: (M C) cache miss after the other atoms\' C were dropped'
                        add_cols(MC, self.M[rows[c0:c1]][:, rows2] @ Cd2, cols2)
                    if mc_cache is not None:
                        mc_cache[X, c0, c1] = MC
                tc['MC'] += time.time() - ta; ta = time.time()
                W = unpack_sym(MC, ii32, jj32, nao)                   # (nb, nao, nao)
                MC = None
                Cf = unpack_sym(Cd[c0:c1], li, lj, nS)
                tc['unpack'] += time.time() - ta
                if c0 == 0 and nx < nS:
                    assert abs(Cf[:, nx:, nx:]).max() == 0, 'CX-CX: C_P not of the (X, any) pair form'
                # C_k = [[C_xx, C_xn], [C_nx, 0]]:  R[x] += sum_k (C_k[x, S] D[S]) W_k,  R[n] += sum_k C_k[n, x] (D[x] W_k)
                ta = time.time()
                Z = (Cf[:, :nx, :].reshape(nb * nx, nS) @ DSh).reshape(nb, nx, nD, nao)
                Z = swap02(Z).reshape(nD * nx, nb * nao)              # (nD, nx, nb, nao)
                Rx = Z @ W.reshape(nb * nao, nao)                     # (nD nx, nao)
                Z = None
                tc['CD'] += time.time() - ta; ta = time.time()
                R = np.zeros((nD, nS, nao))
                R[:, :nx] = Rx.reshape(nD, nx, nao)
                if nx < nS:
                    # (D[x] W_k)^T = W_k D[x]^T for all k in one GEMM: rows (k, b), columns (d, a) -> (d, b, k, a)
                    Y = (W.reshape(nb * nao, nao) @ DxT).reshape(nb, nao, nD, nx)
                    Y = swap02(Y).reshape(nD, nao, nb * nx)
                    CnT = np.ascontiguousarray(Cf[:, nx:, :nx].transpose(0, 2, 1)).reshape(nb * nx, nS - nx)
                    Rn = (Y.reshape(nD * nao, nb * nx) @ CnT).reshape(nD, nao, nS - nx)
                    R[:, nx:] = Rn.transpose(0, 2, 1)                 # sum_k C_k[n, x] D[x] W_k
                    Y = CnT = Rn = None
                tc['GEMM'] += time.time() - ta
                ta = time.time()
                for n in range(nD):
                    add_rows(Ks[n], R[n], S)                          # C kernel
                tc['add'] += time.time() - ta
                W = CD = Cf = R = None
        while dist_c and done[0] < len(groups):                       # the collective passes of the busier ranks
            mc_pass(groups[done[0]])
            done[0] += 1
        self.cc_timings = tc
        # MPI: with this rank's (M C)_P cached, the C blocks of the other aux atoms are not needed any more
        if comm is not None and mc_cache is not None and not getattr(self, '_cx_dropped', False):
            for X in range(len(blocks)):
                if not (mine_x('cs', X) or mine_x('cc', X)):
                    blocks[X] = (blocks[X][0], blocks[X][1], None)
            self.Cs = None
            self._cx_dropped = True
            self._kblk_fixed = kblk
        t_cc = time.time() - t
        t = time.time()
        # LR: sum_g L_g D L_g - L^SS_g D_SS L^SS_g, L_g the full LR rows, all densities stacked
        sm32 = np.asarray(sm, dtype=np.int32)
        gb = getattr(self, 'lr_batch', None)                          # LR rows per batch: L_g, T ~ (nD + 2) gb nao^2
        if gb is None:
            gb = int(min(16, max(1, 0.1 * self.avail_bytes() / ((nD + 2) * 8 * nao ** 2))))
        if comm is not None:
            gb = comm.allreduce(gb, op=MPI.MIN)
        Dh = np.hstack(dms)                                           # (nao, nD nao)
        Dsh = np.hstack([d[np.ix_(sm, sm)] for d in dms])
        nss = len(sm)
        # this rank's LR rows: its range [g_lo, g_hi) of the RSDF LR rows (Lcx holds only them once distributed)
        nlr_all = self.Xi_lr.shape[0]
        if getattr(self, '_lr_range', None) is not None:
            g_lo, g_hi = self._lr_range
        elif 'lr' in split:
            g_lo, g_hi = nlr_all * rank // size, nlr_all * (rank + 1) // size
            Lcx = Lcx[g_lo:g_hi]
        else:
            g_lo, g_hi = (0, nlr_all) if rank == 0 else (0, 0)
            Lcx = Lcx[g_lo:g_hi]
        Lss, si32, sj32 = self._lr_ss_rows(g_lo, g_hi)
        Racc = np.zeros((nD, nao, nao))                               # sum_g L_g D L_g
        Rss = np.zeros((nD, nss, nss))                                # sum_g L^SS_g D_SS L^SS_g
        tl = dict.fromkeys(('Ls', 'unpack', 'GEMM', 'add'), 0.)
        if lr_factors is not None:
            F, terms = lr_factors
            prods = sorted({(x, y) for tn in terms for _, x, y in tn})
            roff = np.cumsum([0] + [f.shape[1] for f in F])
            Fc = np.hstack(F)                                           # (nao, sum r)
            Fsc = np.ascontiguousarray(Fc[sm])
            Mp = {xy: np.zeros((nao, nao)) for xy in prods}
            Mps = {xy: np.zeros((nss, nss)) for xy in prods}
        for g0 in range(0, len(Lcx), gb):                             # local rows (global g_lo + g)
            g1 = min(g0 + gb, len(Lcx))
            ta = time.time()
            if Lss is not None:
                Ls = unpack_sym(Lss[g0:g1], si32, sj32, nss)          # cached THC LR rows of the SS pairs
            else:
                Ls = np.stack([(Pm.T * Blr[g_lo + g]) @ Pm for g in range(g0, g1)])
            tl['Ls'] += time.time() - ta; ta = time.time()
            Lg = embed(unpack_sym(Lcx[g0:g1], ii32, jj32, nao), Ls, sm32)
            tl['unpack'] += time.time() - ta
            if lr_factors is not None:
                # P_x = [L_1 F_x ... L_k F_x] (n, k r_x); sum_g (L_g x)(L_g y)^T = P_x P_y^T
                ta = time.time()
                for L, Fm, n_, Macc in ((Lg, Fc, nao, Mp), (Ls, Fsc, nss, Mps)):
                    k = len(L)
                    P = swap02((L.reshape(k * n_, n_) @ Fm).reshape(k, 1, n_, Fm.shape[1]))   # (n, 1, k, sum r)
                    P = P.reshape(n_, k, Fm.shape[1])
                    Ph = [np.ascontiguousarray(P[:, :, roff[x]:roff[x + 1]]).reshape(n_, -1) for x in range(len(F))]
                    for x, y in prods:
                        Macc[x, y] += Ph[x] @ Ph[y].T
                    P = Ph = None
                tl['GEMM'] += time.time() - ta
                Lg = Ls = None
                continue
            for L, Dst, n_, acc in ((Lg, Dh, nao, Racc), (Ls, Dsh, nss, Rss)):
                ta = time.time()
                k = len(L)
                L2 = L.reshape(k * n_, n_)
                # two large GEMMs per batch (n ~ nao GEMMs per g run far below peak): T = [L_g D_d] rearranged
                # to rows (d, a), columns (g, b), then T [L_1; ...; L_k]
                T = swap02((L2 @ Dst).reshape(k, n_, nD, n_)).reshape(nD * n_, k * n_)
                acc += (T @ L2).reshape(nD, n_, n_)                   # sum_g L_g D L_g per density
                tl['GEMM'] += time.time() - ta
            Lg = Ls = T = R = None
        if lr_factors is not None:
            for n in range(nD):
                for c, x, y in terms[n]:
                    Racc[n] += c * Mp[x, y]
                    Rss[n] += c * Mps[x, y]
            Mp = Mps = None
        for n in range(nD):
            Ks[n] += Racc[n]
            Ks[n][np.ix_(sm, sm)] -= Rss[n]
        Racc = Rss = None
        self.lr_timings = tl
        if comm is not None:
            from mpi4py import MPI
            ta = time.time()
            comm.Allreduce(MPI.IN_PLACE, Kall, op=MPI.SUM)
            tl['allreduce'] = time.time() - ta
        self.timings.update({'K SS (THC)': t_ss, 'K CX-SS (SR)': t_cross, 'K CX-CX (SR)': t_cc,
                             'K LR': time.time() - t})
        return list(Kall)

    def _lr_ss_rows(self, g_lo=0, g_hi=None, pc=20000):
        '''The LR rows of the SS pairs from THC, Xi_lr X_SS (nlr, nSS) with X_SS[mu, (a, b)] = phi_a(r_mu) phi_b(r_mu)
        over the packed smooth pairs a >= b (local smooth indices): density independent, built once and cached.'''
        if g_hi is None:
            g_hi = self.Xi_lr.shape[0]
        if getattr(self, '_lrss', None) is not None and self._lrss_range == (g_lo, g_hi):
            return self._lrss
        Pm = self.phimu
        si, sj = np.tril_indices(Pm.shape[1])
        Xi = self.Xi_lr[g_lo:g_hi]
        if not self.fits(Xi.shape[0] * len(si) * 8):                 # not cached: per batch in get_k_multi
            self.log('   LR rows of the SS pairs not cached (%.1f GB over the budget)' % (Xi.shape[0] * len(si) * 8 / 1e9))
            return None, si.astype(np.int32), sj.astype(np.int32)
        out = np.empty((Xi.shape[0], len(si)))
        for p0 in range(0, len(si), pc):
            p1 = min(p0 + pc, len(si))
            out[:, p0:p1] = Xi @ (Pm[:, si[p0:p1]] * Pm[:, sj[p0:p1]])
        self._lrss = (out, si.astype(np.int32), sj.astype(np.int32))
        self._lrss_range = (g_lo, g_hi)
        self.log('   cached LR rows of the SS pairs: %s (%.1f GB)' % (out.shape, out.nbytes / 1e9))
        return self._lrss

    def _mc_broadcast(self, blocks, keys, nCX, xown):
        '''(M C)_P rows for the keys (X, c0, c1) of this rank, distributed C: every aux atom's block (columns, C)
        broadcast by its owner in turn (one remote block held at a time), M[rows of X][:, rows of Y] C_Y added.'''
        from socutils.pbc.thc_local import bcast_array
        comm = self.comm
        rank = comm.Get_rank()
        out = {k: np.zeros((k[2] - k[1], nCX)) for k in keys}
        for Y, (rowsY, colsY, CdY) in enumerate(blocks):
            root = xown[Y]
            colsY = bcast_array(comm, colsY if rank == root else None, root)
            CdY = bcast_array(comm, CdY if rank == root else None, root)
            if len(colsY) == 0:
                continue
            for (X, c0, c1) in keys:
                rX = blocks[X][0][c0:c1]
                add_cols(out[X, c0, c1], self.M[rX][:, rowsY] @ CdY, colsY)
        return out

    def _x_owner(self, blocks, size):
        '''Aux atom (block index) -> rank, greedy by the block size (the work of CX-SS and CX-CX), cached.'''
        if getattr(self, '_xown', None) is not None and len(self._xown) == len(blocks) and self._xown_size == size:
            return self._xown
        if getattr(self, '_row_owner', None) is not None:              # distributed C: its aux-atom owner
            self._xown, self._xown_size = [self._row_owner[X] for X in self._cxb_atoms], size
            return self._xown
        w = [len(r) * len(c) for r, c, _ in blocks]
        load = np.zeros(size)
        own = [0] * len(blocks)
        for X in np.argsort(w)[::-1]:
            k = int(np.argmin(load))
            own[X] = k
            load[k] += w[X]
        self._xown, self._xown_size = own, size
        return own

    def _cx_atom_blocks(self, Ccx):
        '''C (naux, nCX) as dense blocks per aux atom X: (aux rows of X, sorted CX columns with a nonzero in them,
        C[rows, cols]); the local fit makes these blocks nearly dense, so C^T A is a few BLAS GEMMs (cached).'''
        if getattr(self, '_cxb', None) is not None:
            return self._cxb
        Cr = Ccx.tocsr()
        pat = np.repeat([self.auxmol.bas_atom(i) for i in range(self.auxmol.nbas)], np.diff(self.auxmol.ao_loc_nr()))
        owner = getattr(self, '_row_owner', None)                      # distributed C: own aux atoms' rows only
        rank = self.comm.Get_rank() if owner is not None else 0
        out, atoms = [], []
        for X in range(self.auxmol.natm):
            rows = np.where(pat == X)[0]
            if len(rows) == 0:
                continue
            assert rows[-1] - rows[0] + 1 == len(rows)                 # the aux functions of an atom are a range
            if owner is not None and owner[X] != rank:
                out.append((rows, None, None))                         # held by its owner (same block index)
                atoms.append(X)
                continue
            sub = Cr[rows]
            cols = np.unique(sub.indices)
            if len(cols) or owner is not None:
                out.append((rows, cols, np.ascontiguousarray(sub[:, cols].toarray())))
                atoms.append(X)
        self._cxb, self._cxb_atoms = out, atoms
        self.log('   CX atom blocks: %d, %.2e dense entries for %.2e nonzeros of C'
                 % (len(out), sum(c.size for _, _, c in out if c is not None), Cr.nnz))
        return out

    def j_thc_ss(self):
        '''(P|q)_SR for the SS pairs from THC: A X, X_mu,q = phi_m(r_mu) phi_n(r_mu)  (naux, nSS).'''
        pos = -np.ones(self.nao, dtype=int)
        pos[self.smooth] = np.arange(len(self.smooth))
        m, n = pos[self.ti[self.ss]], pos[self.tj[self.ss]]
        X = self.phimu[:, m] * self.phimu[:, n]
        return self.A @ X

    def get_k(self, dm):
        '''K (vector convention, split basis, real dm): THC SS-SS + local CX (SR) + exact LR rows of the CX blocks.'''
        t = time.time()
        nao, sm = self.nao, self.smooth
        K = np.zeros((nao, nao), dtype=np.result_type(dm, float))
        # SS-SS: K_ab = sum_mu,nu phi_a(mu) [(Phi D Phi^T) o V]_mu,nu phi_b(nu)
        P = self.phimu
        Mmn = (P @ dm[np.ix_(sm, sm)] @ P.T) * self.V
        K[np.ix_(sm, sm)] = P.T @ Mmn @ P
        t_ss = time.time() - t
        t = time.time()
        if getattr(self, 'nonrobust', False):                        # same form as get_k_occ
            J3 = np.zeros((len(self.C), len(self.ti)))
            J3[:, self.ss] = self.j_thc_ss()
            K += k_from_factors(self.C, J3, dm, nao) + k_from_factors(J3, self.C, dm, nao) \
                + k_from_factors(self.Y, self.Y, dm, nao)
        else:
            J3 = self.J.copy()
            J3[:, self.ss] = self.j_thc_ss()
            K += k_from_factors(self.C, J3, dm, nao) + k_from_factors(J3, self.C, dm, nao) \
                - k_from_factors(self.Y, self.Y, dm, nao)
        if getattr(self, 'local_rcut', None):
            pos = -np.ones(nao, dtype=int)
            pos[sm] = np.arange(len(sm))
            X = P[:, pos[self.ti[self.ss]]] * P[:, pos[self.tj[self.ss]]]
            for Fcx, Fss in self._lr_blocks(X, 256):
                K += k_from_factors(Fcx, Fcx, dm, nao) + k_from_factors(Fcx, Fss, dm, nao) \
                    + k_from_factors(Fss, Fcx, dm, nao)
        else:
            Lss = np.zeros_like(self.Llr)
            Lss[:, self.ss] = self.Llr[:, self.ss]
            K += k_from_factors(self.Llr, self.Llr, dm, nao) - k_from_factors(Lss, Lss, dm, nao)
        self.timings.update({'K SS (THC)': t_ss, 'K CX (prototype, dense factors)': time.time() - t})
        return K

    def _aux_ft(self):
        '''Analytic FTs of the aux functions on the half spectrum, (naux, nG) complex128, cached (prototype; a
        production code would put the local SR potentials of the aux functions on the grid in real space).  Double
        precision is needed: the fitted compact density is dominated by the core charges (E_J ~ 1e4 Eh), and single
        precision left ~1e-3 Eh of noise in J that is not linear in D (it broke the incremental Fock build).'''
        if getattr(self, '_Pg', None) is None:
            Pg = np.empty((self.auxmol.nao, len(self.Gv)), dtype=np.complex128)
            for g0 in range(0, len(self.Gv), 20000):
                g1 = min(g0 + 20000, len(self.Gv))
                Pg[:, g0:g1] = ft_ao(self.auxmol, self.Gv[g0:g1]).T
            self._Pg = Pg
        return self._Pg

    def _fit_vectors(self):
        '''L = W J3 (W = M^-1/2 on the kept space), the fitted SR vectors of all pairs, cached: RI-J as L^T (L d)
        passes the roundoff through sqrt(cond M) instead of cond M (M^-1 b differed by 4e-3 in J between equivalent
        evaluation orders for EuOCl).'''
        if getattr(self, '_Lfit', None) is None:
            v, e = _inv_metric(self.M)
            self._Lfit = np.asarray((v / np.sqrt(e)).T @ self.J, order='C')
        return self._Lfit

    def get_k_occ(self, cs, blk=64):
        '''K from occupied factors (occ-RI), split basis: cs = [C^s] (nao, nocc) per spin component s (occupations
        absorbed, D^{ss'} = C^s C^s'^H); returns K[s][s'] (nao, nao).  Every factor pair is applied as
        K^{ss'} = sum_P (A_P C^s)(B_P C^s')^H, the half transforms shared by all spin blocks; the (J3, C) term is the
        conjugate transpose of the (C, J3) one; the SS columns of J3 come from THC per block of aux rows.'''
        nao, sm, ns = self.nao, self.smooth, len(cs)
        t = time.time()
        K = [[np.zeros((nao, nao), dtype=complex) for _ in range(ns)] for _ in range(ns)]
        Pm = self.phimu
        U = [Pm @ c[sm] for c in cs]                                  # (N_mu, nocc)
        for s in range(ns):
            for s2 in range(ns):
                M = (U[s] @ U[s2].conj().T) * self.V
                K[s][s2][np.ix_(sm, sm)] += Pm.T @ M @ Pm
        t_ss = time.time() - t
        t = time.time()
        pos = -np.ones(nao, dtype=int)
        pos[sm] = np.arange(len(sm))
        X = Pm[:, pos[self.ti[self.ss]]] * Pm[:, pos[self.tj[self.ss]]]   # (N_mu, nSS)
        local = getattr(self, 'local_rcut', None)
        if not local:
            Lss = self.Llr.copy()
            Lss[:, self.cx] = 0.

        nocc = cs[0].shape[1]
        cri = [np.hstack([c.real, c.imag]) for c in cs]               # (nao, 2 nocc) real

        def half(F):
            # (A_P C^s)_a,i for a block of factor rows, as one real GEMM per spin: (nP nao, nao) x (nao, 2 nocc)
            f = lib.unpack_tril(F).reshape(-1, nao)
            out = []
            for c in cri:
                h = (f @ c).reshape(len(F), nao, 2, nocc)
                out.append(np.ascontiguousarray((h[:, :, 0] + 1j * h[:, :, 1]).transpose(1, 0, 2)).reshape(nao, -1))
            return out                                                 # [(nao, nP nocc)] complex

        def acc(HA, HB, sign, swap=False):
            for s in range(ns):
                for s2 in range(ns):
                    k = HA[s] @ HB[s2].conj().T
                    if swap:
                        K[s2][s] += sign * k.conj().T
                    K[s][s2] += sign * k

        naux = len(self.C)
        nonrobust = getattr(self, 'nonrobust', False)
        for p0 in range(0, naux, blk):
            p1 = min(p0 + blk, naux)
            Cb = self.C[p0:p1]
            if not Cb.any():
                continue
            if nonrobust:                                              # CX-SS one-sided, CX-CX C^T M C
                J3 = np.zeros((p1 - p0, len(self.ti)))
            else:
                J3 = self.J[p0:p1].copy()
            J3[:, self.ss] = self.A[p0:p1] @ X
            HC = half(Cb)
            acc(HC, half(J3), 1., swap=True)                           # (C, J3) + (J3, C)
            HY = half(self.Y[p0:p1])
            acc(HY, HY, 1. if nonrobust else -1.)
        if local:                                                      # LR of the CX-involving blocks
            for Fcx, Fss in self._lr_blocks(X, blk):
                HL = half(Fcx)
                acc(HL, HL, 1.)
                acc(HL, half(Fss), 1., swap=True)
        else:
            for p0 in range(0, len(self.Llr), blk):
                p1 = min(p0 + blk, len(self.Llr))
                HL = half(self.Llr[p0:p1])
                acc(HL, HL, 1.)
                HS = half(Lss[p0:p1])
                acc(HS, HS, -1.)
        self.timings.update({'K SS (THC, occ)': t_ss, 'K CX (occ-RI)': time.time() - t})
        return K

    def get_j(self, dm):
        '''J (vector convention, split basis).  The pair-local CX fit is good for K but not for J: its residual
        densities add up over the pairs with one sign, so J is robust RI-J of the total densities instead:
            n_SS(r) on the uniform grid (exact for the smooth functions);
            rho_CX = sum_p d_p rho_p (p CX), b = (P|rho_CX)_SR exact three-centre, global fit c_CX = M^-1 b;
            J_p (SS p) = (rho_p | n_SS)_grid + (rho_p | sum_P c_CX,P P)_SR      (the latter by the SR 3c integrals)
            J_p (CX p) = (rho_p | P)_SR . M^-1 [b + (P|n_SS)_SR]               (the global fit of all of rho)
        (the derivative of the robust (rho_CX|n_SS) + 1/2 (rho_CX|rho_CX); the error is quadratic in the fit
        error of the total density), plus the LR rows of the CX-involving blocks.'''
        if np.iscomplexobj(dm):
            return self.get_j(dm.real) + 1j * self.get_j(dm.imag)
        if getattr(self, 'local_rcut', None):                         # route 1: no three-centre integrals
            if getattr(self, '_gapw', None) is None:
                from socutils.pbc.thc_gapw import GAPWJ
                self._gapw = GAPWJ(self, **getattr(self, 'gapw_opts', {}))
            return self._gapw.get_j(dm)
        nao, sm = self.nao, self.smooth
        ti, tj = self.ti, self.tj
        d = (dm + dm.T)[ti, tj]
        d[ti == tj] *= .5                                           # packed density: D_mn + D_nm (m > n), D_mm
        cxi = np.where(self.cx)[0]
        ss = self.ss
        vol, ng, mesh = self.cell.vol, self.ng, self.mesh
        hshape = (mesh[0], mesh[1], mesh[2] // 2 + 1)
        L = self._fit_vectors()
        Lb = L[:, cxi] @ d[cxi]                                     # W (P|rho_CX)_SR
        Ln = L[:, ss] @ d[ss]                                       # W (P|n_SS)_SR, exact three-centre (local, sparse)
        n = np.einsum('ri,ri->r', self.phi @ dm[np.ix_(sm, sm)], self.phi)
        nG = scipy.fft.rfftn(n.reshape(mesh), workers=-1).reshape(-1) * (vol / ng)
        vr = scipy.fft.irfftn((self.w_vec * nG).reshape(hshape), s=mesh, workers=-1).reshape(-1)   # v(r) vol / ng
        Jss = self.phi.T @ (vr[:, None] * self.phi)
        jp = np.zeros(len(d))
        jp[ss] = Jss[np.searchsorted(sm, ti[ss]), np.searchsorted(sm, tj[ss])] + L[:, ss].T @ Lb
        jp[cxi] = L[:, cxi].T @ (Lb + Ln)
        Lss = np.zeros_like(self.Llr)
        Lss[:, self.ss] = self.Llr[:, self.ss]
        jp += self.Llr.T @ (self.Llr @ d) - Lss.T @ (Lss @ d)
        J = np.zeros((nao, nao))
        J[ti, tj] = jp
        J[tj, ti] = jp
        return J

    def get_j_thc(self, dm):
        '''J with THC for the SS blocks (diagnostic: the interpolation errors add up coherently).'''
        nao = self.nao
        ti, tj = self.ti, self.tj
        d = (dm + dm.T)[ti, tj]
        d[ti == tj] *= .5
        pos = -np.ones(nao, dtype=int)
        pos[self.smooth] = np.arange(len(self.smooth))
        X = self.phimu[:, pos[ti[self.ss]]] * self.phimu[:, pos[tj[self.ss]]]   # (N_mu, nSS)
        jp = np.zeros(len(d), dtype=d.dtype)
        jp[self.ss] = X.T @ (self.V @ (X @ d[self.ss]))
        J3 = self.J.copy()
        J3[:, self.ss] = self.A @ X
        jp += self.C.T @ (J3 @ d) + J3.T @ (self.C @ d) - self.Y.T @ (self.Y @ d)
        Lss = np.zeros_like(self.Llr)
        Lss[:, self.ss] = self.Llr[:, self.ss]
        jp += self.Llr.T @ (self.Llr @ d) - Lss.T @ (Lss @ d)
        J = np.zeros((nao, nao), dtype=jp.dtype)
        J[ti, tj] = jp
        J[tj, ti] = jp
        return J

    def get_jk(self, dms, with_j=True, with_k=True):
        dms = np.asarray(dms)
        kf = self.get_k_sparse if getattr(self, 'local_rcut', None) else self.get_k
        vj = np.array([self.get_j(d) for d in dms]) if with_j else None
        vk = np.array([kf(d) for d in dms]) if with_k else None
        return vj, vk

    def storage(self):
        nmu, nao_s = len(self.pts), len(self.smooth)
        return {'V': nmu * nmu, 'phi(r_mu)': nmu * nao_s, '(P|xi)': self.A.size,
                'CX local C (nonzero)': int(self.Cs.nnz) if getattr(self, 'Cs', None) is not None
                else (int(np.count_nonzero(self.C)) if getattr(self, 'C', None) is not None else -1),
                'CX LR rows': int(self.Llr_cx.size) if getattr(self, 'Llr_cx', None) is not None else 0,
                'RSDF cderi (SR rows)': self.nkept * len(self.cx)}


def _subcell(cell, ao_idx):
    '''A cell with only the shells containing the AOs ao_idx (whole shells; used for the ke-cutoff estimate).'''
    loc = cell.ao_loc_nr()
    shells = sorted(set(np.searchsorted(loc, ao_idx, side='right') - 1))
    c = cell.copy(deep=True)
    c._bas = cell._bas[shells]
    return c


class THCJK:
    '''Gamma-point J/K for socutils.pbc.spinor_scf (the RSDF interface): THCExchange in the basis split at alpha_c,
    mapped to the cell's basis (chi_cell = chi_split T); nuclei, S and c_g0 from its RSDF (lr='gspace').
    split_direct.periodic(mf, THCJK(...)) gives the periodic spinor SCF.'''

    def __init__(self, cell, aux, alpha_c=4., nmu_factor=16., selector='coarse', omega=None, nuc_zeta=1e12,
                 verbose=True, local_rcut=None):
        # omega: 0.35 for route 1 (local_rcut; with the far-field fit, EuOCl E(D) within 4e-5 Eh of omega 0.5 at
        # 35% of the LR rows), OMEGA otherwise
        if omega is None:
            omega = OMEGA_ROUTE1 if local_rcut else OMEGA
        self.cell, self.aux, self.alpha_c, self.nmu_factor = cell, aux, alpha_c, nmu_factor
        self.selector, self.omega, self.nuc_zeta, self.verbose = selector, omega, nuc_zeta, verbose
        self.local_rcut = local_rcut   # Angstrom: pair-atom local fit of the CX pairs (thc_local), non-robust K
        self._built = False

    def build(self):
        from pyscf.pbc import gto as pgto
        from socutils.pbc.bench.sr_locality import split_basis
        cell = self.cell
        cs = cell.copy(deep=True)
        cs.basis = split_basis(cell._basis, self.alpha_c)
        cs.build(False, False)
        self.T = np.linalg.lstsq(cs.pbc_intor('int1e_ovlp'), pgto.intor_cross('int1e_ovlp', cs, cell), rcond=1e-12)[0]
        self.th = THCExchange(cs, self.aux, self.alpha_c, self.omega, self.verbose, self.selector,
                              self.nuc_zeta, local_rcut=self.local_rcut)
        self.th.dense_c = not self.local_rcut                          # route 1: the sparse C only
        self.th.gapw_opts = getattr(self, 'gapw_opts', {})
        self.th.comm = getattr(self, 'comm', None)                    # MPI: set jk.comm before build()
        self.th.stream = getattr(self, 'stream', False)
        if hasattr(self, 'isdf_cut'):
            self.th.isdf_cut = self.isdf_cut
        self.th.stream_gb = getattr(self, 'stream_gb', None)       # GB; None: from the memory budget
        self.th.stream_max_blocks = getattr(self, 'stream_max_blocks', 1 << 30)
        self.th.piv_file = getattr(self, 'piv_file', None)
        for k in ('coarsen', 'blocked_pivots', 'far_fit', 'phi_ondemand', 'symm_grid', 'symm_pivots',
                  'ke_precision', 'mesh_override', 'drop_nyquist', 'g_sphere', 'distribute_c'):   # if set
            if hasattr(self, k):
                setattr(self.th, k, getattr(self, k))
        rc = getattr(self, 'exx_trunc', None)                         # smoothed truncated Coulomb exchange
        if isinstance(rc, str) and rc == 'auto':
            # the inradius of the Wigner-Seitz cell (half the shortest lattice vector): at Gamma the kernel is
            # periodised over the cell's lattice, and a sphere beyond it overlaps its images (double-counted
            # exchange; the Spencer-Alavi R_c = (3V / 4 pi)^(1/3) does that, and diverges for a 2x1x1 cell)
            a_ = np.asarray(cell.lattice_vectors())
            n_ = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])
            rc = .5 * np.linalg.norm(n_ @ a_, axis=1).min()
        self.exx_trunc = self.th.exx_trunc = rc
        self.th.kmax_factor = self.nmu_factor                         # pivots: only the N_mu prefix is used
        self.th.build_common()
        self.th.build_thc(int(round(self.nmu_factor * len(self.th.smooth))))
        self.th.distribute()
        self.S = cell.pbc_intor('int1e_ovlp')
        self._built = True
        return self

    @property
    def c_g0(self):
        return self.th.r.c_g0

    def exx_g0(self):
        '''G = 0 exchange coefficient of S D S for the truncated kernel: c0 (the SR part, as w_vec(0) / V) plus
        v_TC(0) / V = 2 pi R_c^2 / V (spinor_scf exxdiv 'stc'); None without exx_trunc.'''
        rc = getattr(self, 'exx_trunc', None)
        if not rc:
            return None
        return self.c_g0 + 2 * np.pi * rc ** 2 / self.cell.vol

    def g0_constant(self, nelectron):
        return self.th.r.g0_constant(nelectron)

    def get_nuc(self):
        return self.T.T @ self.th.r.get_nuc() @ self.T

    def energy_nuc(self):
        return self.th.r.energy_nuc()

    def _lr_occ(self, nocc):
        '''LR exchange from the occupied factors: 10 products of rank nocc vs 6 dense densities, worth it for
        nocc <~ 0.3 nao (EuOCl 2x1x1, nocc 0.52 nao: 44 s factored vs 30 s dense); th.lr_occ True / False forces.'''
        f = getattr(self.th, 'lr_occ', 'auto')
        return nocc <= 0.3 * self.th.nao if f == 'auto' else bool(f)

    def get_jk_mo(self, mol, mo_coeff, mo_occ):
        '''Spinor J and K for the density sum_i n_i |i><i| (spinor MOs, n_i >= 0) by occ-RI (no G = 0 terms).'''
        if not self._built:
            self.build()
        comm = getattr(self.th, 'comm', None)
        if comm is not None:                                          # every rank contracts rank 0's orbitals
            mo_coeff = np.array(mo_coeff, dtype=complex, order='C')    # writable copies (Bcast fills them)
            mo_occ = np.array(np.broadcast_to(np.asarray(mo_occ, dtype=float), (mo_coeff.shape[1],)), order='C')
            comm.Bcast(mo_coeff, root=0)
            comm.Bcast(mo_occ, root=0)
        nao, T = self.cell.nao, self.T
        if getattr(self, '_c2', None) is None:
            self._c2 = np.vstack(mol.sph2spinor_coeff())              # (2 nao, n2c)
        c2 = self._c2
        mo_coeff = np.asarray(mo_coeff)
        occ = np.broadcast_to(np.asarray(mo_occ, dtype=float), (mo_coeff.shape[1],))
        sel = occ > 1e-12
        Cs = (c2 @ mo_coeff[:, sel]) * np.sqrt(occ[sel])
        cs = [T @ Cs[:nao], T @ Cs[nao:]]                              # spin components in the split basis
        if getattr(self.th, 'local_rcut', None):                      # route 1: sparse DM-contracted K
            # one batched K for the spin blocks: aa, bb Hermitian (real part symmetric, imaginary antisymmetric),
            # ab general; ba = ab^H
            Daa, Dbb, Dab = cs[0] @ cs[0].conj().T, cs[1] @ cs[1].conj().T, cs[0] @ cs[1].conj().T
            # real factors ar, ai, br, bi of the occupied spinor components (for the LR term)
            F = [np.ascontiguousarray(cs[0].real), np.ascontiguousarray(cs[0].imag),
                 np.ascontiguousarray(cs[1].real), np.ascontiguousarray(cs[1].imag)]
            terms = [[(1., 0, 0), (1., 1, 1)], [(1., 1, 0), (-1., 0, 1)],          # aa: Re, Im
                     [(1., 2, 2), (1., 3, 3)], [(1., 3, 2), (-1., 2, 3)],          # bb
                     [(1., 0, 2), (1., 1, 3)], [(1., 1, 2), (-1., 0, 3)]]          # ab = ca cb^H
            Kr = self.th.get_k_multi([Daa.real, Daa.imag, Dbb.real, Dbb.imag, Dab.real, Dab.imag],
                                     sym=[1, -1, 1, -1, 0, 0],
                                     lr_factors=(F, terms) if self._lr_occ(F[0].shape[1]) else None)
            Kaa, Kbb, Kab = Kr[0] + 1j * Kr[1], Kr[2] + 1j * Kr[3], Kr[4] + 1j * Kr[5]
            Ks = [[Kaa, Kab], [Kab.conj().T, Kbb]]
        else:
            Ks = self.th.get_k_occ(cs)
        dtr = cs[0] @ cs[0].conj().T + cs[1] @ cs[1].conj().T
        Jsp = T.T @ self.th.get_j(dtr.real) @ T
        ksph = np.zeros((2 * nao, 2 * nao), dtype=complex)
        jsph = np.zeros((2 * nao, 2 * nao), dtype=complex)
        for s in range(2):
            jsph[s * nao:(s + 1) * nao, s * nao:(s + 1) * nao] = Jsp
            for s2 in range(2):
                ksph[s * nao:(s + 1) * nao, s2 * nao:(s2 + 1) * nao] = T.T @ Ks[s][s2] @ T
        c2h = c2.conj().T
        return c2h @ jsph @ c2, c2h @ ksph @ c2

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True, direct_scf_tol=None, omega=None, exxdiv=None,
               with_g0=False):
        if omega or exxdiv:
            raise NotImplementedError
        if not self._built:
            self.build()
        dm = np.asarray(dm)
        comm = getattr(self.th, 'comm', None)
        if comm is not None:                                          # every rank contracts rank 0's density
            dm = np.array(dm, dtype=np.result_type(dm, float), order='C')
            comm.Bcast(dm, root=0)
        nao = self.cell.nao
        dms = dm.reshape(-1, nao, nao)
        T = self.T
        ds = lib.einsum('pi,nij,qj->npq', T, dms, T)
        vj, vk = self.th.get_jk(ds, with_j, with_k)
        if with_j:
            vj = lib.einsum('pi,npq,qj->nij', T, vj, T)
        if with_k:
            vk = lib.einsum('pi,npq,qj->nij', T, vk, T)
        if with_g0 and self.c_g0:
            c0, S = self.c_g0, self.S
            if with_j:
                vj -= c0 * np.einsum('nij,ji->n', dms, S)[:, None, None] * S
            if with_k:
                vk -= c0 * np.einsum('ij,njk,kl->nil', S, dms, S)
        return (None if vj is None else vj.reshape(dm.shape)), (None if vk is None else vk.reshape(dm.shape))
