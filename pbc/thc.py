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
import scipy.fft
from pyscf import lib
from pyscf.gto.ft_ao import ft_ao
from pyscf.df import addons as df_addons
from pyscf.pbc.gto.cell import estimate_ke_cutoff
from socutils.pbc.bench.hybrid_k import _inv_metric, k_from_factors
from socutils.pbc.rsdf import RSDF

OMEGA = 0.5


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
        self.coarsen = 3             # coarse: grid coarsening per direction for the point selection (4: EuOCl fails)
        self.blocked_pivots = True   # coarse: blocked (BLAS-3) pivoted Cholesky
        self.nuc_zeta = nuc_zeta
        self.verbose = verbose
        self.timings = {}

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

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
            self.Cs = local_cx_fit(cell, self.aux, aohard, omega=self.omega, r_cut=local, verbose=self.verbose,
                                   sparse=True)[0]
            self.C = self.Cs.toarray() if getattr(self, 'dense_c', True) else None
            self.nonrobust = True
        else:
            self._bloch_fit()
        if self.C is not None:
            ew, U = scipy.linalg.eigh(self.M)
            self.Y = ((U * np.sqrt(np.maximum(ew, 0))) @ U.T) @ self.C
        self.timings['local fit'] = time.time() - t
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
        self.ke = estimate_ke_cutoff(sm_cell, 1e-10)
        self.mesh = cell.cutoff_to_mesh(self.ke)
        coords = self.coords = cell.get_uniform_grids(self.mesh)
        self.phi = cell.pbc_eval_gto('GTOval_sph', coords)[:, self.smooth]  # (ng, nao_s)
        self.ng = len(coords)
        self.timings['grid'] = time.time() - t
        t = time.time()
        if self.selector == 'coarse':
            # pivoted Cholesky on a grid coarsened by coarsen per direction, pivots snapped to the nearest point of
            # the full grid (the smooth pair densities are band-limited; xi is fitted on the full grid)
            mc = np.maximum(np.asarray(self.mesh) // self.coarsen, 1)
            cc = cell.get_uniform_grids(mc)
            phic = cell.pbc_eval_gto('GTOval_sph', cc)[:, self.smooth]
            f = pivots_blocked if self.blocked_pivots else pivots
            kmax = min(len(cc), int(getattr(self, 'kmax_factor', kmax_factor) * len(self.smooth)))
            pf = getattr(self, 'piv_file', None)                      # cache of the coarse pivots (restarts)
            if pf and os.path.exists(pf) and len(np.load(pf)) >= kmax:
                pc = np.load(pf)[:kmax]
            else:
                pc, _ = f(phic, kmax)
                if pf:
                    np.save(pf, pc)
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
        self.Gv, self.fG = half_gv(cell, self.mesh)
        self.w_vec, self.w_sr = kernels(self.Gv, self.omega)
        self.log('   THC common: nao %d (smooth %d), CX pairs %d of %d, ke %.0f Eh, mesh %s = %d points, %d pivots; '
                 'RSDF %.0f s, local fit %.0f s, grid %.0f s, pivots %.0f s'
                 % (nao, len(self.smooth), self.cx.sum(), len(self.cx), self.ke, self.mesh, self.ng, len(self.piv),
                    self.timings['rsdf'], self.timings['local fit'], self.timings['grid'], self.timings['pivots']))

    # ---- THC factors for N_mu points
    def _build_thc_stream(self, pts, cut=1e-13, nb=None, gchunk=20000, rchunk=100000):
        '''V, A, A_lr and the LR rows Xi of the ISDF vectors without storing xi (N_mu x ngrid). Blocks b of mu:
        xi_b = G+[b] Gc with the Gram columns Gc[P, r] = (phi(r_P) . phi(r))^2 regenerated in grid chunks, its FFT
        gives A[:, b], A_lr[:, b] and Xi[:, b] as in the full build, and V[b] = int (v * xi_b) Gc^T G+ (one G+ on
        the right of an explicitly formed xi_b; V = G+ Z G+ from Z = Gc v Gc^T loses ~1e-3 Eh to the conditioning).'''
        vol, ng, mesh = self.cell.vol, self.ng, self.mesh
        hshape = (mesh[0], mesh[1], mesh[2] // 2 + 1)
        phi = self.phi
        PP = phi[pts]
        nmu = len(pts)
        e, v = scipy.linalg.eigh((PP @ PP.T) ** 2)
        keep = e > cut * e.max()
        Ginv = (v[:, keep] / e[keep]) @ v[:, keep].T
        if nb is None:
            nb = max(64, min(nmu, int(getattr(self, 'stream_gb', 30) * 1e9 / (8 * ng * 3))))   # ~2-3 (nb, ng) live
        V = np.zeros((nmu, nmu))
        A = np.zeros((self.auxmol.nao, nmu))
        A_lr = np.zeros((self.auxmol.nao, nmu))
        sw = self.fG * self.w_sr / vol
        swl = self.fG * (self.w_vec - self.w_sr) / vol
        lrg = np.where(np.abs(swl) > 1e-14 * np.abs(swl).max())[0]       # the LR kernel lives at small G only
        Pl = ft_ao(self.auxmol, self.Gv[lrg]) * swl[lrg, None]           # (nG_lr, naux)
        Plr, Pli = np.ascontiguousarray(Pl.real.T), np.ascontiguousarray(Pl.imag.T)
        Pl = None
        local = getattr(self, 'local_rcut', None)
        if local:
            Xi = np.empty((2 * len(self.r.lr_G), nmu))
        G2 = np.einsum('gx,gx->g', self.Gv, self.Gv)
        gord = np.argsort(G2, kind='stable')
        t0 = time.time()
        tt = dict.fromkeys(('xi', 'FFT', 'ft_ao', 'A GEMM', 'V'), 0.)
        for ib, b0 in enumerate(range(0, nmu, nb)):
            if ib >= getattr(self, 'stream_max_blocks', 1 << 30):     # timing runs: stop after a few blocks
                self.log('   streamed THC: stopped after %d blocks (timing run; V, A incomplete)' % ib)
                break
            b1 = min(b0 + nb, nmu)
            ta = time.time()
            xi = np.empty((b1 - b0, ng))
            for g0 in range(0, ng, rchunk):
                g1 = min(g0 + rchunk, ng)
                xi[:, g0:g1] = Ginv[b0:b1] @ ((PP @ phi[g0:g1].T) ** 2)
            tt['xi'] += time.time() - ta
            ta = time.time()
            Xh = scipy.fft.rfftn(xi.reshape(b1 - b0, *mesh), axes=(1, 2, 3), workers=-1).reshape(b1 - b0, -1)
            xi = None
            Xh *= vol / ng                                              # xi(G) = int_cell xi(r) e^{-iGr}
            tt['FFT'] += time.time() - ta
            for g0 in range(0, len(self.Gv), gchunk):
                g1 = min(g0 + gchunk, len(self.Gv))
                ta = time.time()
                idx = gord[g0:g1]                                       # G sorted by |G|: chunks of growing |G|
                sub, cols = self._aux_upto(G2[idx[0]])                  # aux shells not negligible at |G| >= G_min
                if sub is None:
                    continue
                Pg = np.ascontiguousarray(ft_ao(sub, self.Gv[idx]) * sw[idx, None])   # (nG, naux_sub), C order
                tt['ft_ao'] += time.time() - ta
                ta = time.time()
                # Re(P^* X) = Pr Xr + Pi Xi as one real GEMM over (G, re/im) with contiguous operands
                P2 = np.ascontiguousarray(Pg.view(float).reshape(g1 - g0, -1, 2).transpose(0, 2, 1)).reshape(-1, Pg.shape[1])
                Pg = None
                X2 = np.ascontiguousarray(Xh[:, idx]).view(float)       # (nb, 2 nG)
                A[cols, b0:b1] += (X2 @ P2).T
                tt['A GEMM'] += time.time() - ta
            Xc = Xh[:, lrg]
            A_lr[:, b0:b1] = Plr @ Xc.real.T + Pli @ Xc.imag.T
            Xc = Pg = None
            if local:
                Xi[:, b0:b1] = self._lr_rows_of_xi(Xh)
            # potential of the block in real space, (v * xi_b)(r) = irfft(w_vec xi_b(G)) ng / vol
            ta = time.time()
            Xh *= self.w_vec[None]
            xi = scipy.fft.irfftn(Xh.reshape((b1 - b0,) + hshape), s=tuple(mesh), axes=(1, 2, 3),
                                  workers=-1, overwrite_x=True).reshape(b1 - b0, -1)
            Xh = None
            Vb = np.zeros((b1 - b0, nmu))
            for g0 in range(0, ng, rchunk):
                g1 = min(g0 + rchunk, ng)
                Vb += xi[:, g0:g1] @ ((PP @ phi[g0:g1].T) ** 2).T
            V[b0:b1] = Vb @ Ginv                                        # int (v * xi_b) xi^T (Parseval)
            tt['V'] += time.time() - ta
            xi = Vb = None
            self.log('   streamed THC: mu block %d-%d of %d, %.0f s (%s)'
                     % (b0, b1, nmu, time.time() - t0, ', '.join('%s %.0f' % kv for kv in tt.items())))
        self.V = .5 * (V + V.T)
        self.A, self.A_lr = A, A_lr
        if local:
            self.Xi_lr = Xi
        return int(keep.sum())

    def _aux_upto(self, g2min, eps=1e-15):
        '''The aux shells whose FT is not negligible at |G|^2 >= g2min (exp(-G^2 / 4 alpha) > eps for the smallest
        exponent of the shell), as a sub-molecule and the aux AO indices of its functions (cached per prefix).'''
        am = self.auxmol
        if getattr(self, '_aux_cut', None) is None:
            amin = np.array([am.bas_exp(i).min() for i in range(am.nbas)])
            cut = 4 * amin * np.log(1 / eps)                            # |G|^2 beyond which shell i is negligible
            order = np.argsort(-cut, kind='stable')
            self._aux_cut = (cut[order], order, {})
        cut, order, cache = self._aux_cut
        if not getattr(self, 'ft_prune', True):
            k = am.nbas
        else:
            k = int(np.searchsorted(-cut, -g2min, side='right'))       # shells with cut >= g2min (a prefix)
        if k == 0:
            return None, None
        if k not in cache:
            sel = np.sort(order[:k])
            sub = am.copy(deep=False)
            sub._bas = am._bas[sel]
            loc = am.ao_loc_nr()
            cols = np.concatenate([np.arange(loc[i], loc[i + 1]) for i in sel])
            cache.clear()                                               # keep one (consecutive chunks share it)
            cache[k] = (sub, cols)
        return cache[k]

    def build_thc(self, nmu):
        if getattr(self, 'stream', False):
            t = time.time()
            pts = self.piv[:nmu]
            rank = self._build_thc_stream(pts)
            self.pts = pts
            self.phimu = self.phi[pts]
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
            xi, rank = isdf_xi_ls(self.phi, pts)
            nmu = len(pts)
        elif self.selector == 'coarse':
            pts = self.piv[:nmu]
            xi, rank = isdf_xi_ls(self.phi, pts)
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
        s = np.sqrt(self.fG * self.w_vec / vol)
        W = Xh * s
        V = W.real @ W.real.T + W.imag @ W.imag.T                   # (xi_mu | xi_nu) with w_vec
        W = None
        # A_P,mu = (P|xi_mu)_SR, analytic FT of the aux functions, G blocks
        A = np.zeros((self.auxmol.nao, nmu))
        A_lr = np.zeros((self.auxmol.nao, nmu))
        sw = self.fG * self.w_sr / vol
        swl = self.fG * (self.w_vec - self.w_sr) / vol                # LR (G != 0; the G = 0 terms cancel)
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
        sq = np.sqrt(r.lr_vG / self.cell.vol)[:, None]
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

    def get_k_multi(self, dms, sym=None, pblk=20000, kblk=64):
        '''K (vector convention, split basis) of several real densities at once (route 1).  The density-independent
        intermediates -- B_p = (C^T A)_p of CX-SS, (M C)_P and C_P of CX-CX, the LR rows L_g -- are built once and
        applied to all densities stacked.  sym[n] = +1 / -1 marks a symmetric / antisymmetric density (its CX-SS
        transpose term follows from the direct one), 0 a general one.  Terms as in get_k_sparse_v1.'''
        nD = len(dms)
        dms = [np.ascontiguousarray(d, dtype=float) for d in dms]
        sym = [0] * nD if sym is None else list(sym)
        t = time.time()
        nao, sm = self.nao, self.smooth
        Pm = self.phimu
        Ks = []
        for d in dms:
            K = np.zeros((nao, nao))
            K[np.ix_(sm, sm)] = Pm.T @ ((Pm @ d[np.ix_(sm, sm)] @ Pm.T) * self.V) @ Pm
            Ks.append(K)
        t_ss = time.time() - t
        t = time.time()
        cxc = self.lr_cols
        ii, jj = self.ti[cxc], self.tj[cxc]
        off = ii != jj
        ii32, jj32 = ii.astype(np.int32), jj.astype(np.int32)
        Ccx = self.Cs[:, cxc].tocsc()
        blocks = self._cx_atom_blocks(Ccx)
        Bsr, Blr, Lcx = self.A, self.Xi_lr, self.Llr_cx
        # CX-SS: cross(D) = H(D) Pm with H[a] = sum_{p=(a,c)} B_p E_c, E = D[:, SS] Pm^T; for each density the
        # direct term and (unless sym) the transposed one
        Es, owner = [], []
        for n, d in enumerate(dms):
            Es.append(np.ascontiguousarray(d[:, sm] @ Pm.T)); owner.append((n, 0))
            if sym[n] == 0:
                Es.append(np.ascontiguousarray(d.T[:, sm] @ Pm.T)); owner.append((n, 1))
        Hs = [np.zeros((nao, len(Pm))) for _ in Es]
        for p0 in range(0, len(cxc), pblk):
            p1 = min(p0 + pblk, len(cxc))
            Bp = np.zeros((p1 - p0, Bsr.shape[1]))
            for rows, cols, Cd in blocks:
                a, b = np.searchsorted(cols, (p0, p1))
                if b > a:
                    Bp[cols[a:b] - p0] += Cd[:, a:b].T @ Bsr[rows]
            for H, E in zip(Hs, Es):
                pair_accum(H, Bp, E, ii32[p0:p1], jj32[p0:p1], off[p0:p1])
        Bp = Es = None
        cr = {}
        for (n, tr), H in zip(owner, Hs):
            cr[n, tr] = H @ Pm                                        # (nao, nao_s)
        Hs = None
        for n in range(nD):
            Ks[n][:, sm] += cr[n, 0]
            Ks[n][sm, :] += (cr[n, 1] if sym[n] == 0 else sym[n] * cr[n, 0]).T
        cr = None
        t_cross = time.time() - t
        t = time.time()
        # CX-CX (SR, non-robust): K[S_X] += sum_P (C_P D) W_P per aux atom X and chunk of P, all densities stacked
        nCX = len(cxc)
        for rows, cols, Cd in blocks:
            S = np.unique(np.concatenate([ii[cols], jj[cols]]))
            pos = np.full(nao, -1)
            pos[S] = np.arange(len(S))
            li, lj = pos[ii[cols]], pos[jj[cols]]
            DS = np.hstack([d[S] for d in dms])                       # (|S|, nD nao)
            nS = len(S)
            for c0 in range(0, len(rows), kblk):
                c1 = min(c0 + kblk, len(rows))
                nb = c1 - c0
                MC = np.zeros((nb, nCX))
                for rows2, cols2, Cd2 in blocks:
                    MC[:, cols2] += self.M[np.ix_(rows[c0:c1], rows2)] @ Cd2
                W = unpack_sym(MC, ii32, jj32, nao).reshape(nb * nao, nao)
                MC = None
                Cf = unpack_sym(Cd[c0:c1], li, lj, nS)
                CD = (Cf.reshape(nb * nS, nS) @ DS).reshape(nb, nS, nD, nao)
                CD = np.ascontiguousarray(CD.transpose(2, 1, 0, 3)).reshape(nD * nS, nb * nao)
                R = (CD @ W).reshape(nD, nS, nao)
                for n in range(nD):
                    Ks[n][S] += R[n]
                W = CD = Cf = R = None
        t_cc = time.time() - t
        t = time.time()
        # LR: sum_g L_g D L_g - L^SS_g D_SS L^SS_g, L_g the full LR rows, all densities stacked
        sm32 = np.asarray(sm, dtype=np.int32)
        gb = getattr(self, 'lr_batch', 16)
        Dh = np.hstack(dms)                                           # (nao, nD nao)
        Dsh = np.hstack([d[np.ix_(sm, sm)] for d in dms])
        nss = len(sm)
        for g0 in range(0, len(Lcx), gb):
            g1 = min(g0 + gb, len(Lcx))
            Ls = np.stack([(Pm.T * Blr[g]) @ Pm for g in range(g0, g1)])
            Lg = embed(unpack_sym(Lcx[g0:g1], ii32, jj32, nao), Ls, sm32)
            for L, Dst, n_, sub in ((Lg, Dh, nao, None), (Ls, Dsh, nss, sm)):
                k = len(L)
                L2 = L.reshape(k * n_, n_)
                T = (L2 @ Dst).reshape(k, n_, nD, n_).transpose(2, 1, 0, 3).reshape(nD * n_, k * n_)
                R = (T @ L2).reshape(nD, n_, n_)                      # sum_g L_g D L_g per density
                for n in range(nD):
                    if sub is None:
                        Ks[n] += R[n]
                    else:
                        Ks[n][np.ix_(sub, sub)] -= R[n]
            Lg = Ls = T = R = None
        self.timings.update({'K SS (THC)': t_ss, 'K CX-SS (SR)': t_cross, 'K CX-CX (SR)': t_cc,
                             'K LR': time.time() - t})
        return Ks

    def _cx_atom_blocks(self, Ccx):
        '''C (naux, nCX) as dense blocks per aux atom X: (aux rows of X, sorted CX columns with a nonzero in them,
        C[rows, cols]); the local fit makes these blocks nearly dense, so C^T A is a few BLAS GEMMs (cached).'''
        if getattr(self, '_cxb', None) is not None:
            return self._cxb
        Cr = Ccx.tocsr()
        pat = np.repeat([self.auxmol.bas_atom(i) for i in range(self.auxmol.nbas)], np.diff(self.auxmol.ao_loc_nr()))
        out = []
        for X in range(self.auxmol.natm):
            rows = np.where(pat == X)[0]
            if len(rows) == 0:
                continue
            sub = Cr[rows]
            cols = np.unique(sub.indices)
            if len(cols):
                out.append((rows, cols, np.ascontiguousarray(sub[:, cols].toarray())))
        self._cxb = out
        self.log('   CX atom blocks: %d, %.2e dense entries for %.2e nonzeros of C'
                 % (len(out), sum(c.size for _, _, c in out), Cr.nnz))
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
                else int(np.count_nonzero(self.C)),
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

    def __init__(self, cell, aux, alpha_c=4., nmu_factor=16., selector='coarse', omega=OMEGA, nuc_zeta=1e12,
                 verbose=True, local_rcut=None):
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
        self.th.stream = getattr(self, 'stream', False)
        self.th.stream_gb = getattr(self, 'stream_gb', 30)
        self.th.stream_max_blocks = getattr(self, 'stream_max_blocks', 1 << 30)
        self.th.piv_file = getattr(self, 'piv_file', None)
        for k in ('coarsen', 'blocked_pivots'):                        # point-selection options, if set
            if hasattr(self, k):
                setattr(self.th, k, getattr(self, k))
        self.th.kmax_factor = self.nmu_factor                         # pivots: only the N_mu prefix is used
        self.th.build_common()
        self.th.build_thc(int(round(self.nmu_factor * len(self.th.smooth))))
        self.S = cell.pbc_intor('int1e_ovlp')
        self._built = True
        return self

    @property
    def c_g0(self):
        return self.th.r.c_g0

    def g0_constant(self, nelectron):
        return self.th.r.g0_constant(nelectron)

    def get_nuc(self):
        return self.T.T @ self.th.r.get_nuc() @ self.T

    def energy_nuc(self):
        return self.th.r.energy_nuc()

    def get_jk_mo(self, mol, mo_coeff, mo_occ):
        '''Spinor J and K for the density sum_i n_i |i><i| (spinor MOs, n_i >= 0) by occ-RI (no G = 0 terms).'''
        if not self._built:
            self.build()
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
            Kr = self.th.get_k_multi([Daa.real, Daa.imag, Dbb.real, Dbb.imag, Dab.real, Dab.imag],
                                     sym=[1, -1, 1, -1, 0, 0])
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
