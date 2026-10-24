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
import time
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
    def __init__(self, cell, aux, alpha_c, omega=OMEGA, verbose=True, selector='cholesky', nuc_zeta=1e12):
        '''selector: 'cholesky' (column-by-column pivoted Cholesky of the grid Gram matrix, O(k^2 ngrid), reference),
        'coarse' (the same on the grid coarsened by self.coarsen per direction, least-squares xi on the full grid),
        'cholesky_blocked' (blocked candidates, BLAS-3; worse points, not faster), 'kmeans'
        (weighted K-means on the grid + least-squares xi, BLAS-3; fast but poor points for all-electron smooth
        functions) or 'kmeans+chol' (K-means candidates, cand_factor per point, then the pivots of a blocked
        pivoted Cholesky among them; least-squares xi).'''
        self.cell, self.aux, self.ac, self.omega = cell, aux, alpha_c, omega
        self.selector = selector
        self.cand_factor = 4.        # kmeans+chol: K-means candidates per point
        self.coarsen = 2             # coarse: grid coarsening per direction for the point selection
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
        r = RSDF(cell, self.aux, omega=self.omega, precision=1e-10, nuc_zeta=self.nuc_zeta, lr='gspace')
        r.keep_raw = True
        r.cderi_in_memory = True
        r.verbose = 0
        self.r = r.build()
        nao = self.nao = cell.nao
        self.J = r.int3c_sr
        self.M = r.metric_sr
        self.nkept = r.cderi.shape[0] - r.nlr
        self.Llr = r.cderi[self.nkept:]
        self.timings['rsdf'] = time.time() - t
        aohard = np.repeat([cell.bas_exp(i).min() >= ac for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
        self.aoat = np.repeat([cell.bas_atom(i) for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
        self.smooth = np.where(~aohard)[0]
        ti, tj = np.tril_indices(nao)
        self.ti, self.tj = ti, tj
        self.cx = aohard[ti] | aohard[tj]
        self.ss = np.where(~self.cx)[0]
        self.auxmol = df_addons.make_auxmol(cell, self.aux)
        # local fit of the CX pairs (level 2)
        t = time.time()
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
        ew, U = scipy.linalg.eigh(self.M)
        self.Y = ((U * np.sqrt(np.maximum(ew, 0))) @ U.T) @ C
        self.timings['local fit'] = time.time() - t
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
            pc, _ = pivots(phic, min(len(cc), kmax_factor * len(self.smooth)))
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
        return self

    # ---- THC factors for N_mu points
    def build_thc(self, nmu):
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
        sw = self.fG * self.w_sr / vol
        for g0 in range(0, len(self.Gv), 20000):
            g1 = min(g0 + 20000, len(self.Gv))
            Pg = ft_ao(self.auxmol, self.Gv[g0:g1])                 # (nG, naux)
            A += ((Pg.conj() * sw[g0:g1, None]).T @ Xh[:, g0:g1].T).real
        t_v = time.time() - t
        self.pts, self.V, self.A = pts, V, A
        self.phimu = self.phi[pts]                                  # (N_mu, nao_s)
        self.timings.update({'xi': t_xi, 'V, A (FFT)': t_v})
        return rank

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
        J3 = self.J.copy()
        J3[:, self.ss] = self.j_thc_ss()
        K += k_from_factors(self.C, J3, dm, nao) + k_from_factors(J3, self.C, dm, nao) \
            - k_from_factors(self.Y, self.Y, dm, nao)
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
        for p0 in range(0, naux, blk):
            p1 = min(p0 + blk, naux)
            Cb = self.C[p0:p1]
            if not Cb.any():
                continue
            J3 = self.J[p0:p1].copy()
            J3[:, self.ss] = self.A[p0:p1] @ X
            HC = half(Cb)
            acc(HC, half(J3), 1., swap=True)                           # (C, J3) + (J3, C)
            HY = half(self.Y[p0:p1])
            acc(HY, HY, -1.)
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
        vj = np.array([self.get_j(d) for d in dms]) if with_j else None
        vk = np.array([self.get_k(d) for d in dms]) if with_k else None
        return vj, vk

    def storage(self):
        nmu, nao_s = len(self.pts), len(self.smooth)
        return {'V': nmu * nmu, 'phi(r_mu)': nmu * nao_s, '(P|xi)': self.A.size,
                'CX local C (nonzero)': int(np.count_nonzero(self.C)),
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
                 verbose=True):
        self.cell, self.aux, self.alpha_c, self.nmu_factor = cell, aux, alpha_c, nmu_factor
        self.selector, self.omega, self.nuc_zeta, self.verbose = selector, omega, nuc_zeta, verbose
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
                              self.nuc_zeta).build_common()
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
