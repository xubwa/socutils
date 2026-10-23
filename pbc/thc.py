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
from socutils.pbc.bench.hybrid_k import _inv_metric, k_from_factors, build as rsdf_build

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


def isdf_xi(L, piv, nmu):
    '''Lu-Ying least-squares interpolation vectors for the first nmu pivots, xi (nmu, ngrid):
    xi = G[:, P] G[P, P]^-1 = L^T L[:, P]^-T with the Cholesky factor (L[:, P] is upper triangular in pivot
    order), i.e. xi^T = L[:, P]^-1 L -- no explicit inverse of the ill-conditioned G[P, P].'''
    Lk = L[:nmu]
    U = Lk[:, piv[:nmu]]
    return scipy.linalg.solve_triangular(U, Lk, lower=False, check_finite=False), nmu


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
    def __init__(self, cell, aux, alpha_c, omega=OMEGA, verbose=True):
        self.cell, self.aux, self.ac, self.omega = cell, aux, alpha_c, omega
        self.verbose = verbose
        self.timings = {}

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ---- pieces independent of N_mu
    def build_common(self, kmax_factor=20):
        cell, ac = self.cell, self.ac
        t = time.time()
        self.r = r = rsdf_build(cell, self.aux)
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
        coords = cell.get_uniform_grids(self.mesh)
        self.phi = cell.pbc_eval_gto('GTOval_sph', coords)[:, self.smooth]  # (ng, nao_s)
        self.ng = len(coords)
        self.timings['grid'] = time.time() - t
        t = time.time()
        self.piv, self.L = pivots(self.phi, min(self.ng, kmax_factor * len(self.smooth)))
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
        K = np.zeros((nao, nao))
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
