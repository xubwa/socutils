#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
# Gamma-point Coulomb matrix J of socutils.pbc.thc without three-centre integrals (HANDOFF Open item 2, route 1):
# smooth density on the uniform grid, compact density by atom-centred one-centre expansions with compensation
# charges (the Coulomb part of GAPW, Lippert, Hutter, Parrinello, Theor. Chem. Acc. 103, 124 (1999)).
#
# Basis split at alpha_c (smooth / compact functions).  The density is split exactly,
#     rho = n_SS + sum_A rho_A,   rho_A = sum_{(a,b) owned by A} D_ab phi_a phi_b,
# a pair with a compact function being owned by the atom of its compact function (two compact functions: the one
# with the larger index, as in thc_local).  rho_A lives in the compact region of A and is sampled on A's atomic
# grid.  With its multipoles Q^A_lm (l <= L) and smooth compensation Gaussians g_lm (exponent eta,
# int g_lm r^l Y_lm = 1),
#     rho = [n_SS + sum_A rho^_A] + sum_A (rho_A - rho^_A),     rho^_A = sum_lm Q^A_lm g_lm(r - R_A),
# the bracket is smooth: its potential v_grid is an FFT on the THC grid with the kernel of the RSDF vectors
# (4 pi / G^2, pi / w^2 at G = 0); rho_A - rho^_A has no multipoles up to L, its potential v_A is local and is
# solved by radial Poisson on A's grid (real spherical harmonics up to lmax, scaled cumulative quadrature).
# Matrix elements:
#     SS pair:          int phi_a phi_b v_grid (uniform grid) + sum_A int phi_a phi_b v_A (A's grid, all space)
#     pair owned by A:  int phi_a^(0) phi_b [v_grid + v_A + sum_{B, images} v_B] on A's grid,
# with v_grid interpolated to the atomic grid (zero-padded spectrum + local interpolation) and the neighbours'
# local potentials (and the images of A's own) from their radial lm expansions, plus the constant that restores
# the G = 0 average of the bare-Coulomb potential of the neutral rho_A - rho^_A (+ c S).  No three-centre
# integrals, no fit.  Quartz (alpha_c 4): J vs RSDF converges with the radial grid, max element 2.6e-4 / 2.1e-5 /
# 4.1e-6 at 150 / 300 / 600 radial points (RSDF vs exact: 1.7e-6).
#
import numpy as np
import scipy.fft
import scipy.special
from scipy.interpolate import CubicSpline
from pyscf.dft import radi, gen_grid
from pyscf.symm import sph


def _ylm_transforms(lmax, unit, wang):
    '''Per l, the (2l+1, 2l+1) Loewdin transform making pyscf's real_sph_vec orthonormal on the sphere (from the
    Lebedev grid unit/wang, weights summing to 4 pi).'''
    ys = sph.real_sph_vec(unit, lmax, reorder_p=True)
    out = []
    for y in ys:
        y = np.asarray(y, dtype=float)
        e, v = np.linalg.eigh((y * wang) @ y.T)
        out.append((v / np.sqrt(e)) @ v.T)
    return out


def _real_ylm_at(dirs, lmax, unit, wang):
    '''Orthonormal real spherical harmonics at the unit vectors dirs: list per l of (2l+1, n).'''
    tr = _ylm_transforms(lmax, unit, wang)
    return [t @ np.asarray(y, dtype=float) for t, y in zip(tr, sph.real_sph_vec(dirs, lmax, reorder_p=True))]


def _real_ylm(unit, lmax, wang):
    return _real_ylm_at(unit, lmax, unit, wang)


class GAPWJ:
    def __init__(self, th, L=4, eta=2.0, lmax=12, nrad=150, nang=590):
        '''th: a THCExchange after build_common (split cell, smooth functions on the uniform grid, kernels).'''
        self.th, self.L, self.eta, self.lmax, self.nrad, self.nang = th, L, eta, lmax, nrad, nang
        self.S = th.cell.pbc_intor('int1e_ovlp')
        self.r_compact = 4.0          # bohr: where the compact densities live (neighbours' local potentials there)
        cell = th.cell
        nao = cell.nao
        compact = np.ones(nao, bool)
        compact[th.smooth] = False
        aoat = th.aoat
        # owner atom of every pair (-1: SS)
        owner = -np.ones((nao, nao), dtype=int)
        a_, b_ = np.meshgrid(np.arange(nao), np.arange(nao), indexing='ij')
        both = compact[a_] & compact[b_]
        owner[compact[a_] & ~compact[b_]] = aoat[a_][compact[a_] & ~compact[b_]]
        owner[~compact[a_] & compact[b_]] = aoat[b_][~compact[a_] & compact[b_]]
        big = np.maximum(a_, b_)
        owner[both] = aoat[big][both]
        self.owner = owner
        # the home factor of an owned pair: its compact function (two compact functions: the larger index); in the
        # Gamma matrix element int_cell phi_a^G phi_b^G v = int_space phi_home^(0) phi_other^G v the home factor is
        # the cell-0 function only, which keeps rho_A in the compact region of A (no image densities)
        home = np.zeros((nao, nao), bool)
        home[compact[a_] & ~compact[b_]] = True
        home[both & (a_ > b_)] = True
        home[both & (a_ == b_)] = True
        self.home = home
        self.atoms = sorted(set(owner[owner >= 0].ravel().tolist()))
        # atomic grids: radial (Treutler-Ahlrichs) x Lebedev, centred on each owning atom
        ang = gen_grid.MakeAngularGrid(nang)
        self.unit, self.wang = ang[:, :3], 4 * np.pi * ang[:, 3]        # pyscf's Lebedev weights sum to 1
        self.Y = _real_ylm(self.unit, max(lmax, L), self.wang)                  # per l (2l+1, nang)
        X = cell.atom_coords()
        self.grids, self.rads, self.wrs = {}, {}, {}
        for A in self.atoms:
            r, dr = radi.treutler_ahlrichs(nrad, cell.atom_charge(A))
            o = np.argsort(r)
            r, wr = r[o], (dr * r ** 2)[o]
            self.rads[A] = r
            self.wrs[A] = wr
            pts = (r[:, None, None] * self.unit[None]).reshape(-1, 3) + X[A]
            w = (wr[:, None] * self.wang[None]).ravel()
            phi0 = cell.eval_gto('GTOval_sph', pts)                 # cell-0 functions (no lattice sum)
            phi0[:, ~(compact & (aoat == A))] = 0.                     # only A's compact functions are home factors
            self.grids[A] = (pts, w, cell.pbc_eval_gto('GTOval_sph', pts), phi0)
        # compensation Gaussians: normalisation N_l (int N_l r^l e^{-eta r^2} Y_lm r^l Y_lm = 1)
        self.Nl = np.array([1. / (scipy.special.gamma(l + 1.5) / (2 * eta ** (l + 1.5))) for l in range(L + 1)])
        # their FTs on the THC half spectrum (without the phase e^{-iG.R_A}), per l (2l+1, nG)
        Gv = th.Gv
        G = np.linalg.norm(Gv, axis=1)
        Gu = np.where(G[:, None] > 1e-12, Gv / np.maximum(G, 1e-300)[:, None], np.array([[0., 0., 1.]]))
        YG = _real_ylm_at(Gu, L, self.unit, self.wang)
        self.gG = []
        for l in range(L + 1):
            rad = 4 * np.pi * np.sqrt(np.pi) * G ** l * np.exp(-G ** 2 / (4 * eta)) / (2 ** (l + 2) * eta ** (l + 1.5))
            self.gG.append(self.Nl[l] * (-1j) ** l * rad[None] * YG[l])

    def _phase(self, A):
        return np.exp(-1j * self.th.Gv @ self.th.cell.atom_coord(A))

    def _poisson(self, f, A, keep=None):
        '''Potential of a density f on A's atomic grid ((nrad*nang,)), expanded to lmax.  Radial part per lm:
            v_lm(r) = 4 pi / (2l+1) [ r^-(l+1) int_0^r f s^(l+2) ds + r^l int_r^inf f s^(1-l) ds ],
        as scaled cumulative sums over the radial intervals, R_i = (r_{i-1}/r_i)^(l+1) R_{i-1} + r_i^-(l+1) seg_i and
        S_i = (r_i/r_{i+1})^l S_{i+1} + r_i^l seg_i (factors <= 1).  The interval integrals are done in the grid's
        own variable t (the node index; the Treutler mapping makes steep core densities smooth in t): cubic splines
        of log r(t) and of f_lm(t), Gauss-Legendre in t.'''
        r = self.rads[A]
        n = len(r)
        f = f.reshape(n, -1)
        v = np.zeros_like(f)
        tn = np.arange(n, dtype=float)
        lr = CubicSpline(tn, np.log(r))
        xg, wg = np.polynomial.legendre.leggauss(8)
        tq = (tn[:-1, None] + .5 * (xg[None] + 1))                       # interval [i, i+1] -> nodes i, i+1
        wq = .5 * wg[None] * np.ones_like(tq)
        rq = np.exp(lr(tq))
        drq = rq * lr(tq, 1)                                              # dr/dt
        # the first interval [0, r_0]: in s directly (f ~ f(r_0) there)
        x0 = .5 * (xg + 1) * r[0]
        w0 = .5 * wg * r[0]
        for l in range(self.lmax + 1):
            Yl = self.Y[l]
            flm = f @ (Yl * self.wang).T                                 # (n, 2l+1)
            fs = CubicSpline(tn, flm, axis=0)
            fq = fs(tq.ravel()).reshape(tq.shape + (flm.shape[1],))      # (n-1, 8, 2l+1)
            # inner integrals: R_i = r_i^-(l+1) int_0^{r_i} f s^(l+2) ds
            seg_first = np.einsum('q,m->m', w0 * (x0 / r[0]) ** (l + 1) * x0, flm[0] if l == 0 else np.zeros(flm.shape[1]))
            segs = np.einsum('iq,iqm->im', wq * (rq / r[1:, None]) ** (l + 1) * rq * drq, fq)   # interval i -> i+1
            R = np.zeros_like(flm)
            R[0] = seg_first
            for i in range(1, n):
                R[i] = (r[i - 1] / r[i]) ** (l + 1) * R[i - 1] + segs[i - 1]
            # outer integrals: S_i = r_i^l int_{r_i}^inf f s^(1-l) ds
            sego = np.einsum('iq,iqm->im', wq * (r[:-1, None] / rq) ** l * rq * drq, fq)       # r_i^l int_i^{i+1}
            S = np.zeros_like(flm)
            for i in range(n - 2, -1, -1):
                S[i] = sego[i] + (r[i] / r[i + 1]) ** l * S[i + 1]
            vlm = 4 * np.pi / (2 * l + 1) * (R + S)
            v += vlm @ Yl
            if keep is not None:
                keep.append(vlm)
        return v.ravel()

    def _vgrid_at(self, vG, pts, rad, r_max=8., exact=False, blk=4000):
        '''v_grid(r) = 1/V sum_G v(G) e^{iGr} at the points with rad < r_max (bohr; 0 beyond: only where the compact
        densities live): tricubic Lagrange interpolation (4^3 stencil, periodic, in fractional coordinates) of v on
        the uniform grid (v is smooth; exact=True: direct Fourier sum over the half spectrum, for checks).'''
        th = self.th
        out = np.zeros(len(pts))
        sel = np.where(rad < r_max)[0]
        if exact:
            c = th.fG * vG / th.cell.vol
            for p0 in range(0, len(sel), blk):
                idx = sel[p0:p0 + blk]
                out[idx] = (np.exp(1j * pts[idx] @ th.Gv.T) @ c).real
            return out
        if getattr(self, '_vr_cache', None) is None or self._vr_cache[0] is not vG:
            self._vr_cache = (vG, self._oversampled(vG))
        vr, mesh = self._vr_cache[1]
        frac = pts[sel] @ np.linalg.inv(np.asarray(th.cell.lattice_vectors())) * mesh   # grid units
        i0 = np.floor(frac).astype(int)
        t = frac - i0                                                  # in [0, 1)
        nodes = np.arange(-2, 4)                                       # 6-point Lagrange (quintic)
        w = np.ones(t.shape + (6,))
        for k, xk in enumerate(nodes):
            for xj in nodes:
                if xj != xk:
                    w[..., k] *= (t - xj) / (xk - xj)
        val = np.zeros(len(sel))
        for a in range(6):
            ia = (i0[:, 0] + nodes[a]) % mesh[0]
            for b in range(6):
                ib = (i0[:, 1] + nodes[b]) % mesh[1]
                wab = w[:, 0, a] * w[:, 1, b]
                for c in range(6):
                    ic = (i0[:, 2] + nodes[c]) % mesh[2]
                    val += wab * w[:, 2, c] * vr[ia, ib, ic]
        out[sel] = val
        return out

    def _oversampled(self, vG, f=2):
        '''v(r) on the uniform grid refined f times per direction (zero padding of the spectrum: exact).'''
        th = self.th
        m = np.asarray(th.mesh)
        M = f * m
        small = vG.reshape(m[0], m[1], m[2] // 2 + 1).copy()
        nz = m[2] // 2 + 1
        if m[2] % 2 == 0:
            small[:, :, -1] *= .5                                      # Nyquist plane: counted twice on the big grid
        k0 = np.rint(np.fft.fftfreq(m[0]) * m[0]).astype(int)
        k1 = np.rint(np.fft.fftfreq(m[1]) * m[1]).astype(int)
        big = np.zeros((M[0], M[1], M[2] // 2 + 1), dtype=complex)
        big[np.ix_(k0 % M[0], k1 % M[1], np.arange(nz))] = small
        # even x / y meshes: the Nyquist row (k = -m/2) is split between -m/2 and +m/2
        if m[0] % 2 == 0:
            j = m[0] // 2
            row = big[(-j) % M[0]].copy() * .5
            big[(-j) % M[0]] = row
            big[j % M[0]] += row
        if m[1] % 2 == 0:
            j = m[1] // 2
            col = big[:, (-j) % M[1]].copy() * .5
            big[:, (-j) % M[1]] = col
            big[:, j % M[1]] += col
        vr = scipy.fft.irfftn(big, s=tuple(M), workers=-1) * (np.prod(M) / th.cell.vol)
        return vr, M

    def _vloc_at(self, B, vlms, pts, r_max=10.):
        '''Local potential of atom B (radial coefficients vlms from _poisson) at arbitrary points, with the periodic
        images of B within r_max (bohr) of the points: radial cubic spline x real spherical harmonics.'''
        cell = self.th.cell
        r = self.rads[B]
        if getattr(self, '_ytr', None) is None:
            from socutils.pbc.thc_gapw import _ylm_transforms
            self._ytr = _ylm_transforms(self.lmax, self.unit, self.wang)
        rr = np.concatenate([[0.], r])
        spl = [CubicSpline(rr, np.vstack([v[:1] if l == 0 else np.zeros((1, v.shape[1])), v]), axis=0)
               for l, v in enumerate(vlms)]
        A_ = np.asarray(cell.lattice_vectors())
        T = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]) @ A_
        out = np.zeros(len(pts))
        for t in T:
            d = pts - (cell.atom_coord(B) + t)
            dist = np.linalg.norm(d, axis=1)
            sel = np.where((dist < r_max) & (dist > 1e-12))[0]
            if len(sel) == 0:
                continue
            u = d[sel] / dist[sel, None]
            ys = sph.real_sph_vec(u, self.lmax, reorder_p=True)
            for l in range(self.lmax + 1):
                out[sel] += np.einsum('pm,mp->p', spl[l](dist[sel]), self._ytr[l] @ np.asarray(ys[l], dtype=float))
        return out

    def _vloc_at_home(self, B, vlms, pts, r_max=10.):
        '''The home copy of _vloc_at (no images).'''
        cell = self.th.cell
        save = cell.lattice_vectors
        d = pts - cell.atom_coord(B)
        dist = np.linalg.norm(d, axis=1)
        out = np.zeros(len(pts))
        sel = np.where((dist < r_max) & (dist > 1e-12))[0]
        if len(sel) == 0:
            return out
        r = self.rads[B]
        rr = np.concatenate([[0.], r])
        u = d[sel] / dist[sel, None]
        ys = sph.real_sph_vec(u, self.lmax, reorder_p=True)
        for l, v in enumerate(vlms):
            spl = CubicSpline(rr, np.vstack([v[:1] if l == 0 else np.zeros((1, v.shape[1])), v]), axis=0)
            out[sel] += np.einsum('pm,mp->p', spl(dist[sel]), self._ytr[l] @ np.asarray(ys[l], dtype=float))
        return out

    def get_j(self, dm):
        '''J (vector convention) in the split basis for a real symmetric dm.'''
        th, L, cell = self.th, self.L, self.th.cell
        nao, sm = th.nao, th.smooth
        vol, ng, mesh = cell.vol, th.ng, th.mesh
        hshape = (mesh[0], mesh[1], mesh[2] // 2 + 1)
        dm = .5 * (dm + dm.T)
        # smooth density on the uniform grid
        n = np.einsum('ri,ri->r', th.phi @ dm[np.ix_(sm, sm)], th.phi)
        nG = scipy.fft.rfftn(n.reshape(mesh), workers=-1).reshape(-1) * (vol / ng)
        # compact densities on the atomic grids, multipoles, compensation charges
        rhoA, QA = {}, {}
        for A in self.atoms:
            pts, w, phi, phi0 = self.grids[A]
            DA = np.where((self.owner == A) & self.home, dm, 0.)
            DA[np.diag_indices(nao)] *= .5                             # diagonal pairs counted once
            rho = 2 * np.einsum('ri,ri->r', phi0 @ DA, phi)
            rhoA[A] = rho
            rad = np.repeat(self.rads[A], len(self.wang))
            QA[A] = [((rho * w * rad ** l)[None] * np.tile(self.Y[l], len(self.rads[A]))).sum(1) for l in range(L + 1)]
            ph = self._phase(A)
            for l in range(L + 1):
                nG = nG + ph * (QA[A][l] @ self.gG[l])
        vG = th.w_vec * nG
        vr = scipy.fft.irfftn(vG.reshape(hshape), s=mesh, workers=-1).reshape(-1)   # v_grid(r) vol / ng
        J = np.zeros((nao, nao))
        J[np.ix_(sm, sm)] = th.phi.T @ (vr[:, None] * th.phi)
        vloc, vlms = {}, {}
        c_avg = 0.
        for A in self.atoms:
            rad = np.repeat(self.rads[A], len(self.wang))
            rhat = np.zeros_like(rhoA[A])
            for l in range(L + 1):
                rhat += self.Nl[l] * rad ** l * np.exp(-self.eta * rad ** 2) * (QA[A][l] @ np.tile(self.Y[l], len(self.rads[A])))
            vlms[A] = []
            vloc[A] = self._poisson(rhoA[A] - rhat, A, keep=vlms[A])     # local potential of rho_A - rho^_A
            # the bare-Coulomb potential of the neutral rho_A - rho^_A averages to -(2 pi / 3V) int r^2 (rho_A -
            # rho^_A) over the cell; the kernel of the grid part has no G = 0 average: add the constant back
            c_avg += 2 * np.pi / (3 * vol) * (self.grids[A][1] * rad ** 2 * (rhoA[A] - rhat)).sum()
        X = cell.atom_coords()
        for A in self.atoms:
            pts, w, phi, phi0 = self.grids[A]
            rad = np.repeat(self.rads[A], len(self.wang))
            u = self._vgrid_at(vG, pts, rad)                          # smooth potential (interpolated)
            # neighbours' local potentials where the compact density of A lives
            vn = np.zeros_like(rad)
            near = np.where(rad < self.r_compact)[0]
            for B in self.atoms:
                if B != A or True:
                    vB = self._vloc_at(B, vlms[B], pts[near])
                    if B == A:                                         # A's own: its images only (the home copy is vloc)
                        vB -= self._vloc_at_home(A, vlms[A], pts[near])
                    vn[near] += vB
            # pairs owned by A: smooth + own local + neighbours' local potentials
            JA = phi0.T @ ((w * (u + vloc[A] + vn))[:, None] * phi)       # (home a, any b)
            own = (self.owner == A) & self.home
            J[own] += JA[own]
            J[own.T] += JA.T[own.T]
            dg = np.diag(own)
            J[dg, dg] -= np.diag(JA)[dg]                               # diagonal added twice above
            # SS pairs: the local potential of A (all space, on A's grid)
            JS = phi[:, sm].T @ ((w * vloc[A])[:, None] * phi[:, sm])
            J[np.ix_(sm, sm)] += JS
        J += c_avg * self.S                                           # constant potential: c S_ab
        return J
