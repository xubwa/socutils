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
from pyscf import lib
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


def _rowdot(A, B):
    '''sum_j A[r, j] B[r, j] (OpenMP kernel of libsocthc when built).'''
    from socutils.pbc.thc import _libthc, _ptr
    lib_ = _libthc()
    if lib_ is None or not hasattr(lib_, 'SOCTHCrowdot'):
        return np.einsum('ri,ri->r', A, B)
    import ctypes
    A, B = np.ascontiguousarray(A, dtype=float), np.ascontiguousarray(B, dtype=float)
    out = np.empty(len(A))
    lib_.SOCTHCrowdot(_ptr(out), _ptr(A), _ptr(B), ctypes.c_int(A.shape[0]), ctypes.c_int(A.shape[1]))
    return out


def _scale_rows(w, A):
    '''w[:, None] * A (OpenMP kernel of libsocthc when built).'''
    from socutils.pbc.thc import _libthc, _ptr
    lib_ = _libthc()
    if lib_ is None or not hasattr(lib_, 'SOCTHCscale_rows'):
        return w[:, None] * A
    import ctypes
    A, w = np.ascontiguousarray(A, dtype=float), np.ascontiguousarray(w, dtype=float)
    out = np.empty_like(A)
    lib_.SOCTHCscale_rows(_ptr(out), _ptr(A), _ptr(w), ctypes.c_int(A.shape[0]), ctypes.c_int(A.shape[1]))
    return out


def _wdens(phi, D, blk=262144):
    '''rho(r) = sum_ij phi_i(r) D_ij phi_j(r) in grid blocks.'''
    out = np.empty(len(phi))
    for g0 in range(0, len(phi), blk):
        p = phi[g0:g0 + blk]
        out[g0:g0 + blk] = _rowdot(p @ D, p)
    return out


def _wmat(phi_a, w, phi_b=None, blk=262144):
    '''sum_r phi_a(r)^T w(r) phi_b(r) in grid blocks.'''
    phi_b = phi_a if phi_b is None else phi_b
    out = np.zeros((phi_a.shape[1], phi_b.shape[1]))
    for g0 in range(0, len(phi_a), blk):
        out += phi_a[g0:g0 + blk].T @ _scale_rows(w[g0:g0 + blk], phi_b[g0:g0 + blk])
    return out


class GAPWJ:
    def __init__(self, th, L=4, eta=2.0, lmax=12, nrad=None, nang=302):
        '''th: a THCExchange after build_common (split cell, smooth functions on the uniform grid, kernels).'''
        self.th, self.L, self.eta, self.lmax, self.nrad, self.nang = th, L, eta, lmax, nrad, nang
        self.S = th.cell.pbc_intor('int1e_ovlp')
        self.r_compact = 4.0          # bohr: where the compact densities live (neighbours' local potentials there)
        self.r_store = 7.0            # bohr: atomic-grid points kept (densities and local potentials live inside)
        self.f_thr = 1e-10            # functions kept on an atomic grid: max |phi| above this
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
        # MPI (th.comm): each rank builds and integrates the atomic grids of its atoms only (round-robin)
        self.comm = getattr(th, 'comm', None)
        rank, size = (self.comm.Get_rank(), self.comm.Get_size()) if self.comm is not None else (0, 1)
        self.my_atoms = [A for i, A in enumerate(self.atoms) if i % size == rank]
        # atomic grids: radial (Treutler-Ahlrichs) x Lebedev, centred on each owning atom
        ang = gen_grid.MakeAngularGrid(nang)
        self.unit, self.wang = ang[:, :3], 4 * np.pi * ang[:, 3]        # pyscf's Lebedev weights sum to 1
        self.Y = _real_ylm(self.unit, max(lmax, L), self.wang)                  # per l (2l+1, nang)
        X = cell.atom_coords()
        self.grids, self.rads, self.wrs = {}, {}, {}
        for A in self.my_atoms:
            nr, scheme = self.radial_grid(cell.atom_charge(A))
            r, dr = getattr(radi, scheme)(nr, cell.atom_charge(A))
            o = np.argsort(r)
            r, wr = r[o], (dr * r ** 2)[o]
            self.rads[A] = r
            self.wrs[A] = wr
            pts = (r[:, None, None] * self.unit[None]).reshape(-1, 3) + X[A]
            w = (wr[:, None] * self.wang[None]).ravel()
            # stored: the points within r_store (the compact densities and the local potentials live there) and
            # the functions not negligible on them; the home factors are A's compact functions (cell 0 only)
            rad_ = np.repeat(r, len(self.wang))
            inner = np.where(rad_ < self.r_store)[0]
            cA = np.where(compact & (aoat == A))[0]
            phi_in = cell.pbc_eval_gto('GTOval_sph', pts[inner])
            cols = np.where(np.abs(phi_in).max(axis=0) > self.f_thr)[0]
            phi_in = np.ascontiguousarray(phi_in[:, cols])
            phi0c = np.ascontiguousarray(cell.eval_gto('GTOval_sph', pts[inner])[:, cA])
            self.grids[A] = (pts, w, inner, cols, cA, phi_in, phi0c)
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
            vr, mesh = self._oversampled(vG)
            if getattr(self, 'interp', 'spline') == 'spline':
                from scipy.ndimage import spline_filter
                vr = spline_filter(vr, order=5, mode='grid-wrap')        # quintic B-spline coefficients, once
            self._vr_cache = (vG, (vr, mesh))
        vr, mesh = self._vr_cache[1]
        frac = pts[sel] @ np.linalg.inv(np.asarray(th.cell.lattice_vectors())) * mesh   # grid units
        if getattr(self, 'interp', 'spline') == 'spline':
            from scipy.ndimage import map_coordinates
            from concurrent.futures import ThreadPoolExecutor
            nt = lib.num_threads()
            ch = np.array_split(np.arange(len(sel)), max(1, min(nt, len(sel) // 2000)))
            f = lambda c: map_coordinates(vr, frac[c].T, order=5, mode='grid-wrap', prefilter=False)
            with ThreadPoolExecutor(len(ch)) as ex:
                for c, v in zip(ch, ex.map(f, ch)):
                    out[sel[c]] = v
            return out
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

    def _vloc_spline(self, B, vlms, tol=None):
        '''One cubic spline for all lm radial functions of B's local potential, and its support radius (beyond it
        every |v_lm| < tol of the largest: the neutral rho_B - rho^_B has no potential there).'''
        if tol is None:
            tol = getattr(self, 'vloc_tol', 1e-9)
        r = self.rads[B]
        V = np.hstack(vlms)                                             # (nrad, sum_l 2l+1)
        big = np.abs(V).max(axis=1)
        inside = np.where(big > tol * big.max())[0]
        r_v = r[inside[-1]] if len(inside) else 0.
        if getattr(self, 'debug', False):
            prof = ' '.join('%.0e@%.1f' % (big[i], r[i]) for i in np.linspace(0, len(r) - 1, 12).astype(int))
            print('   vloc atom %d: support %.1f bohr (rmax %.1f); max|v_lm|(r): %s' % (B, r_v, r[-1], prof), flush=True)
        rr = np.concatenate([[0.], r])
        V0 = np.vstack([np.concatenate([vlms[0][:1].ravel(), np.zeros(V.shape[1] - 1)])[None], V])
        lm = np.where(np.abs(V).max(axis=0) > tol * big.max())[0]       # lm components above tol anywhere
        return CubicSpline(rr, V0[:, lm], axis=0), r_v, lm

    def _vloc_eval(self, spl, d):
        '''sum_lm v_lm(|d|) Y_lm(d/|d|) at displacements d (n, 3).'''
        if getattr(self, '_ytr', None) is None:
            self._ytr = _ylm_transforms(self.lmax, self.unit, self.wang)
        dist = np.linalg.norm(d, axis=1)
        dist = np.maximum(dist, 1e-12)
        ys = sph.real_sph_vec(d / dist[:, None], self.lmax, reorder_p=True)
        Y = np.vstack([t @ np.asarray(y, dtype=float) for t, y in zip(self._ytr, ys)])   # (sum 2l+1, n)
        return np.einsum('pm,mp->p', spl(dist), Y)

    def _vloc_at(self, B, vlms, pts, home=True, center=None, radius=None):
        '''Local potential of atom B and its periodic images at the points (only within B's support radius).'''
        cell = self.th.cell
        key = (B, id(vlms))
        if getattr(self, '_vspl', None) is None or self._vspl[0] != key:
            self._vspl = (key, self._vloc_spline(B, vlms))
        spl, r_v, lm = self._vspl[1]
        out = np.zeros(len(pts))
        if r_v <= 0:
            return out
        A_ = np.asarray(cell.lattice_vectors())
        n = [int(np.ceil(r_v / np.linalg.norm(A_[i]))) + 1 for i in range(3)]
        T = np.array([[i, j, k] for i in range(-n[0], n[0] + 1) for j in range(-n[1], n[1] + 1)
                      for k in range(-n[2], n[2] + 1)]) @ A_
        for t in T:
            if not home and np.linalg.norm(t) < 1e-12:
                continue
            if center is not None and np.linalg.norm(cell.atom_coord(B) + t - center) > r_v + radius:
                continue                                              # no point within the support of this image
            ck = (B, home, center is not None and tuple(np.round(center, 8)), len(pts), tuple(np.round(t, 8)), r_v)
            geo = self._geo.get(ck) if getattr(self, '_geo', None) is not None else None
            if geo is None:
                d = pts - (cell.atom_coord(B) + t)
                sel = np.where(np.einsum('px,px->p', d, d) < r_v ** 2)[0]
                if len(sel) == 0:
                    geo = (sel, None, None)
                else:
                    dist, Y = self._ylm_at(d[sel])
                    geo = (sel, dist, Y)
                    self._cache_geo(ck, geo)
            sel, dist, Y = geo
            if len(sel):
                out[sel] += np.einsum('pm,mp->p', spl(dist), Y if len(lm) == len(Y) else Y[lm])
        return out

    def _ylm_at(self, d):
        '''|d| and the (Loewdin-orthonormalised) real harmonics up to lmax at directions d (n, 3): (n,), (nlm, n).'''
        if getattr(self, '_ytr', None) is None:
            self._ytr = _ylm_transforms(self.lmax, self.unit, self.wang)
        dist = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        ys = sph.real_sph_vec(d / dist[:, None], self.lmax, reorder_p=True)
        return dist, np.vstack([t @ np.asarray(y, dtype=float) for t, y in zip(self._ytr, ys)])

    def _cache_geo(self, key, geo):
        '''Keep the geometry of a neighbour-potential evaluation (fixed over the SCF) within cache_gb.'''
        if getattr(self, '_geo', None) is None:
            self._geo, self._geo_bytes = {}, 0
        nb = geo[0].nbytes + geo[1].nbytes + geo[2].nbytes
        cap = getattr(self, 'cache_gb', None)
        cap = cap * 1e9 if cap is not None else min(16e9, 0.1 * self.th.avail_bytes() + self._geo_bytes)
        if self._geo_bytes + nb <= cap:
            self._geo[key] = geo
            self._geo_bytes += nb

    def radial_grid(self, Z):
        '''(points, pyscf.dft.radi scheme) of the radial grid of an atom of charge Z: self.nrad (and self.radial)
        if given, else Treutler-Ahlrichs 600 for Z <= 18 (quartz: E_J 2e-6 vs RSDF) and Mura-Knowles 2000 above
        (EuOCl: E_J converged to ~3e-7; TA 600 is off by 1e-4 for the Eu core).'''
        if self.nrad is not None:
            return self.nrad, getattr(self, 'radial', 'treutler_ahlrichs')
        return (600, 'treutler_ahlrichs') if Z <= 18 else (2000, 'mura_knowles')

    def _ss_first(self):
        '''Permute the stored functions of every atomic grid so that the smooth ones come first (once).'''
        if getattr(self, '_nss', None) is not None:
            return
        sm = self.th.smooth
        self._nss, self._compact_pts = {}, {}
        for A in self.my_atoms:
            pts, w, inner, cols, cA, phi_in, phi0c = self.grids[A]
            ss = np.isin(cols, sm)
            perm = np.concatenate([np.where(ss)[0], np.where(~ss)[0]])
            phi_in = phi_in[:, perm]
            nss = int(ss.sum())
            # stored: the smooth columns only (C order; the SS term with A's local potential), the compact rows
            # below
            self.grids[A] = (pts, w, inner, cols[perm], cA, np.ascontiguousarray(phi_in[:, :nss]), phi0c)
            self._nss[A] = nss
            # the compact home functions (exponents >= alpha_c) vanish within a few bohr: their rows of the grid
            big = np.abs(phi0c).max(axis=1) if phi0c.shape[1] else np.zeros(len(phi0c))
            ci = np.where(big > getattr(self, 'c_thr', 1e-14) * max(big.max(), 1e-300))[0]
            self._compact_pts[A] = (ci, np.ascontiguousarray(phi0c[ci]), np.ascontiguousarray(phi_in[ci]))

    def get_j(self, dm):
        '''J (vector convention) in the split basis for a real symmetric dm.'''
        th, L, cell = self.th, self.L, self.th.cell
        nao, sm = th.nao, th.smooth
        vol, ng, mesh = cell.vol, th.ng, th.mesh
        hshape = (mesh[0], mesh[1], mesh[2] // 2 + 1)
        dm = .5 * (dm + dm.T)
        self._ss_first()
        if th.phi is not None and not th.phi.flags['C_CONTIGUOUS']:     # row blocks of the grid functions
            th.phi = np.ascontiguousarray(th.phi)
        comm = self.comm
        if comm is not None:
            from mpi4py import MPI
            rank, size = comm.Get_rank(), comm.Get_size()
            assert not getattr(self, 'neighbours', False), 'GAPW neighbour potentials not distributed'
        else:
            rank, size = 0, 1
        r0, r1 = ng * rank // size, ng * (rank + 1) // size            # this rank's rows of the uniform grid
        # smooth density on the uniform grid
        n = np.zeros(ng)
        Dss = np.ascontiguousarray(dm[np.ix_(sm, sm)])
        if th.phi is None:                                            # phi on demand: evaluated per grid block
            blk = 56 * 4096                                           # whole eval_gto blocks
            for g0 in range(r0, r1, blk):
                g1 = min(g0 + blk, r1)
                n[g0:g1] = _wdens(th.phi_at(slice(g0, g1)), Dss)
            phi_loc = None
        else:
            phi_loc = th.phi if getattr(th, '_phi_rows', None) == (r0, r1) else th.phi[r0:r1]   # distributed: own rows
            n[r0:r1] = _wdens(phi_loc, Dss)
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, n, op=MPI.SUM)
        nG = scipy.fft.rfftn(n.reshape(mesh), workers=-1).reshape(-1) * (vol / ng)
        nGc = np.zeros_like(nG)                                       # compensation charges of this rank's atoms
        # compact densities on the atomic grids, multipoles, compensation charges
        rhoA, QA = {}, {}
        for A in self.my_atoms:
            pts, w, inner, cols, cA, phi_in, phi0c = self.grids[A]
            msk = (self.owner[np.ix_(cA, cols)] == A) & self.home[np.ix_(cA, cols)]
            DA = np.where(msk, dm[np.ix_(cA, cols)], 0.)
            DA[cA[:, None] == cols[None, :]] *= .5                      # diagonal pairs counted once
            rho = np.zeros(len(pts))
            ci, p0c, pic = self._compact_pts[A]                          # points where phi^(0) is non-negligible
            rho[inner[ci]] = 2 * _rowdot(p0c @ DA, pic)
            rhoA[A] = rho
            rad = np.repeat(self.rads[A], len(self.wang))
            QA[A] = [((rho * w * rad ** l)[None] * np.tile(self.Y[l], len(self.rads[A]))).sum(1) for l in range(L + 1)]
            ph = self._phase(A)
            for l in range(L + 1):
                nGc += ph * (QA[A][l] @ self.gG[l])
        if comm is not None:
            comm.Allreduce(MPI.IN_PLACE, nGc.view(float), op=MPI.SUM)
        nG = nG + nGc
        vG = th.w_vec * nG
        vr = scipy.fft.irfftn(vG.reshape(hshape), s=mesh, workers=-1).reshape(-1)   # v_grid(r) vol / ng
        J = np.zeros((nao, nao))
        if phi_loc is None:
            Jss = np.zeros((len(sm), len(sm)))
            for g0 in range(r0, r1, blk):
                g1 = min(g0 + blk, r1)
                Jss += _wmat(th.phi_at(slice(g0, g1)), vr[g0:g1])
            J[np.ix_(sm, sm)] = Jss
        else:
            J[np.ix_(sm, sm)] = _wmat(phi_loc, vr[r0:r1])
        vloc, vlms = {}, {}
        c_avg = 0.
        for A in self.my_atoms:
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
        for A in self.my_atoms:
            pts, w, inner, cols, cA, phi_in, phi0c = self.grids[A]
            rad = np.repeat(self.rads[A], len(self.wang))
            u = self._vgrid_at(vG, pts, rad, r_max=self.r_compact)    # smooth potential where phi^(0) lives
            # neighbours' local potentials where the compact density of A lives
            vn = np.zeros_like(rad)
            near = np.where(rad < self.r_compact)[0]
            for B in self.atoms:
                if getattr(self, 'neighbours', False):    # quartz: E_J changes 5e-7 without them
                    vn[near] += self._vloc_at(B, vlms[B], pts[near], home=(B != A), center=X[A],
                                              radius=self.r_compact)   # A's own: images only
            # pairs owned by A: smooth + own local + neighbours' local potentials
            ci, p0c, pic = self._compact_pts[A]
            wv = (w * (u + vloc[A] + vn))[inner[ci]]
            JA = _wmat(p0c, wv, pic)                                    # (home a in cA, b in cols)
            own = (self.owner[np.ix_(cA, cols)] == A) & self.home[np.ix_(cA, cols)]
            ia, ib = np.nonzero(own)
            J[cA[ia], cols[ib]] += JA[ia, ib]
            od = cA[ia] != cols[ib]
            J[cols[ib[od]], cA[ia[od]]] += JA[ia[od], ib[od]]
            # SS pairs: the local potential of A (all space, on A's grid; it lives within r_store); the smooth
            # columns come first in cols (see _ss_first), so their block is a view
            nss = self._nss[A]
            ps = phi_in                                                  # (stored: the smooth columns only)
            J[np.ix_(cols[:nss], cols[:nss])] += _wmat(ps, (w * vloc[A])[inner])
        J += c_avg * self.S                                           # constant potential: c S_ab
        if comm is not None:                                          # partial J of every rank (grid rows, atoms)
            comm.Allreduce(MPI.IN_PLACE, J, op=MPI.SUM)
        return J
