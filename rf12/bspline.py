#!/usr/bin/env python
# Copyright 2026 The socutils developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
B-spline radial grid in arbitrary precision.

Everything here is set up in an mpmath context (``ctx``) with guard bits and
handed to the double or quad backend afterwards.  The grid consists of

* a knot sequence t (multiplicity k at r=0 and r=rmax),
* nq Gauss-Legendre points on every non-empty knot interval,
* for each multipole k, the tables that turn values of a density rho on the
  Gauss points into the Coulomb kernel integrals (see coulomb_tables).
  Only the Lagrange interpolant of rho itself is used, which is exact when
  rho is a polynomial on the interval (always true on [0, rmin], where the
  B-splines vanish at r = 0, and true everywhere for the Schroedinger
  case).  Powers s^k are never interpolated: they are integrated exactly
  against the Lagrange basis.
'''

import mpmath


def gauss_legendre(n, ctx):
    '''Nodes and weights on [-1, 1]'''
    x = []
    w = []
    for i in range(1, n + 1):
        z = ctx.cos(ctx.pi * (i - ctx.mpf(1) / 4) / (n + ctx.mpf(1) / 2))
        for _ in range(100):
            p0, p1 = ctx.mpf(1), z
            for j in range(2, n + 1):
                p0, p1 = p1, ((2 * j - 1) * z * p1 - (j - 1) * p0) / j
            dp = n * (z * p1 - p0) / (z * z - 1)
            dz = p1 / dp
            z -= dz
            if abs(dz) < ctx.mpf(2) ** (-ctx.prec + 4):
                break
        p0, p1 = ctx.mpf(1), z
        for j in range(2, n + 1):
            p0, p1 = p1, ((2 * j - 1) * z * p1 - (j - 1) * p0) / j
        dp = n * (z * p1 - p0) / (z * z - 1)
        x.append(z)
        w.append(2 / ((1 - z * z) * dp * dp))
    order = sorted(range(n), key=lambda i: x[i])
    return [x[i] for i in order], [w[i] for i in order]




def make_knots(nb, k, rmin, rmax, ctx, rmid=None, frac=0.5, uniform=None, segments=None):
    '''Knot vector for nb B-splines of order k on [0, rmax].

    Interior knots: geometric from rmin up to rmid (a fraction ``frac`` of
    them), then geometric with a common ratio continuing to rmax.  If rmid is
    None a single geometric sequence rmin..rmax is used.

    uniform = (ra, rb, frac_a, frac_u): geometric rmin..ra (fraction frac_a
    of the knots), uniform ra..rb (fraction frac_u), geometric rb..rmax.  This
    resolves the 1s region, where the high-l pair functions live (their
    width in r1 - r2 is ~ r/l).

    segments = [(r_1, f_1), (r_2, f_2), ...]: piecewise geometric
    rmin..r_1..r_2..rmax, a fraction f_i of the intervals ending at r_i
    (the last segment ends at rmax and takes the rest).  A geometric grid
    has h/r constant, matching the r/l width at every r.'''
    nint = nb - k + 1          # number of intervals
    rmin = ctx.mpf(rmin)
    rmax = ctx.mpf(rmax)
    if segments is not None:
        ends = [ctx.mpf(r) for r, _ in segments] + [rmax]
        counts = [int(round(f * (nint - 1))) for _, f in segments]
        counts.append(nint - 1 - sum(counts))
        pts = [rmin]
        start = rmin
        for e, n in zip(ends, counts):
            pts += [start * (e / start) ** (ctx.mpf(i) / n) for i in range(1, n + 1)]
            start = e
    elif uniform is not None:
        ra, rb, fa, fu = uniform
        ra = ctx.mpf(ra)
        rb = ctx.mpf(rb)
        n1 = int(round(fa * (nint - 1)))
        n2 = int(round(fu * (nint - 1)))
        n3 = nint - 1 - n1 - n2
        pts = [rmin * (ra / rmin) ** (ctx.mpf(i) / n1) for i in range(n1 + 1)]
        pts += [ra + (rb - ra) * ctx.mpf(i) / n2 for i in range(1, n2 + 1)]
        pts += [rb * (rmax / rb) ** (ctx.mpf(i) / n3) for i in range(1, n3 + 1)]
    elif rmid is None:
        pts = [rmin * (rmax / rmin) ** (ctx.mpf(i) / (nint - 1)) for i in range(nint)]
    else:
        rmid = ctx.mpf(rmid)
        n1 = int(round(frac * (nint - 1)))
        pts = [rmin * (rmid / rmin) ** (ctx.mpf(i) / n1) for i in range(n1 + 1)]
        n2 = nint - 1 - n1
        pts += [rmid * (rmax / rmid) ** (ctx.mpf(i) / n2) for i in range(1, n2 + 1)]
    t = [ctx.mpf(0)] * k + pts[:-1] + [rmax] * k
    # interior knots: pts[0..nint-2]; first interval [0, rmin]
    assert len(t) == nb + k
    return t


def bspline_ders(t, k, x, span, ctx, nder=2):
    '''Values and derivatives of the k B-splines non-zero at x (knot span
    ``span`` with t[span] <= x < t[span+1]).  Piegl & Tiller, A2.3.
    Returns ders[d][j] for basis index span-k+1+j.'''
    p = k - 1
    ndu = [[ctx.mpf(0)] * (p + 1) for _ in range(p + 1)]
    left = [ctx.mpf(0)] * (p + 1)
    right = [ctx.mpf(0)] * (p + 1)
    ndu[0][0] = ctx.mpf(1)
    for j in range(1, p + 1):
        left[j] = x - t[span + 1 - j]
        right[j] = t[span + j] - x
        saved = ctx.mpf(0)
        for r in range(j):
            ndu[j][r] = right[r + 1] + left[j - r]
            temp = ndu[r][j - 1] / ndu[j][r]
            ndu[r][j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j][j] = saved
    ders = [[ctx.mpf(0)] * (p + 1) for _ in range(nder + 1)]
    for j in range(p + 1):
        ders[0][j] = ndu[j][p]
    a = [[ctx.mpf(0)] * (p + 1) for _ in range(2)]
    for r in range(p + 1):
        s1, s2 = 0, 1
        a[0][0] = ctx.mpf(1)
        for kk in range(1, nder + 1):
            d = ctx.mpf(0)
            rk = r - kk
            pk = p - kk
            if r >= kk:
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk]
                d = a[s2][0] * ndu[rk][pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = kk - 1 if r - 1 <= pk else p - r
            for j in range(j1, j2 + 1):
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]
                d += a[s2][j] * ndu[rk + j][pk]
            if r <= pk:
                a[s2][kk] = -a[s1][kk - 1] / ndu[pk + 1][r]
                d += a[s2][kk] * ndu[r][pk]
            ders[kk][r] = d
            s1, s2 = s2, s1
    fac = p
    for kk in range(1, nder + 1):
        for j in range(p + 1):
            ders[kk][j] *= fac
        fac *= (p - kk)
    return ders


class Grid:
    '''B-spline basis tabulated on Gauss points, in mpmath numbers.

    Attributes (python lists / nested lists of ctx.mpf):
        r[p], w[p]        quadrature points and weights (all intervals)
        h[m]              interval lengths, nint intervals, nq points each
        B[d][i][p]        d-th derivative of B-spline i at point p
        xref[j]           Gauss nodes on the reference interval [-1, 1]
    '''
    def __init__(self, nb=60, k=10, rmin=1e-5, rmax=60., nq=None, rmid=None,
                 frac=0.5, prec=200, uniform=None, segments=None):
        ctx = mpmath.MPContext()
        ctx.prec = prec
        self.ctx = ctx
        self.nb = nb
        self.k = k
        if nq is None:
            nq = k + 10
        self.nq = nq
        self.signature = repr((nb, k, rmin, rmax, nq, rmid, frac, prec, uniform, segments))
        t = make_knots(nb, k, rmin, rmax, ctx, rmid, frac, uniform, segments)
        self.t = t
        xg, wg = gauss_legendre(nq, ctx)
        self.xref = xg
        # barycentric weights of the reference nodes
        self.bary = [1 / ctx.fprod([xg[j] - xg[i] for i in range(nq) if i != j])
                     for j in range(nq)]
        spans = [m for m in range(k - 1, nb) if t[m + 1] > t[m]]
        self.nint = len(spans)
        r = []
        w = []
        h = []
        B = [[[ctx.mpf(0)] * (self.nint * nq) for _ in range(nb)] for _ in range(3)]
        p = 0
        for m in spans:
            a, b = t[m], t[m + 1]
            hm = b - a
            h.append(hm)
            for j in range(nq):
                x = a + hm * (xg[j] + 1) / 2
                r.append(x)
                w.append(hm * wg[j] / 2)
                ders = bspline_ders(t, k, x, m, ctx)
                for d in range(3):
                    for jj in range(k):
                        B[d][m - k + 1 + jj][p] = ders[d][jj]
                p += 1
        self.r = r
        self.w = w
        self.h = h
        self.B = B
        self.spans = spans
        self._gl = {}

    def _lagrange(self, xi):
        '''all Lagrange basis polynomials of the reference nodes at xi'''
        for j, x in enumerate(self.xref):
            if xi == x:
                return [self.ctx.mpf(int(i == j)) for i in range(self.nq)]
        t = [b / (xi - x) for b, x in zip(self.bary, self.xref)]
        s = self.ctx.fsum(t)
        return [v / s for v in t]

    def _gauss(self, n):
        if n not in self._gl:
            self._gl[n] = gauss_legendre(n, self.ctx)
        return self._gl[n]

    def coulomb_tables(self, k):
        '''Tables for the multipole-k Coulomb integral of densities on the
        grid,

            Atilde(x_p) = x_p^-(k+1) int_0^x_p s^k rho(s) ds
                        = x_p^-(k+1) sum_{m' < m(p)} sum_j F[q] rho(x_q)   (q in m')
                          + sum_j K[m][p][j] rho(x_{m,j})                (same interval)

        F[q]       = int_{interval} s^k l_q(s) ds
        K[m][i][j] = int_{t_m/x_i}^1 u^k l_j(x_i u) du,  x_i the i-th point of m
        xmk[p]     = x_p^-(k+1)
        Every integral is a Gauss-Legendre rule exact for degree k + nq - 1.'''
        import os
        import pickle
        import hashlib
        import multiprocessing
        cache = os.environ.get('RF12_CACHE')
        if cache:
            key = hashlib.sha1(self.signature.encode()).hexdigest()[:16]
            fn = os.path.join(cache, 'coul_%s_k%d.pkl' % (key, k))
            if os.path.exists(fn):
                with open(fn, 'rb') as f:
                    Fs, Ks = pickle.load(f)
                F = [self.ctx.mpf(x) for x in Fs]
                K = [[[self.ctx.mpf(x) for x in row] for row in Km] for Km in Ks]
                return F, K, [x ** (-k - 1) for x in self.r]
        nproc = int(os.environ.get('RF12_NPROC', os.cpu_count() or 1))
        chunks = [list(range(i, self.nint, nproc)) for i in range(nproc)]
        if nproc > 1:
            ctxmp = multiprocessing.get_context('fork')
            global _GRID
            _GRID = self
            with ctxmp.Pool(nproc) as pool:
                res = pool.starmap(_tables_worker, [(k, c) for c in chunks])
        else:
            res = [self._tables_part(k, c) for c in chunks]
        F = [None] * self.nint
        K = [None] * self.nint
        for c, (Fc, Kc) in zip(chunks, res):
            for im, f, kk in zip(c, Fc, Kc):
                F[im] = [self.ctx.mpf(x) for x in f]
                K[im] = [[self.ctx.mpf(x) for x in row] for row in kk]
        F = [x for f in F for x in f]
        if cache:
            os.makedirs(cache, exist_ok=True)
            tostr = lambda x: mpmath.libmp.to_str(x._mpf_, 50)
            with open(fn, 'wb') as f:
                pickle.dump(([tostr(x) for x in F],
                             [[[tostr(x) for x in row] for row in Km] for Km in K]), f)
        xmk = [x ** (-k - 1) for x in self.r]
        return F, K, xmk

    def _tables_part(self, k, intervals):
        ctx = self.ctx
        nq = self.nq
        ng = (k + nq) // 2 + 2
        eta, weta = self._gauss(ng)
        F = []
        K = []
        for im in intervals:
            m = self.spans[im]
            a = self.t[m]
            hm = self.h[im]
            acc = [ctx.mpf(0)] * nq
            for e, we in zip(eta, weta):
                s = a + hm * (e + 1) / 2
                f = we * hm / 2 * s ** k
                lj = self._lagrange(e)
                for j in range(nq):
                    acc[j] += f * lj[j]
            F.append(acc)
            Km = []
            for i in range(nq):
                xi = self.r[im * nq + i]
                lo = a / xi
                row = [ctx.mpf(0)] * nq
                for e, we in zip(eta, weta):
                    u = lo + (1 - lo) * (e + 1) / 2
                    f = we * (1 - lo) / 2 * u ** k
                    lj = self._lagrange(2 * (xi * u - a) / hm - 1)
                    for j in range(nq):
                        row[j] += f * lj[j]
                Km.append(row)
            K.append(Km)
        # plain strings travel between processes
        tostr = lambda x: mpmath.libmp.to_str(x._mpf_, 50)
        return [[tostr(x) for x in f] for f in F], [[[tostr(x) for x in row] for row in Km] for Km in K]


    def eval_at(self, x):
        '''B-spline values/derivatives at arbitrary points x (list)'''
        ctx = self.ctx
        t, k, nb = self.t, self.k, self.nb
        out = [[[ctx.mpf(0)] * len(x) for _ in range(nb)] for _ in range(3)]
        for p, xp in enumerate(x):
            xp = ctx.mpf(xp)
            m = max(i for i in range(k - 1, nb) if t[i] <= xp and t[i + 1] > t[i])
            ders = bspline_ders(t, k, xp, m, ctx)
            for d in range(3):
                for jj in range(k):
                    out[d][m - k + 1 + jj][p] = ders[d][jj]
        return out


_GRID = None


def _tables_worker(k, intervals):
    return _GRID._tables_part(k, intervals)
