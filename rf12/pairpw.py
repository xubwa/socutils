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
Partial-wave second-order pair energy of the 1s1/2^2 state of a He-like ion,
Dirac-Coulomb no-pair, brute-force sum over states (reference implementation).

Zeroth order: hydrogenic Dirac (or Schroedinger) Hamiltonian, perturbation
1/r12 (1/Z expansion).  Scaled units r -> Z r, E -> E/Z^2: the nucleus has
charge 1 and the speed of light is c/Z, so the second-order energy E2 is a
function of Z/c only and E2(Z/c -> 0) is the non-relativistic
-0.15766642946915 Eh.

Radial problem: B-splines with dual kinetic balance (DKB, Shabaev et al.,
PRL 93, 130405 (2004)); for kappa the basis vectors are
    (B_i, D_kappa B_i / 2c)   and   (D_-kappa B_i / 2c, B_i),
D_kappa = d/dr + kappa/r, H shifted by -c^2.  Positive-energy states
(epsilon > -c^2) span the no-pair space.

Second order (a = 1s+, b = 1s-, r, s positive-energy spinors):
    E2 = 1/2 sum_{rs} |<rs||ab>|^2 / (eps_a + eps_b - eps_r - eps_s)
For J = 0 only kappa_r = kappa_s contributes and the multipole is k = l, so
    E2(kappa) = W_kappa / 2 * sum_{nn'} R^l(n n')^2 / D_{nn'}
with R^l(n n') = int int rho_n(1) r<^l/r>^(l+1) rho_n'(2),
rho_n = P_n P_a + Q_n Q_a, and W_kappa from socutils.rf12.angular.
Only the doubly occupied (1s,1s) pair is excluded; the singles 1s -> ns
are part of the hydrogenic E2.

Coulomb integral on the grid:
    R(n,n') = int dx rho_n'(x) Atilde_n(x) + (n <-> n'),
    Atilde_n(x) = x^-(l+1) int_0^x s^l rho_n(s) ds
with Atilde from the tables of Grid.coulomb_tables (only rho is
interpolated, never s^l, so nothing is amplified by x^-(l+1)).
'''

import numpy
from socutils.rf12 import angular
from socutils.rf12.bspline import Grid


class Space:
    '''Grid + backend arrays shared by all kappa (scaled units, Z = 1).'''
    def __init__(self, grid, xp, c=None):
        self.grid = grid
        self.xp = xp
        self.c = None if c is None else grid.ctx.mpf(c)
        ctx = grid.ctx
        self.nb = grid.nb
        self.nint = grid.nint
        self.nq = grid.nq
        r = grid.r
        self.r = xp.from_mp(r)[None, :]
        self.invr = xp.from_mp([1 / x for x in r])[None, :]
        self.invr2 = xp.from_mp([1 / x ** 2 for x in r])[None, :]
        self.w = xp.from_mp(grid.w)[None, :]
        self.B = [xp.from_mp(grid.B[d]) for d in range(3)]
        self.ones_q = xp.from_float(numpy.ones((grid.nq, 1)))
        tri = numpy.zeros((grid.nint, grid.nint))
        tri[numpy.triu_indices(grid.nint, 1)] = 1    # tri[m', m] = 1 for m' < m
        self.tri = xp.from_float(tri)
        self._cache = {}

    def tables(self, k):
        '''Coulomb tables of multipole k in backend format'''
        if k not in self._cache:
            g = self.grid
            if hasattr(g, 'table_cache') and k in g.table_cache:
                F, K, xmk = g.table_cache[k]
            else:
                F, K, xmk = g.coulomb_tables(k)
                if not hasattr(g, 'table_cache'):
                    g.table_cache = {}
                g.table_cache[k] = (F, K, xmk)
            xp = self.xp
            KT = [xp.from_mp([[Km[i][j] for i in range(g.nq)] for j in range(g.nq)]) for Km in K]
            self._cache[k] = (xp.from_mp(F)[None, :], KT,
                              xp.from_mp(xmk).reshape(1, g.nint, g.nq))
        return self._cache[k]


def _components(sp, B, invr, invr2, kappa, c):
    '''Rows of P, Q and D_kappa P of the DKB basis for kappa.'''
    xp = sp.xp
    nb = sp.nb
    iL = numpy.arange(1 if kappa == -1 else 2, nb - 1)
    iS = numpy.arange(1 if kappa == 1 else 2, nb - 1)
    B0, B1, B2 = B
    half_c = 1 / (2 * c)
    PL = B0[iL]
    DL = xp.add(B1[iL], xp.scale(xp.mul(B0[iL], invr), kappa))
    QL = xp.scale(DL, half_c)
    PS = xp.scale(xp.add(B1[iS], xp.scale(xp.mul(B0[iS], invr), -kappa)), half_c)
    QS = B0[iS]
    DS = xp.scale(xp.add(B2[iS], xp.scale(xp.mul(B0[iS], invr2), -kappa * (kappa - 1))), half_c)
    cat = numpy.concatenate
    return cat([PL, PS]), cat([QL, QS]), cat([DL, DS])


def _components_nr(sp, B, l):
    i = numpy.arange(1, sp.nb - 1)
    return B[0][i], B[1][i]


class KappaStates:
    '''Radial eigenstates of one symmetry (kappa, or l if c is None).'''
    def __init__(self, sp, kappa):
        xp = sp.xp
        self.sp = sp
        self.kappa = kappa
        c = sp.c
        w = sp.w
        if c is None:
            l = angular.kappa_lj(kappa)[0]
            P, dP = _components_nr(sp, sp.B, l)
            Pw = xp.mul(P, w)
            S = xp.matmul(Pw, P, trans_b=True)
            T = xp.scale(xp.matmul(xp.mul(dP, w), dP, trans_b=True), 0.5)
            cent = xp.scale(xp.mul(Pw, sp.invr2), l * (l + 1) / 2.)
            V = xp.scale(xp.mul(Pw, sp.invr), -1)
            H = xp.add(xp.add(T, xp.matmul(cent, P, trans_b=True)), xp.matmul(V, P, trans_b=True))
            e, C = xp.sygv(H, S)
            self.P = xp.matmul(C, P, trans_a=True)
            self.Q = None
            self.C = C
            self.e = e
        else:
            P, Q, D = _components(sp, sp.B, sp.invr, sp.invr2, kappa, c)
            Pw = xp.mul(P, w)
            Qw = xp.mul(Q, w)
            S = xp.add(xp.matmul(Pw, P, trans_b=True), xp.matmul(Qw, Q, trans_b=True))
            VP = xp.scale(xp.mul(Pw, sp.invr), -1)
            VQ = xp.axpby(-1, xp.mul(Qw, sp.invr), -2 * c * c, Qw)
            H = xp.add(xp.matmul(VP, P, trans_b=True), xp.matmul(VQ, Q, trans_b=True))
            DQ = xp.matmul(xp.mul(D, w), Q, trans_b=True)
            H = xp.add(H, xp.axpby(c, DQ, c, DQ.T.copy()))
            e, C = xp.sygv(H, S)
            ef = xp.to_float(e)
            pos = numpy.where(ef > -float(c) ** 2)[0]
            self.nneg = len(ef) - len(pos)
            C = numpy.ascontiguousarray(C[:, pos])
            e = numpy.ascontiguousarray(e[pos])
            self.P = xp.matmul(C, P, trans_a=True)
            self.Q = xp.matmul(C, Q, trans_a=True)
            self.C = C
            self.e = e
        self.ef = xp.to_float(self.e)

    def at(self, x):
        '''P_n(x), Q_n(x) at arbitrary radii'''
        sp = self.sp
        xp = sp.xp
        ctx = sp.grid.ctx
        Bx = sp.grid.eval_at(x)
        B = [xp.from_mp(Bx[d]) for d in range(3)]
        invr = xp.from_mp([1 / ctx.mpf(v) for v in x])[None, :]
        invr2 = xp.from_mp([1 / ctx.mpf(v) ** 2 for v in x])[None, :]
        if sp.c is None:
            P, _ = _components_nr(sp, B, None)
            return xp.matmul(self.C, P, trans_a=True), None
        P, Q, _ = _components(sp, B, invr, invr2, self.kappa, sp.c)
        return xp.matmul(self.C, P, trans_a=True), xp.matmul(self.C, Q, trans_a=True)


def _atilde(sp, rho, k):
    '''Atilde_n(x_p) = x_p^-(k+1) int_0^x_p s^k rho_n(s) ds on the grid'''
    xp = sp.xp
    n = rho.shape[0]
    nint, nq = sp.nint, sp.nq
    F, KT, xmk = sp.tables(k)
    full = xp.matmul(xp.mul(rho, F).reshape(n * nint, nq), sp.ones_q).reshape(n, nint)
    cum = xp.matmul(full, sp.tri)
    A = xp.mul(cum[:, :, None], xmk)
    rho3 = rho.reshape(n, nint, nq)
    part = numpy.concatenate([xp.matmul(numpy.ascontiguousarray(rho3[:, m]), KT[m])[:, None, :]
                              for m in range(nint)], axis=1)
    return xp.add(A, part).reshape(n, nint * nq)


def slater_channels(sp, st, ref, k):
    '''R^k split by component of the transition densities:
    R^{XY}(n,n') = int int rho^X_n(1) K rho^Y_n'(2), rho^L = P_n P_a,
    rho^S = Q_n Q_a.  Returns dict LL, SS, LS (= LS + SL).'''
    xp = sp.xp
    rho = {'L': xp.mul(st.P, ref.P[0:1]), 'S': xp.mul(st.Q, ref.Q[0:1])}
    At = {c: xp.mul(_atilde(sp, rho[c], k), sp.w) for c in 'LS'}
    G = {(a, b): xp.matmul(At[a], rho[b], trans_b=True) for a in 'LS' for b in 'LS'}
    out = {}
    for a, b in (('L', 'L'), ('S', 'S')):
        out[a + b] = xp.add(G[a, b], G[b, a].T.copy())
    ls = xp.add(G['L', 'S'], G['S', 'L'].T.copy())
    out['LS'] = xp.add(ls, ls.T.copy())
    return out


def slater_matrix(sp, st, ref, k):
    '''R^k(n n'; a a) for all states n, n' of st, a = ref (index 0 of its
    KappaStates).'''
    xp = sp.xp
    rho = xp.mul(st.P, ref.P[0:1])
    if st.Q is not None:
        rho = xp.add(rho, xp.mul(st.Q, ref.Q[0:1]))
    n = rho.shape[0]
    nint, nq = sp.nint, sp.nq
    F, KT, xmk = sp.tables(k)
    full = xp.matmul(xp.mul(rho, F).reshape(n * nint, nq), sp.ones_q).reshape(n, nint)
    cum = xp.matmul(full, sp.tri)
    A = xp.mul(cum[:, :, None], xmk)
    rho3 = rho.reshape(n, nint, nq)
    part = numpy.concatenate([xp.matmul(numpy.ascontiguousarray(rho3[:, m]), KT[m])[:, None, :]
                              for m in range(nint)], axis=1)
    A = xp.add(A, part).reshape(n, nint * nq)
    G = xp.matmul(xp.mul(A, sp.w), rho, trans_b=True)
    return xp.add(G, G.T.copy())


class PairResult:
    pass


def pair_kappa(sp, ref, kappa, rcoal=(), states=None, channels=False):
    '''Second-order energy and coalescence amplitudes of partial wave kappa.'''
    xp = sp.xp
    ctx = sp.grid.ctx
    st = states if states is not None else (ref if kappa == -1 else KappaStates(sp, kappa))
    l = angular.kappa_lj(kappa)[0]
    R = slater_matrix(sp, st, ref, l)
    ea = ref.e[0:1]
    D = xp.axpby(2, ea[:, None], -1, xp.add(st.e[:, None], st.e[None, :]))
    if kappa == -1:
        D[0, 0] = xp.scalar(1)
    t = xp.mul(R, xp.recip(D))
    if kappa == -1:
        t[0, 0] = xp.scalar(0)
    if sp.c is None:
        weight = ctx.mpf(1) / (2 * l + 1)
    else:
        weight = angular.pair_weight(kappa, ctx) / 2
    e2 = xp.scale(xp.sum(xp.mul(t, R)), weight)
    res = PairResult()
    res.kappa = kappa
    res.l = l
    res.e2 = xp.to_mp(e2, ctx)[0]
    res.nstates = st.e.shape[0]
    res.e2ch = {}
    if channels and st.Q is not None:
        for name, Rc in slater_channels(sp, st, ref, l).items():
            res.e2ch[name] = xp.to_mp(xp.scale(xp.sum(xp.mul(t, Rc)), weight), ctx)[0]
    res.coal = {}
    if len(rcoal):
        Px, Qx = st.at(list(rcoal))
        comps = {'L': Px, 'S': Qx}
        for cmp in (('LL', 'SS', 'LS') if Qx is not None else ('LL',)):
            X = comps[cmp[0]]
            Y = comps[cmp[1]]
            TY = xp.matmul(t, Y)
            U = xp.sum_axis0(xp.mul(X, TY)) if hasattr(xp, 'sum_axis0') else \
                xp.matmul(xp.from_float(numpy.ones((1, X.shape[0]))), xp.mul(X, TY))[0]
            res.coal[cmp] = xp.to_mp(U, ctx)
    return res, st
