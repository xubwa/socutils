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
Angular algebra for the 1s1/2^2 (J=0) pair, brute force in uncoupled
(m_r, m_s) form so that no recoupling formula has to be trusted.

Conventions:  1/r12 = sum_k r_<^k / r_>^(k+1)  C^k(1) . C^k(2),
C^k . C^k = sum_q (-1)^q C^k_q C^k_-q,  Wigner-Eckart
<j m|C^k_q|j' m'> = (-1)^(j-m) (j k j'; -m q m') <j||C^k||j'>,
<ka||C^k||kb> = (-1)^(ja+1/2) sqrt([ja][jb]) (ja jb k; -1/2 1/2 0) Pi(la+lb+k)
(W. R. Johnson, Atomic Structure Theory, Springer 2007).
Spinor: phi = (1/r) (P Omega_{kappa m}, i Q Omega_{-kappa m}).
All half-integers are passed doubled (j2 = 2j, m2 = 2m).
'''

import functools
import mpmath


def kappa_lj(kappa):
    '''(l, 2j) for kappa'''
    if kappa > 0:
        return kappa, 2 * kappa - 1
    return -kappa - 1, -2 * kappa - 1


def _f(ctx, n2):
    assert n2 % 2 == 0
    return ctx.factorial(n2 // 2)


def wigner3j(j1, j2, j3, m1, m2, m3, ctx=mpmath.mp):
    '''Racah formula, doubled arguments'''
    if m1 + m2 + m3 != 0:
        return ctx.mpf(0)
    if any((a + b) % 2 for a, b in ((j1, m1), (j2, m2), (j3, m3))):
        return ctx.mpf(0)
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return ctx.mpf(0)
    if j3 > j1 + j2 or j3 < abs(j1 - j2) or (j1 + j2 + j3) % 2:
        return ctx.mpf(0)
    f = lambda n2: _f(ctx, n2)
    tri = f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3) / f(j1 + j2 + j3 + 2)
    pre = ctx.sqrt(tri * f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2)
                   * f(j3 + m3) * f(j3 - m3))
    s = ctx.mpf(0)
    for t2 in range(0, 2 * (j1 + j2 + j3) + 2, 2):
        a = [t2, j3 - j2 + t2 + m1, j3 - j1 + t2 - m2, j1 + j2 - j3 - t2,
             j1 - t2 - m1, j2 - t2 + m2]
        if min(a) < 0:
            continue
        den = 1
        for x in a:
            den *= f(x)
        s += (-1) ** (t2 // 2) / den
    phase = (-1) ** ((j1 - j2 - m3) // 2)
    return phase * pre * s


def clebsch(j1, m1, j2, m2, j, m, ctx=mpmath.mp):
    '''<j1 m1 j2 m2|j m>, doubled arguments'''
    return (-1) ** ((j1 - j2 + m) // 2) * ctx.sqrt(j + 1) * wigner3j(j1, j2, j, m1, m2, -m, ctx)


def reduced_ck(ka, kb, k, ctx=mpmath.mp):
    la, ja = kappa_lj(ka)
    lb, jb = kappa_lj(kb)
    if (la + lb + k) % 2:
        return ctx.mpf(0)
    return ((-1) ** ((ja + 1) // 2) * ctx.sqrt((ja + 1) * (jb + 1))
            * wigner3j(ja, jb, 2 * k, -1, 1, 0, ctx))


def ck_me(ka, ma, kb, mb, k, q, ctx=mpmath.mp):
    '''<ka ma|C^k_q|kb mb>, ma, mb doubled'''
    _, ja = kappa_lj(ka)
    _, jb = kappa_lj(kb)
    return ((-1) ** ((ja - ma) // 2) * wigner3j(ja, 2 * k, jb, -ma, 2 * q, mb, ctx)
            * reduced_ck(ka, kb, k, ctx))


@functools.lru_cache(maxsize=None)
def pair_amplitudes(kappa, ctx=mpmath.mp):
    '''Angular part of <r s||a b> for r=(kappa,m_r), s=(kappa,m_s),
    a=(1s,+1/2), b=(1s,-1/2).  The radial part is the same Slater integral
    R^l for direct and exchange (a, b share the radial function), so
    <rs||ab> = R^l(r,s) * T[m_r][m_s].  Returns k (=l) and T as a dict.'''
    l, j2 = kappa_lj(kappa)
    k = l
    T = {}
    for mr in range(-j2, j2 + 1, 2):
        for ms in range(-j2, j2 + 1, 2):
            d = ctx.mpf(0)
            e = ctx.mpf(0)
            # <m_r|C_q|m_a> needs q = m_r - m_a, <m_s|C_-q|m_b> needs -q = m_s - m_b
            if mr + ms == 0:
                q = (mr - 1) // 2
                if abs(q) <= k:
                    d = (-1) ** q * ck_me(kappa, mr, -1, 1, k, q, ctx) * ck_me(kappa, ms, -1, -1, k, -q, ctx)
                q = (mr + 1) // 2
                if abs(q) <= k:
                    e = (-1) ** q * ck_me(kappa, mr, -1, -1, k, q, ctx) * ck_me(kappa, ms, -1, 1, k, -q, ctx)
            T[(mr, ms)] = d - e
    return k, T


def pair_weight(kappa, ctx=mpmath.mp):
    '''W = sum_{m_r m_s} T^2 ; E2(kappa) = W/2 sum_{nn'} R^2 / D'''
    _, T = pair_amplitudes(kappa, ctx)
    return sum(v * v for v in T.values())


def omega_z(kappa, m2, ctx=mpmath.mp):
    '''Omega_{kappa m}(z-hat) as a 2-vector (spin up, spin down)'''
    l, j2 = kappa_lj(kappa)
    out = [ctx.mpf(0), ctx.mpf(0)]
    for isg, s2 in enumerate((1, -1)):
        ml2 = m2 - s2
        if ml2 != 0:
            continue          # Y_l^{ml}(z) = 0 unless ml = 0
        out[isg] = ctx.sqrt(ctx.mpf(2 * l + 1) / (4 * ctx.pi)) * clebsch(2 * l, 0, 1, s2, j2, m2, ctx)
    return out


@functools.lru_cache(maxsize=None)
def coalescence_tensor(kappa, comp, ctx=mpmath.mp):
    '''Spin tensor G[s1][s2] of the kappa partial wave of the first-order pair
    function at r1 = r2 = r z-hat, per unit radial amplitude
    U^{XY}(r) = sum_{nn'} t_{nn'} X_n(r) Y_n'(r):

        u^{XY}(r z, r z) = 1/(sqrt2 r^2) sum_kappa U^{XY}_kappa(r) G^{XY}_kappa

    comp in 'LL', 'LS', 'SL', 'SS'.  Small components carry the factor i.'''
    _, T = pair_amplitudes(kappa, ctx)
    om = {'L': lambda m: omega_z(kappa, m, ctx), 'S': lambda m: omega_z(-kappa, m, ctx)}
    ph = {'L': 1, 'S': 1j}
    G = [[0, 0], [0, 0]]
    for (mr, ms), t in T.items():
        o1 = om[comp[0]](mr)
        o2 = om[comp[1]](ms)
        for a in range(2):
            for b in range(2):
                G[a][b] += t * o1[a] * o2[b]
    f = ph[comp[0]] * ph[comp[1]]
    return [[complex(f) * ctx.mpc(G[a][b]) for b in range(2)] for a in range(2)]


def reference_tensor(comp, ctx=mpmath.mp):
    '''Same for Phi0 = |1s+ 1s-|:  Phi0^{XY}(r z, r z) = X_a Y_a /(sqrt2 r^2) G0'''
    om = {'L': lambda m: omega_z(-1, m, ctx), 'S': lambda m: omega_z(1, m, ctx)}
    ph = {'L': 1, 'S': 1j}
    a1, b1 = om[comp[0]](1), om[comp[0]](-1)
    a2, b2 = om[comp[1]](1), om[comp[1]](-1)
    f = ph[comp[0]] * ph[comp[1]]
    return [[complex(f) * ctx.mpc(a1[s] * b2[t] - b1[s] * a2[t]) for t in range(2)] for s in range(2)]


def singlet(G):
    return (G[0][1] - G[1][0]) / mpmath.sqrt(2)


def triplet0(G):
    return (G[0][1] + G[1][0]) / mpmath.sqrt(2)


def legendre_r12(l, ctx=mpmath.mp):
    '''Legendre coefficient c_l of r12/r = sqrt(2(1-x)) at r1 = r2 = r:
    sqrt(2(1-x)) = sum_l c_l P_l(x).  Uses
    int_{-1}^1 (1-x)^a P_l dx = (-1)^l 2^(a+1) Gamma(a+1)^2 / (Gamma(a+l+2) Gamma(a+1-l)).'''
    a = ctx.mpf(1) / 2
    integral = (-1) ** l * 2 ** (a + 1) * ctx.gamma(a + 1) ** 2 * ctx.rgamma(a + 1 - l) / ctx.gamma(a + l + 2)
    return (2 * l + 1) * ctx.sqrt(2) * integral / 2
