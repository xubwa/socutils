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
Short-distance model of the no-pair first-order pair function (inference,
to be tested against the brute-force partial-wave data).

Near coalescence, the large-large spin-singlet part of the first-order pair
function is taken as

    u_LL(r1, r2) ~ Phi0_LL(r, r) h(r12),
    h~(q) = -4 pi / q^2 * Lambda_LL(q)^2 / E(q),

i.e. the Coulomb kernel 4 pi / q^2, divided by the kinetic energy of
relative motion of two free electrons with momenta +-q, and multiplied by
the large-large block of the free positive-energy projector of each
electron (scaled units: nucleus charge 1, speed of light c'):

    E(q)         = 2 c'^2 (sqrt(1 + x^2) - 1),       x = q / c'
    Lambda_LL(q) = (1 + 1 / sqrt(1 + x^2)) / 2
    h~(q)        = -4 pi g(x) / q^4,
    g(x)         = Lambda_LL^2 * x^2 / (2 (sqrt(1 + x^2) - 1))

g -> 1 (c' -> inf) gives h = r12 / 2 (Kato).  Small x: g = 1 - x^2/4 + ...,
the 1/c'^2 term gives h += 1 / (4 c'^2 r12), whose partial-wave energy tail
is exactly Kutzelnigg's 2 pi^2 alpha^2 int |psi(r,r)|^2 r^3 dr (l+1/2)^-2.
Large x: g -> x / 8, h -> ln(r12) / (4 pi c'): a logarithmic, not a cusp,
singularity with coefficient linear in Z alpha (= 1/c').

Legendre coefficient at r1 = r2 = r (plane-wave expansion):
    c_l(r) = (2l+1)/(2 pi^2) int q^2 h~(q) j_l(qr)^2 dq
           = -(2 (2l+1)/pi) int g(q/c') j_l(qr)^2 / q^2 dq      (l >= 1)
'''

import numpy
from scipy.special import spherical_jn


def g(x):
    s = numpy.sqrt(1 + x * x)
    lam = (1 + 1 / s) / 2
    # x^2 / (2 (s - 1)) = (s + 1) / 2, no cancellation
    return lam * lam * (s + 1) / 2


def legendre_coeff(l, r, cprime=None, qmax_fac=2000., npp=40):
    '''c_l of h(r12) at r1 = r2 = r; cprime None -> non-relativistic.
    Double precision (model curve only): composite Gauss-Legendre over
    half periods of j_l^2 up to Q, averaged tail beyond.'''
    l = int(l)
    Q = qmax_fac * (l + 1) / r
    nseg = int(Q * r / numpy.pi * 2) + 1
    xg, wg = numpy.polynomial.legendre.leggauss(npp)
    # finer segments at small q where q^-2 j_l^2 is large
    edges = numpy.concatenate([numpy.linspace(0, (l + 1) / r, 200),
                               numpy.linspace((l + 1) / r, Q, nseg)[1:]])
    a, b = edges[:-1], edges[1:]
    q = (a[:, None] + b[:, None]) / 2 + (b - a)[:, None] / 2 * xg[None, :]
    w = (b - a)[:, None] / 2 * wg[None, :]
    gg = 1. if cprime is None else g(q / cprime)
    f = gg * spherical_jn(l, q * r) ** 2 / q ** 2
    tot = (f * w).sum()
    if cprime is None:
        tot += 1 / (6 * r ** 2 * Q ** 3)
    else:
        # g ~ x/8 + 1/4 + O(1/x):  int_Q^inf g / (2 q^4 r^2) dq
        tot += (1 / (8 * cprime) / (2 * Q ** 2) + 0.25 / (3 * Q ** 3)) / (2 * r ** 2)
    return -2 * (2 * l + 1) / numpy.pi * tot


def legendre_coeff2(l, r1, r2, cprime=None, qmax_fac=2000., npp=40):
    '''c_l(r1, r2) of h(r12): (2l+1)/(2 pi^2) int q^2 h~ j_l(q r1) j_l(q r2) dq'''
    l = int(l)
    r = min(r1, r2)
    Q = qmax_fac * (l + 1) / r
    nseg = int(Q * max(r1, r2) / numpy.pi * 2) + 1
    xg, wg = numpy.polynomial.legendre.leggauss(npp)
    edges = numpy.concatenate([numpy.linspace(0, (l + 1) / r, 200),
                               numpy.linspace((l + 1) / r, Q, nseg)[1:]])
    a, b = edges[:-1], edges[1:]
    q = (a[:, None] + b[:, None]) / 2 + (b - a)[:, None] / 2 * xg[None, :]
    w = (b - a)[:, None] / 2 * wg[None, :]
    gg = 1. if cprime is None else g(q / cprime)
    f = gg * spherical_jn(l, q * r1) * spherical_jn(l, q * r2) / q ** 2
    return -2 * (2 * l + 1) / numpy.pi * (f * w).sum()
