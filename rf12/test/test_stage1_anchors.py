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
Stage-1 anchors as automated tests (reduced grids, a few minutes).
Production numbers: rf12/notes/stage1_results.md.

A1  NR limit: Schwartz increments and the Kato cusp of the pair function
A2  c -> inf (quad): relativistic shift -> +(3/64)(Z alpha)^2 (l+1/2)^-2
A3  heavy ion: shift changes sign, LL coalescence amplitude follows the
    no-pair short-distance model (logarithm, not cusp)
A4  energy tail of a pure ln(r12) pair function: -(1/4)(l+1/2)^-3
'''

import unittest
import mpmath
from socutils.rf12 import angular, bspline, pairpw, relmodel
from socutils.rf12 import xp as X

C = '137.035999084'


def setUpModule():
    global grid
    grid = bspline.Grid(nb=110, k=9, rmin=1e-3, rmax=40., nq=18, prec=160,
                        segments=[(0.08, 0.12), (8., 0.78)])


def tearDownModule():
    global grid
    del grid


def increments(sp, ls, rcoal=()):
    ref = pairpw.KappaStates(sp, -1)
    out = {}
    coal = {}
    for l in ls:
        kaps = [-l - 1] if sp.c is None else [l, -l - 1]
        e = 0
        a = 0
        for k in kaps:
            r = pairpw.pair_kappa(sp, ref, k, rcoal=rcoal)[0]
            e += r.e2
            if rcoal:
                g = angular.singlet(angular.coalescence_tensor(k, 'LL'))
                if sp.c is None:
                    g += angular.singlet(angular.coalescence_tensor(l, 'LL'))
                a += r.coal['LL'][0] * g
        out[l] = e
        coal[l] = a
    if rcoal:
        pa = ref.at(list(rcoal))[0]
        phi0 = sp.xp.to_mp(pa, grid.ctx)[0] ** 2 * angular.singlet(angular.reference_tensor('LL'))
        coal = {l: (v / phi0).real for l, v in coal.items()}
    return out, coal


class Anchors(unittest.TestCase):
    def test_a1_nonrelativistic(self):
        sp = pairpw.Space(grid, X.Double(), None)
        e, a = increments(sp, (8, 9), rcoal=[grid.ctx.mpf(1)])
        x = mpmath.mpf(8.5)
        S = -e[8] * x ** 4 * 256 / 45
        # two-term Schwartz formula (the O((l+1/2)^-4) term is ~8e-4 at l = 8),
        # and the converged production value (xfine grid) 0.9835338
        self.assertAlmostEqual(float(S), float(1 - mpmath.mpf(5) / 4 / x ** 2), 2)
        self.assertAlmostEqual(float(S), 0.9835338, 4)
        p = -mpmath.log(e[9] / e[8]) / mpmath.log(9.5 / 8.5)
        self.assertAlmostEqual(float(p), 3.97, 1)
        s = a[8] / angular.legendre_r12(8)       # Kato: 1/2
        self.assertAlmostEqual(float(s), 0.5, 1)

    def test_a2_c_to_infinity(self):
        c = grid.ctx.mpf('1370.35999084')
        q = X.Quad()
        e_nr, _ = increments(pairpw.Space(grid, q, None), (5, 6))
        e_dc, _ = increments(pairpw.Space(grid, q, c), (5, 6))
        za2 = 1 / c ** 2
        for l in (5, 6):
            ratio = (e_dc[l] - e_nr[l]) * (l + mpmath.mpf(0.5)) ** 2 / za2
            self.assertLess(abs(ratio / (mpmath.mpf(3) / 64) - 1), 0.01)
            # the increments themselves are the NR ones to 1e-5
            self.assertLess(abs(e_dc[l] / e_nr[l] - 1), 1e-5)

    def test_a3_heavy_ion(self):
        c = grid.ctx.mpf(C) / 80
        e_nr, _ = increments(pairpw.Space(grid, X.Double(), None), (2, 10))
        e_dc, a = increments(pairpw.Space(grid, X.Double(), c), (2, 10), rcoal=[grid.ctx.mpf(1)])
        self.assertGreater(e_dc[2] - e_nr[2], 0)       # DPT regime: positive shift
        self.assertLess(e_dc[10] - e_nr[10], 0)        # beyond l ~ 1/(Z alpha): negative
        ratio = a[10] / relmodel.legendre_coeff(10, 1.0, float(c))
        self.assertTrue(1.0 < ratio < 1.2, ratio)      # production grid: 1.09

    def test_a4_log_tail(self):
        mp = mpmath.mp
        l = 20

        def a_exact(t):
            f = lambda x: mp.log(1 - 2 * t * x + t * t) / 2 * mp.legendre(l, x)
            return (2 * l + 1) * mp.quad(f, mp.linspace(-1, 1, 9)) / 2
        E = 32 * mp.quad(lambda t: t ** (l + 2) * a_exact(t) * 24 / (2 * (1 + t)) ** 5,
                         [0, 0.8, 0.9, 0.95, 0.98, 1]) / (2 * l + 1)
        self.assertAlmostEqual(float(E * (l + 0.5) ** 3), -0.25, 3)


if __name__ == '__main__':
    print('Stage-1 anchor tests for socutils.rf12')
    unittest.main()
