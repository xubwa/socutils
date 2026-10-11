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

import unittest
import numpy
import scipy.linalg
import mpmath
from socutils.rf12 import angular, bspline, pairpw, relmodel
from socutils.rf12 import xp as X


def setUpModule():
    global grid
    grid = bspline.Grid(nb=40, k=8, rmin=1e-4, rmax=40., nq=14, prec=160)


def tearDownModule():
    global grid
    del grid


class KnownValues(unittest.TestCase):
    def test_angular_weights(self):
        # W(l-1/2) + W(l+1/2) = 2/(2l+1): the jj pair weights recombine into
        # the non-relativistic 1/(2l+1)
        for l in range(1, 6):
            w = angular.pair_weight(l) + angular.pair_weight(-l - 1)
            self.assertAlmostEqual(float(w), 2. / (2 * l + 1), 14)
        self.assertAlmostEqual(float(angular.pair_weight(-1)), 2., 14)

    def test_legendre_r12(self):
        # sqrt(2(1-x)) vanishes at x = 1; c_l ~ -(l+1/2)^-2
        s = mpmath.fsum(angular.legendre_r12(l) for l in range(4000))
        self.assertAlmostEqual(float(s), 0, 3)
        l = 1000
        self.assertAlmostEqual(float(angular.legendre_r12(l) * (l + .5) ** 2), -1, 5)
        ref = mpmath.quad(lambda x: mpmath.sqrt(2 - 2 * x) * mpmath.legendre(3, x), [-1, 1]) * 7 / 2
        self.assertAlmostEqual(float(angular.legendre_r12(3)), float(ref), 14)

    def test_quad_sygv(self):
        q = X.Quad()
        rng = numpy.random.default_rng(3)
        a = rng.random((40, 40))
        a = a + a.T
        b = rng.random((40, 40))
        b = b @ b.T + 40 * numpy.eye(40)
        w, v = q.sygv(q.from_float(a), q.from_float(b))
        w0 = scipy.linalg.eigh(a, b, eigvals_only=True)
        self.assertAlmostEqual(abs(q.to_float(w) - w0).max(), 0, 12)
        vf = q.to_float(v)
        self.assertAlmostEqual(abs(vf.T @ b @ vf - numpy.eye(40)).max(), 0, 12)

    def test_dirac_1s_quad(self):
        # c/Z = 1e4: the Dirac - Schroedinger 1s difference on the same grid
        # must equal the exact relativistic shift (-1.25e-9 + ...); the
        # radial-basis error (~1e-8) cancels.  Needs quad precision.
        ctx = grid.ctx
        c = ctx.mpf(10000)
        q = X.Quad()
        e = {}
        for cc in (c, None):
            sp = pairpw.Space(grid, q, cc)
            e[cc] = q.to_mp(pairpw.KappaStates(sp, -1).e[0:1], ctx)[0]
        exact = c * c * (ctx.sqrt(1 - 1 / c ** 2) - 1) + ctx.mpf(1) / 2
        self.assertLess(abs((e[c] - e[None]) - exact), 1e-13)

    def test_double_vs_quad(self):
        c = grid.ctx.mpf('137.035999084') / 80
        e = {}
        for be in (X.Double(), X.Quad()):
            sp = pairpw.Space(grid, be, c)
            ref = pairpw.KappaStates(sp, -1)
            e[be.name] = [pairpw.pair_kappa(sp, ref, k)[0].e2 for k in (-1, 1, -2)]
        for a, b in zip(e['double'], e['quad']):
            self.assertAlmostEqual(float(a), float(b), 10)

    def test_nr_partial_waves(self):
        sp = pairpw.Space(grid, X.Double(), None)
        ref = pairpw.KappaStates(sp, -1)
        e0 = pairpw.pair_kappa(sp, ref, -1)[0].e2
        e1 = pairpw.pair_kappa(sp, ref, -2)[0].e2
        # s and p limits of the hydrogenic second-order energy (Schwartz 1962)
        self.assertAlmostEqual(float(e0), -0.12533, 5)
        self.assertAlmostEqual(float(e1), -0.026495, 4)

    def test_channels_sum(self):
        c = grid.ctx.mpf('137.035999084') / 50
        sp = pairpw.Space(grid, X.Double(), c)
        ref = pairpw.KappaStates(sp, -1)
        for k in (-1, 2):
            r = pairpw.pair_kappa(sp, ref, k, channels=True)[0]
            self.assertAlmostEqual(float(sum(r.e2ch.values())), float(r.e2), 14)

    def test_relmodel(self):
        for l in (1, 7):
            self.assertAlmostEqual(relmodel.legendre_coeff(l, 1.), float(0.5 * angular.legendre_r12(l)), 10)
        # large q: g -> x/8 (logarithm), small q: g -> 1 - x^2/4 (Kato + alpha^2/(4 r12))
        self.assertAlmostEqual(relmodel.g(1e6) / (1e6 / 8), 1, 5)
        self.assertAlmostEqual((1 - relmodel.g(1e-3)) / (1e-6 / 4), 1, 4)


if __name__ == '__main__':
    print('Full Tests for socutils.rf12')
    unittest.main()
