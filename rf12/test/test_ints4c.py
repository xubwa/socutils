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
Integral layer of stage 3 (socutils.rf12.ints4c, lib/rf12/f12ints.c).

Four centres, one shell each (s, p, d, p), so a derivative with respect to
an atom position acts on exactly one basis function:
  (d_a i d_b j | K | k l)          = d2/dR_i,a dR_j,b (ij|K|kl)
  (d_a i d_b j | K | d_c k d_d l)  = d2/dR_k,c dR_l,d (d_a i d_b j|K|kl)
  (dd i j | K | dd k l)            = d2/dR_k dR_k (dd i j|K|kl)
and the Yukawa kernel tends to Coulomb as zeta -> 0 with an O(zeta^2)
Richardson error.
'''

import unittest
import numpy
from pyscf import gto
from socutils.rf12 import ints4c

COORDS = numpy.array([[0, 0, 0], [0.9, 0.2, -0.3], [-0.4, 1.1, 0.5], [0.3, -0.6, 1.2]])
BASIS = {'X1': [[0, [1.3, 1.0]]], 'X2': [[1, [0.9, 1.0]]], 'X3': [[2, [1.1, 1.0]]], 'X4': [[1, [0.7, 1.0]]]}
Z = 1.3
H = 1e-4


def mk(c):
    return gto.M(atom=[['X%d' % (n + 1), x] for n, x in enumerate(c)], basis=BASIS, unit='bohr', verbose=0)


def fd2(cls, kernel, a0, d0, a1, d1):
    out = 0
    for s0, s1 in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        c = COORDS.copy()
        c[a0, d0] += s0 * H
        c[a1, d1] += s1 * H
        out = out + s0 * s1 * ints4c.full(mk(c), kernel, cls, Z)
    return out / (4 * H * H)


class Ints4c(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = mk(COORDS)
        ao = cls.mol.ao_loc_nr()
        cls.blk = tuple(slice(ao[n], ao[n + 1]) for n in range(4))

    def test_plain_vs_pyscf(self):
        mol = self.mol
        mol.set_f12_zeta(Z)
        for k, name in (('G', 'int2e'), ('F', 'int2e_stg'), ('Y', 'int2e_yp')):
            self.assertLess(abs(ints4c.full(mol, k, 'plain', Z)[0, 0, 0, 0] - mol.intor(name)).max(), 1e-13)

    def test_ipvip1_fd(self):
        for k in ('G', 'F', 'Y'):
            I = ints4c.full(self.mol, k, 'ipvip1', Z)
            for a in range(3):
                for b in range(3):
                    fd = fd2('plain', k, 0, a, 1, b)[0, 0, 0, 0]
                    self.assertLess(abs(I[a, b, 0, 0][self.blk] - fd[self.blk]).max(), 1e-7)

    def test_ipvip1ipvip2_fd(self):
        for k in ('F', 'Y'):
            I = ints4c.full(self.mol, k, 'ipvip1ipvip2', Z)
            for c in (0, 2):
                for d in (1, 2):
                    fd = fd2('ipvip1', k, 2, c, 3, d)
                    s = (slice(None), slice(None)) + self.blk
                    self.assertLess(abs(I[:, :, c, d][s] - fd[:, :, 0, 0][s]).max(), 1e-7)

    def test_ipip1ipip2_fd(self):
        I = ints4c.full(self.mol, 'F', 'ipip1ipip2', Z)
        for c, d in ((0, 0), (0, 2), (1, 2)):
            fd = fd2('ipip1', 'F', 2, c, 2, d)
            s = (slice(None),) + self.blk
            self.assertLess(abs(I[:, 0, 3 * c + d, 0][s] - fd[:, 0, 0, 0][s]).max(), 1e-7)

    def test_yukawa_to_coulomb(self):
        for cls in ('ipvip1ipvip2', 'ipip1ipip2'):
            g = ints4c.full(self.mol, 'G', cls)
            err = []
            for z in (2e-3, 1e-3):
                yp = 2 * ints4c.full(self.mol, 'Y', cls, z) - ints4c.full(self.mol, 'Y', cls, 2 * z)
                err.append(abs(yp - g).max())
            self.assertAlmostEqual(err[0] / err[1], 4.0, 1)     # O(zeta^2)
            self.assertLess(err[1], 1e-4)


    def test_contraction_modes(self):
        # all four modes against the full integral tensor
        mol = self.mol
        nao = mol.nao
        rng = numpy.random.default_rng(7)
        spec = {0: ('adi,bfk,cdefijkl->acjbel', 0, 2), 1: ('adj,bfk,cdefijkl->acibel', 1, 2),
                2: ('adi,bfl,cdefijkl->acjbek', 0, 3), 3: ('adj,bfl,cdefijkl->acibek', 1, 3)}
        for cls, nc in (('ipip1', (9, 1, 1, 1)), ('ipvip1', (3, 3, 1, 1)), ('ip1ip2', (3, 1, 3, 1))):
            I = ints4c.full(mol, 'F', cls, Z)            # [ci,cj,ck,cl,i,j,k,l]
            for mode, (sub, x1, x2) in spec.items():
                A = rng.random((2, nc[x1], nao)) + 1j * rng.random((2, nc[x1], nao))
                B = rng.random((3, nc[x2], nao)) + 1j * rng.random((3, nc[x2], nao))
                T = ints4c.contract2e(mol, 'F', cls, A, B, Z, mode=mode)
                # einsum letters: c,d,e,f = comps of i,j,k,l; contracted comp index d/f relabelled
                s1 = sub.replace('ad', 'a' + 'cd'[x1]).replace('bf', 'b' + 'ef'[x2 - 2])
                out = s1.split('->')[1]
                out = 'a' + 'cd'[1 - x1] + out[2] + 'b' + 'ef'[3 - x2] + out[5]
                ref = numpy.einsum(s1.split('->')[0] + '->' + out, A, B, I)
                self.assertLess(abs(T - ref).max(), 1e-12, (cls, mode))


if __name__ == '__main__':
    unittest.main()
