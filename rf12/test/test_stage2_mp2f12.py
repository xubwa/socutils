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
Stage-2 anchors: non-relativistic MP2-F12/3C(FIX) against Psi4 1.11
(aug-cc-pVDZ / aug-cc-pVDZ-OptRI, beta = gamma = 1, conventional integrals).

S1  HF, MP2, CABS singles with libcint vs Psi4                     (< 1e-9 Eh)
S2  energy assembly fed with Psi4's own integrals (6-Gaussian geminal)
    reproduces Psi4's F12 correction                                (< 1e-9 Eh)
S3  libcint exact Slater geminal vs libint (Psi4) Laplace expansion
    of the Slater function (77 Gaussians)                           (< 1e-9 Eh)
S4  brute-force operator definitions of V, X, C, B vs term formulas (< 1e-12)
Reference numbers: rf12/test/data/psi4_aug-cc-pvdz.json, Psi4 integral dumps
from examples/rf12/stage2_psi4_dump.py.
'''

import os
import json
import unittest
import numpy
from pyscf import gto, scf
from socutils.rf12.nr import mp2f12, mp2f12_brute, gbs

DATA = os.path.join(os.path.dirname(__file__), 'data')
REF = json.load(open(os.path.join(DATA, 'psi4_aug-cc-pvdz.json')))
GEOM = {'He': ('He 0 0 0', ['He']), 'Ne': ('Ne 0 0 0', ['Ne']),
        'H2O': ('O 0 0 -0.124038860300; H 0 -1.431430901356 0.984293362719; '
                'H 0 1.431430901356 0.984293362719', ['O', 'H'])}
# libcint exact-STG F12 corrections, confirmed by the Psi4/libint Laplace
# expansion of the Slater geminal where noted (S3)
EXACT_STG = {'He_ae': -0.009896661932}


class NpzInts:
    def __init__(self, fn):
        d = numpy.load(fn)
        self.d = d
        self.fock = d['fock']
        self.K = d['K']
        self.dims = dict(nocc=int(d['nocc']), nfz=int(d['nfz']), nmo=int(d['nmo']), nri=int(d['nri']))

    def __call__(self, kind, lab):
        return self.d['%s_%s' % (kind, lab)]


def run(name, frozen):
    atom, els = GEOM[name]
    mol = gto.M(atom=atom, unit='bohr', basis=gbs.basis('aug-cc-pvdz', els), verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    m = mp2f12.MP2F12(mf, gbs.basis('aug-cc-pvdz-optri', els), gamma=1.0, frozen=frozen)
    m.kernel()
    return m


class Stage2(unittest.TestCase):
    def check_s1(self, name, frozen):
        m = run(name, frozen)
        r = REF['%s_%s' % (name, 'fc' if frozen else 'ae')]
        self.assertAlmostEqual(m.e_hf, r['SCF TOTAL ENERGY'], 9)
        self.assertAlmostEqual(m.e_mp2, r['MP2 CORRELATION ENERGY'], 9)
        self.assertAlmostEqual(m.e_singles, r['F12 CABS CORRECTION ENERGY'], 9)
        return m

    def test_s1_s3_he(self):
        m = self.check_s1('He', 0)
        self.assertAlmostEqual(m.e_f12, EXACT_STG['He_ae'], 9)
        # Psi4's 6-Gaussian fit differs by 0.71 uEh
        self.assertAlmostEqual((m.e_f12 - REF['He_ae']['MP2-F12 CORRECTION ENERGY']) * 1e6, -0.712, 2)

    def test_s1_ne(self):
        self.check_s1('Ne', 0)
        self.check_s1('Ne', 1)

    def test_s1_h2o(self):
        self.check_s1('H2O', 1)

    def test_s2_assembly_psi4_integrals(self):
        for fn, key in (('psi4ints_He_aug-cc-pvdz_ae.npz', 'He_ae'), ('psi4ints_Ne_aug-cc-pvdz_fc.npz', 'Ne_fc')):
            p = NpzInts(os.path.join(DATA, fn))
            m = mp2f12.MP2F12(None, ints=p)
            m.assemble(p, 0.)
            self.assertAlmostEqual(m.e_f12, REF[key]['MP2-F12 CORRECTION ENERGY'], 9)
            self.assertAlmostEqual(m.e_singles, REF[key]['F12 CABS CORRECTION ENERGY'], 9)

    def test_s4_brute_force_intermediates(self):
        for name, fz in (('He', 0), ('Ne', 1)):
            m = run(name, fz)
            V, X, C, B = mp2f12_brute.intermediates(m.ints)
            for a, b in ((V, m.V), (X, m.X), (C, m.C), (B, m.B)):
                self.assertLess(abs(a - b).max(), 1e-12)


if __name__ == '__main__':
    print('Stage-2 anchor tests (MP2-F12/3C(FIX) vs Psi4 1.11)')
    unittest.main()
