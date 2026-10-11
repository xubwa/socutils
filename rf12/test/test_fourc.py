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
Four-component layer of stage 3 (socutils.rf12.fourc).

F1  4c MO integrals assembled from sph derivative classes == pyscf spinor
    integrals (int2e_spinor, int2e_spsp1_spinor, int2e_spsp1spsp2_spinor)
F2  anchor (a) for the reference and the conventional pair part: DHF and
    no-pair DHF-MP2 at c = 10 c0, 20 c0, Richardson in 1/c^2 -> the
    non-relativistic HF / MP2 of stage 2 (same basis)
F3  RKB complete space "1" (dc_space): orthonormal, occupied block of the
    Fock matrix = DHF orbital energies, and the no-pair CABS singles at
    c -> infinity == stage-2 CABS singles (this anchor rejected the
    block-diagonal definition of C+-, see dc_space docstring)
'''

import unittest
import numpy
from pyscf import gto, scf, mp
from socutils.rf12 import fourc, dc_space
from socutils.rf12.nr import gbs

C0 = 137.035999084


class FourC(unittest.TestCase):
    def test_f1_mo_eri_vs_pyscf(self):
        mol = gto.M(atom='Ne 0 0 0; H 0 0 1.8', unit='bohr', basis='cc-pvdz', charge=1, verbose=0)
        with fourc.light_speed(C0):
            mf = scf.DHF(mol)
            mf.conv_tol = 1e-10
            mf.kernel()
            n2c = mol.nao_2c()
            sel = numpy.r_[n2c:n2c + 3, n2c + 10:n2c + 14]
            orb = fourc.orbitals(mol, mf.mo_coeff[:, sel], C0)
            mine = fourc.mo_eri(mol, 'G', orb, orb, orb, orb)
            cL, cS = mf.mo_coeff[:n2c, sel], mf.mo_coeff[n2c:, sel] / (2 * C0)
            t = lambda I, a, b, c, d: numpy.einsum('ijkl,ip,jq,kr,ls->pqrs', I, a.conj(), b, c.conj(), d, optimize=True)
            SL = t(mol.intor('int2e_spsp1_spinor'), cS, cS, cL, cL)
            chem = (t(mol.intor('int2e_spinor'), cL, cL, cL, cL) + SL + SL.transpose(2, 3, 0, 1)
                    + t(mol.intor('int2e_spsp1spsp2_spinor'), cS, cS, cS, cS))
            self.assertLess(abs(mine - chem.transpose(0, 2, 1, 3)).max(), 1e-12)

    def test_f2_c_to_infinity(self):
        for atom, els in (('He 0 0 0', ['He']), ('Ne 0 0 0', ['Ne'])):
            mol = gto.M(atom=atom, unit='bohr', basis=gbs.basis('aug-cc-pvdz', els), verbose=0)
            rhf = scf.RHF(mol)
            rhf.conv_tol = 1e-12
            rhf.kernel()
            emp2 = mp.MP2(rhf).kernel()[0]
            e = {}
            for f in (10, 20):
                with fourc.light_speed(C0 * f):
                    mf = scf.DHF(mol)
                    mf.conv_tol = 1e-11
                    mf.kernel()
                    e[f] = (mf.e_tot, fourc.mp2(mf, C0 * f))
            for i, ref in ((0, rhf.e_tot), (1, emp2)):
                self.assertAlmostEqual((4 * e[20][i] - e[10][i]) / 3, ref, 7)


    def test_f3_complete_space_singles(self):
        import json, os
        ref = json.load(open(os.path.join(os.path.dirname(__file__), 'data', 'psi4_aug-cc-pvdz.json')))
        for name, tol in (('He', 9), ('Ne', 7)):
            mol = gto.M(atom=name + ' 0 0 0', unit='bohr', basis=gbs.basis('aug-cc-pvdz', [name]), verbose=0)
            es = {}
            for f in (10, 20):
                with fourc.light_speed(C0 * f):
                    mf = scf.DHF(mol)
                    mf.conv_tol = 1e-11
                    mf.kernel()
                sp = dc_space.DCSpace(mf, gbs.basis('aug-cc-pvdz-optri', [name]), C0 * f)
                self.assertLess(sp.metric_error, 1e-9)
                es[f] = sp.cabs_singles()
            self.assertAlmostEqual((4 * es[20] - es[10]) / 3, ref[name + '_ae']['F12 CABS CORRECTION ENERGY'], tol)


if __name__ == '__main__':
    unittest.main()
