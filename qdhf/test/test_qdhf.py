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
from pyscf import gto, scf
from socutils.qdhf import qarray, integrals, dhf, x2c
from socutils.qdhf.qarray import QArray, qmp


def setUpModule():
    global mol, mol_lih
    mol = gto.M(atom='O 0 0 0.1; H 0 0.8 -0.4; H 0.1 -0.7 -0.5',
                basis='6-31g*', verbose=0)
    mol_lih = gto.M(atom='Li 0 0 0; H 0.1 0.2 1.5', basis='6-31g', verbose=0)


def tearDownModule():
    global mol, mol_lih
    del mol, mol_lih


class KnownValues(unittest.TestCase):
    def test_arithmetic(self):
        x = QArray.scalar(2).sqrt()
        self.assertEqual(qmp.nstr(x.item(), 34), qmp.nstr(qmp.sqrt(2), 34))
        third = QArray.scalar(1) / 3
        self.assertEqual(third.item(), qmp.mpf(1) / 3)
        a = QArray.from_mpf(numpy.array([[qmp.mpf(1)/3, qmp.mpf(2)/7]], dtype=object))
        b = a.to_mpmath()
        self.assertEqual(b[0, 0], qmp.mpf(1)/3)
        self.assertEqual(b[0, 1], qmp.mpf(2)/7)

    def test_eigh(self):
        numpy.random.seed(2)
        n = 20
        a = numpy.random.rand(n, n) + 1j * numpy.random.rand(n, n)
        a = a + a.conj().T
        aq = QArray.from_numpy(a)
        w, v = qarray.eigh(aq)
        self.assertAlmostEqual(abs(w.to_numpy() - numpy.linalg.eigvalsh(a)).max(), 0, 12)
        wd = QArray.zeros((n, n))
        for i in range(n):
            wd[i, i] = w[i]
        self.assertLess(float((aq @ v - v @ wd).max_abs()), 1e-30)
        self.assertLess(float((v.conj().T @ v - QArray.eye(n)).max_abs()), 1e-31)

    def test_cart2sph(self):
        for l in range(5):
            c = integrals.cart2sph(l)
            ref = gto.cart2sph(l)
            self.assertAlmostEqual(abs(numpy.array(c, dtype=float) - ref).max(), 0, 14)

    def test_int1e(self):
        for cart in (False, True):
            mol1 = mol.copy()
            mol1.cart = cart
            mol1.build()
            n = mol1.nao
            qi = integrals.QuadIntegrals(mol1)
            s = qi.ovlp_x().to_numpy()
            self.assertAlmostEqual(abs(s - mol1.intor('int1e_ovlp')).max(), 0, 14)
            v = qi.nuc_x().to_numpy()
            self.assertAlmostEqual(abs(v - mol1.intor('int1e_nuc')).max(), 0, 12)
            sy = qi.ovlp_y().to_numpy().reshape(n,3,n,3).transpose(1,3,0,2).reshape(9,n,n)
            self.assertAlmostEqual(abs(sy - mol1.intor('int1e_ipovlpip')).max(), 0, 13)
            vy = qi.nuc_y().to_numpy().reshape(n,3,n,3).transpose(1,3,0,2).reshape(9,n,n)
            self.assertAlmostEqual(abs(vy - mol1.intor('int1e_ipnucip')).max(), 0, 11)

    def test_int1e_gaussian_nuc(self):
        mol1 = mol.copy()
        mol1.nucmod = 'G'
        mol1.build()
        qi = integrals.QuadIntegrals(mol1)
        v = qi.nuc_x().to_numpy()
        self.assertAlmostEqual(abs(v - mol1.intor('int1e_nuc')).max(), 0, 12)
        n = mol1.nao
        vy = qi.nuc_y().to_numpy().reshape(n,3,n,3).transpose(1,3,0,2).reshape(9,n,n)
        self.assertAlmostEqual(abs(vy - mol1.intor('int1e_ipnucip')).max(), 0, 11)

    def test_int2e(self):
        n = mol_lih.nao
        qi = integrals.QuadIntegrals(mol_lih)
        eri = qi.eri_xxxx().to_numpy()
        self.assertAlmostEqual(abs(eri - mol_lih.intor('int2e')).max(), 0, 13)
        eri = qi.eri_yyxx().to_numpy()
        eri = eri.reshape(n,3,n,3,n,n).transpose(1,3,0,2,4,5).reshape(9,n,n,n,n)
        self.assertAlmostEqual(abs(eri - mol_lih.intor('int2e_ipvip1')).max(), 0, 13)

    def test_hamiltonian_vs_pyscf(self):
        # the quad 4c matrices transformed to the spinor basis of pyscf.scf.dhf
        ham = dhf.build_rkb_hamiltonian(mol_lih)
        ca, cb = mol_lih.sph2spinor_coeff()
        u = numpy.vstack([ca, cb])
        n2, n2c = u.shape
        u4 = numpy.zeros((n2*2, n2c*2), dtype=complex)
        u4[:n2, :n2c] = u
        u4[n2:, n2c:] = u
        mf = scf.DHF(mol_lih)
        h = u4.conj().T @ ham.hcore.to_numpy() @ u4
        s = u4.conj().T @ ham.ovlp.to_numpy() @ u4
        self.assertAlmostEqual(abs(h - mf.get_hcore()).max(), 0, 9)
        self.assertAlmostEqual(abs(s - mf.get_ovlp()).max(), 0, 13)

        dm = mf.get_init_guess(key='hcore')
        vj, vk = ham.get_jk(QArray.from_numpy(dhf.spinor_to_spin_orbital_dm(mol_lih, dm)))
        vj = u4.conj().T @ vj.to_numpy() @ u4
        vk = u4.conj().T @ vk.to_numpy() @ u4
        vj0, vk0 = mf.get_jk(mol_lih, dm)
        self.assertAlmostEqual(abs(vj - vj0).max(), 0, 9)
        self.assertAlmostEqual(abs(vk - vk0).max(), 0, 9)

    def test_dhf(self):
        mf0 = scf.DHF(mol_lih)
        mf0.conv_tol = 1e-12
        e_ref = mf0.kernel()

        mf = dhf.DHF(mol_lih)
        e_quad = mf.kernel()
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(float(e_quad), e_ref, 9)

        # same quad precision result from a different initial guess
        mf1 = dhf.DHF(mol_lih)
        mf1.init_guess = 'double'
        e1 = mf1.kernel()
        self.assertLess(abs(e1 - e_quad), 1e-26)

        # libcint double precision Hamiltonian solved in quad precision
        mf2 = dhf.DHF(mol_lih, integrals='libcint')
        e2 = mf2.kernel()
        self.assertTrue(mf2.converged)
        self.assertAlmostEqual(float(e2), e_ref, 9)
        self.assertLess(abs(e2 - e_quad), 1e-11)

        # orbital energies
        n2c = mol_lih.nao_2c()
        e_mo = mf.mo_energy.to_numpy()
        self.assertAlmostEqual(abs(e_mo[n2c:n2c+8] - mf0.mo_energy[n2c:n2c+8]).max(), 0, 8)

    def test_rhf(self):
        e_ref = scf.RHF(mol).run(conv_tol=1e-12).e_tot
        mf = dhf.RHF(mol)
        e = mf.kernel()
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(float(e), e_ref, 9)
        mf1 = dhf.RHF(mol)
        mf1.init_guess = 'double'
        self.assertLess(abs(mf1.kernel() - e), 1e-26)
        dm = mf.make_rdm1()
        s = integrals.QuadIntegrals(mol).ovlp_x()
        self.assertLess(abs(qarray.trace_dot(dm, s) - mol.nelectron), 1e-28)

    def test_ghf(self):
        mol1 = gto.M(atom='O 0 0 0', spin=2, basis='6-31g', verbose=0)
        e_ref = scf.UHF(mol1).run(conv_tol=1e-12).e_tot
        mf = dhf.GHF(mol1)
        e = mf.kernel()
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(float(e), e_ref, 9)

    def test_nonrelativistic_limit(self):
        mol1 = gto.M(atom='He', basis='6-31g', verbose=0)
        e_hf = dhf.RHF(mol1).kernel()
        a = []
        for c in ('1e4', '1e5'):
            mf = dhf.DHF(mol1, light_speed=c)
            mf.init_guess = 'double'
            a.append((mf.kernel() - e_hf) * qmp.mpf(c)**2)
        # c^2 (E_DHF - E_HF) = A + O(1/c^2)
        self.assertLess(abs(a[0] - a[1]), 1e-7)

    def test_sfx2c1e(self):
        from pyscf import lib
        from pyscf.x2c import x2c as px2c
        mol1 = gto.M(atom='He 0 0 0; He 0.2 0 2.5', basis='unc-cc-pvdz', verbose=0)
        c = lib.param.LIGHT_SPEED
        t = mol1.intor_symmetric('int1e_kin')
        v = mol1.intor_symmetric('int1e_nuc')
        w = mol1.intor_symmetric('int1e_pnucp')
        s = mol1.intor_symmetric('int1e_ovlp')
        ref = px2c._get_hcore_fw(t, v, w, s, px2c._x2c1e_xmatrix(t, v, w, s, c), c)
        h = x2c.get_hcore(mol1).to_numpy()
        self.assertAlmostEqual(abs(h - ref).max(), 0, 11)

        mf0 = scf.RHF(mol1).sfx2c1e()
        mf0.with_x2c.xuncontract = False
        mf0.with_x2c.approx = 'atom1e'
        e_ref = mf0.kernel()
        mf = x2c.SFX2C1E_RHF(mol1)
        mf.approx = 'atom1e'
        h = x2c.get_hcore(mol1, approx='atom1e').to_numpy()
        self.assertAlmostEqual(abs(h - mf0.get_hcore()).max(), 0, 12)
        self.assertAlmostEqual(float(mf.kernel()), e_ref, 10)

    def test_sfx2c1e_xuncontract(self):
        from pyscf import lib
        from pyscf.x2c import x2c as px2c
        mol1 = gto.M(atom='He 0 0 0; He 0.3 0 2.5', basis='cc-pvdz', verbose=0)
        xmol, contr = x2c.uncontract_mol(mol1)
        cd = contr.to_numpy()
        self.assertAlmostEqual(abs(cd.T @ xmol.intor('int1e_ovlp') @ cd
                                   - mol1.intor('int1e_ovlp')).max(), 0, 14)
        c = lib.param.LIGHT_SPEED
        t, v, w, s = [xmol.intor_symmetric(k) for k in
                      ('int1e_kin', 'int1e_nuc', 'int1e_pnucp', 'int1e_ovlp')]
        ref = cd.T @ px2c._get_hcore_fw(t, v, w, s, px2c._x2c1e_xmatrix(t, v, w, s, c), c) @ cd
        self.assertAlmostEqual(abs(x2c.get_hcore(mol1).to_numpy() - ref).max(), 0, 12)

        mf0 = scf.RHF(mol1).sfx2c1e()
        mf0.with_x2c.approx = 'atom1e'
        h = x2c.get_hcore(mol1, approx='atom1e').to_numpy()
        self.assertAlmostEqual(abs(h - mf0.get_hcore()).max(), 0, 12)
        mf = x2c.SFX2C1E_RHF(mol1)
        mf.approx = 'atom1e'
        self.assertAlmostEqual(float(mf.kernel()), mf0.kernel(), 10)

    def test_dipole_and_direct_jk(self):
        mol1 = gto.M(atom='F 0 0 0; H 0 0 0.917', basis='cc-pvdz', verbose=0)
        n = mol1.nao
        qi = integrals.QuadIntegrals(mol1)
        with mol1.with_common_orig((0, 0, 0)):
            r = mol1.intor('int1e_r')
            prp = mol1.intor('int1e_sprsp').reshape(3, 4, n, n)[:, 3]
        for k in range(3):
            self.assertAlmostEqual(abs(qi.dip_x(k).to_numpy() - r[k]).max(), 0, 13)
            dy = qi.dip_y(k)
            dy = (dy[0::3, 0::3] + dy[1::3, 1::3] + dy[2::3, 2::3]).to_numpy()
            self.assertAlmostEqual(abs(dy - prp[k]).max(), 0, 12)
        dm = scf.RHF(mol1).run().make_rdm1()
        vj, vk = qi.get_jk_direct(QArray.from_numpy(dm))
        vj0, vk0 = scf.hf.get_jk(mol1, dm)
        self.assertAlmostEqual(abs(vj.to_numpy() - vj0).max(), 0, 12)
        self.assertAlmostEqual(abs(vk.to_numpy() - vk0).max(), 0, 12)
        e1 = dhf.RHF(mol1).kernel()
        mf = dhf.RHF(mol1)
        mf.direct = True
        self.assertLess(abs(mf.kernel() - e1), 1e-28)

    def test_sfx2c_rs1e(self):
        # range-separated X2C-1e, formula of pyscf branch range-separated-x2c1e
        from pyscf import lib
        from pyscf.x2c import x2c as px2c, sfx2c1e
        mol1 = gto.M(atom='F 0 0 0; H 0 0 0.917; He 0 0 4', basis='cc-pvdz', verbose=0)
        omega = 0.8
        xmol, contr = sfx2c1e.SpinFreeX2CHelper(mol1).get_xmol()
        c = lib.param.LIGHT_SPEED
        t, v, w, s = [xmol.intor_symmetric(k) for k in
                      ('int1e_kin', 'int1e_nuc', 'int1e_pnucp', 'int1e_ovlp')]
        xmol._atm[:, 2] = 2
        xmol._env[xmol._atm[:, 3]] = omega**2
        vlr = xmol.intor_symmetric('int1e_nuc')
        wlr = xmol.intor_symmetric('int1e_pnucp')
        x = px2c._x2c1e_xmatrix(t, v-vlr, w-wlr, s, c)
        ref = contr.T @ (px2c._get_hcore_fw(t, v-vlr, w-wlr, s, x, c) + vlr) @ contr
        h = x2c.get_hcore(mol1, approx='rs1e', omega=omega).to_numpy()
        self.assertAlmostEqual(abs(h - ref).max(), 0, 11)

    def test_he2_x2c_long_range(self):
        # spurious R^-4 tail of the molecular sf-X2C-1e decoupling
        basis = 'unc-cc-pvdz'
        e_he = x2c.SFX2C1E_RHF(gto.M(atom='He', basis=basis, verbose=0)).kernel()
        de = []
        for r in (20, 40):
            mol1 = gto.M(atom='He 0 0 0; He 0 0 %g' % r, basis=basis, verbose=0)
            de.append(x2c.SFX2C1E_RHF(mol1).kernel() - 2 * e_he)
        self.assertAlmostEqual(float(de[0] / de[1]), 16, 4)


if __name__ == "__main__":
    print("Full Tests for quad precision DHF")
    unittest.main()
