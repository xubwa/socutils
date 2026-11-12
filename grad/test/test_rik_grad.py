#!/usr/bin/env python
'''
RI-JK (density-fitted J and K) SCF and gradient: socutils.scf.rik (J / K from
the factor) against pyscf's df_jk, socutils.grad.rhf_fast with
socutils.grad.rik_grad (the fitted K gradient with the auxiliary-basis
response) against pyscf's DF gradient, which computes the same quantity, and
against a finite difference of the RI-JK energy.

    python grad/test/test_rik_grad.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import lib, gto, scf, dft
from socutils.scf import rik
from socutils.grad import rhf_fast, rik_grad

WATER2 = '''
O  0.000000  0.000000  0.000000
H  0.756950  0.000000  0.585882
H -0.756950  0.000000  0.585882
O  0.000000  0.000000  3.000000
H  0.876934  0.133858  3.359583
H -0.280961 -0.841423  3.359583
'''


def _mf(mol, xc, use_rik=True):
    if xc == 'hf':
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol, xc=xc)
        mf.grids.level = 2
    mf = mf.density_fit(auxbasis='cc-pvdz-jkfit')
    if use_rik:
        rik.apply(mf)
    mf.conv_tol = 1e-12
    return mf.run()


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom=WATER2, basis='cc-pvdz', verbose=0)

    def test_get_jk(self):
        mol = self.mol
        mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
        mf.with_df.build()
        nao = mol.nao
        numpy.random.seed(2)
        mo = numpy.linalg.qr(numpy.random.rand(nao, nao))[0]
        occ = numpy.zeros(nao)
        occ[:10] = 2
        dm0 = (mo[:, :10] * 2).dot(mo[:, :10].T)
        ref = mf.with_df.get_jk(dm0)
        for dm in (lib.tag_array(dm0, mo_coeff=mo, mo_occ=occ), dm0):
            vj, vk = rik.get_jk(mf.with_df, dm)
            self.assertAlmostEqual(abs(vj - ref[0]).max(), 0, 11)
            self.assertAlmostEqual(abs(vk - ref[1]).max(), 0, 11)
        # indefinite symmetric (an SCF density change) and non-symmetric densities
        d1 = numpy.random.rand(nao, nao) - .5
        for dm, hermi in ((d1 + d1.T, 1), (d1, 0)):
            ref = mf.with_df.get_jk(dm, hermi=hermi)
            vj, vk = rik.get_jk(mf.with_df, dm, hermi=hermi)
            self.assertAlmostEqual(abs(vj - ref[0]).max(), 0, 11)
            self.assertAlmostEqual(abs(vk - ref[1]).max(), 0, 11)

    def test_scf_energy(self):
        for xc in ('hf', 'pbe0'):
            e0 = _mf(self.mol, xc, use_rik=False).e_tot
            e1 = _mf(self.mol, xc).e_tot
            self.assertAlmostEqual(e1, e0, 9)

    def test_grad_vs_pyscf(self):
        for xc in ('hf', 'pbe0', 'b3lyp'):
            mf = _mf(self.mol, xc)
            g = rhf_fast.Gradients(mf)
            g.grid_response = False      # fixed grid, as pyscf's reference
            g1 = g.kernel()
            self.assertTrue(g._rij_fast)
            g0 = mf.nuc_grad_method().kernel()
            self.assertAlmostEqual(abs(g1 - g0).max(), 0, 7)

    def test_grad_fd(self):
        '''the fitted K gradient (auxiliary response included) against a finite difference of the
        RI-JK HF energy'''
        mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='cc-pvdz', verbose=0)
        g = rhf_fast.Gradients(_mf(mol, 'hf')).kernel()
        h = 1e-4
        for ia, x in ((0, 2), (1, 1), (2, 2)):
            e = []
            for s in (1, -1):
                c = mol.atom_coords()
                c[ia, x] += s * h
                e.append(_mf(mol.set_geom_(c, unit='Bohr', inplace=False), 'hf').e_tot)
            self.assertAlmostEqual((e[0] - e[1]) / (2 * h), g[ia, x], 6)

    def test_grad_from_integrals(self):
        '''the three-centre-integral route of the fitted coefficients equals the factor route'''
        mf = _mf(self.mol, 'hf')
        mol = self.mol
        dm = mf.make_rdm1()
        c = rik_grad._occ_factor(mf.mo_coeff, mf.mo_occ)
        auxmol = mf.with_df.auxmol
        dJ0, dK0 = rik_grad.fitted_coefficients(mol, auxmol, dm, c, with_df=mf.with_df)
        dJ1, dK1 = rik_grad.fitted_coefficients(mol, auxmol, dm, c)
        self.assertAlmostEqual(abs(dJ1 - dJ0).max(), 0, 8)
        self.assertAlmostEqual(abs(dK1 - dK0).max(), 0, 8)


if __name__ == '__main__':
    unittest.main()
