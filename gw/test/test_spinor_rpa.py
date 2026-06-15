#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Tests for the relativistic (spinor) direct-RPA correlation/total energy.

The dRPA correlation energy is a basis-independent total-energy number (no
diagonal approximation), so the non-relativistic limit of the spinor code must
reproduce the restricted ``pyscf.gw.rpa`` value essentially to machine
precision.  The X2CAMF tests exercise a genuinely relativistic (spin-orbit)
reference and require the bundled x2camf / zquatev C backends.
'''

import unittest
import numpy as np

from pyscf import gto, scf
from pyscf.gw import rpa as prpa

from socutils.scf import spinor_hf
from socutils.gw.spinor_rpa import SpinorRPA


def setUpModule():
    global mol, rhf, mf
    mol = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=0)
    rhf = scf.RHF(mol).density_fit()
    rhf.conv_tol = 1e-12
    rhf.kernel()
    mf = spinor_hf.SpinorSCF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()


def tearDownModule():
    global mol, rhf, mf
    del mol, rhf, mf


class KnownValues(unittest.TestCase):
    def test_e_hf_equals_scf(self):
        # the EXX energy of an HF reference is just the SCF total energy
        myrpa = SpinorRPA(mf)
        self.assertAlmostEqual(myrpa.get_e_hf(), mf.e_tot, 9)

    def test_rpa_matches_rhf(self):
        r = prpa.RPA(rhf)
        r.kernel(nw=40)
        s = SpinorRPA(mf)
        s.kernel(nw=40)
        self.assertAlmostEqual(s.e_corr, r.e_corr, 8)
        self.assertAlmostEqual(s.e_tot, r.e_tot, 8)

    def test_frozen_core_matches_rhf(self):
        r = prpa.RPA(rhf, frozen=1)
        r.kernel(nw=40)
        s = SpinorRPA(mf, frozen=2)
        s.kernel(nw=40)
        self.assertEqual(s.nocc, mol.nelectron - 2)
        self.assertAlmostEqual(s.e_corr, r.e_corr, 8)

    def test_parallel_consistency(self):
        a = SpinorRPA(mf)
        a.nthreads = 1
        a.kernel(nw=40)
        b = SpinorRPA(mf)
        b.nthreads = 4
        b.kernel(nw=40)
        self.assertAlmostEqual(a.e_corr, b.e_corr, 10)


class KnownValuesSOC(unittest.TestCase):
    '''Genuinely relativistic reference (X2CAMF spin-orbit coupling).

    Requires the bundled x2camf / zquatev C backends to be compiled.
    '''

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='Ar 0 0 0', basis='cc-pvdz', verbose=0)
        cls.mf = (spinor_hf.SpinorSCF(cls.mol).x2camf()
                  .density_fit(auxbasis='cc-pvdz-jkfit'))
        cls.mf.conv_tol = 1e-11
        cls.mf.kernel()

    def test_soc_rpa_runs(self):
        myrpa = SpinorRPA(self.mf, frozen=10)
        myrpa.kernel(nw=40)
        self.assertTrue(np.isfinite(myrpa.e_corr))
        self.assertLess(myrpa.e_corr, 0.0)            # correlation lowers energy
        self.assertAlmostEqual(myrpa.get_e_hf(), self.mf.e_tot, 8)
        self.assertAlmostEqual(myrpa.e_tot, myrpa.e_hf + myrpa.e_corr, 10)


if __name__ == '__main__':
    print('Tests for relativistic spinor dRPA')
    unittest.main()
