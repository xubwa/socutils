#!/usr/bin/env python
'''Tests of the Cholesky decomposition of the ERIs (socutils.cd.cd).'''

import os
import tempfile
import unittest
import numpy
from pyscf import gto, scf, lib
from socutils.cd import cd


def setUpModule():
    global mol, eri
    mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='cc-pvdz',
                verbose=0)
    eri = mol.intor('int2e', aosym='s4')


def tearDownModule():
    global mol, eri
    del mol, eri


def _build(method, tau, **kwargs):
    c = cd.CD(mol, tau=tau, method=method, **kwargs)
    c.verbose = 0
    return c.build()._cderi


class KnownValues(unittest.TestCase):
    def test_accuracy(self):
        # every integral within tau, for both algorithms
        for method in ('onestep', 'twostep'):
            for tau in (1e-4, 1e-6):
                L = _build(method, tau)
                self.assertLessEqual(abs(eri - L.T.dot(L)).max(), tau)

    def test_same_pivots(self):
        # both algorithms take the same pivots; the old name still works
        L1 = _build('onestep', 1e-5)
        L2 = _build('twostep', 1e-5)
        L3 = _build('threeloop', 1e-5)
        self.assertEqual(L1.shape, L2.shape)
        self.assertEqual(L1.shape, L3.shape)
        self.assertAlmostEqual(abs(L1.T.dot(L1) - L2.T.dot(L2)).max(), 0, 5)

    def test_scf_energy(self):
        # the CD mean field (J and K through the DF machinery) is the RHF of
        # the assembled integrals L^T L
        mf = cd.cholesky(scf.RHF(mol), tau=1e-6)
        mf.with_df.verbose = 0
        e_cd = mf.run(conv_tol=1e-11).e_tot
        L = mf.with_df._cderi
        ref = scf.RHF(mol)
        ref._eri = L.T.dot(L)
        self.assertAlmostEqual(e_cd, ref.run(conv_tol=1e-11).e_tot, 9)
        # reference value of the validated version (3d9dea3)
        self.assertAlmostEqual(e_cd, -76.02676639601, 8)

    def test_point_group_symmetry(self):
        # the pivots are closed under the point group, so the degeneracies
        # of D6h benzene (e1g HOMO, e2u LUMO, ...) are exact in the CD mean
        # field; pivots chosen per function split them by ~tau
        r, R = 1.396, 2.479
        ang = numpy.arange(6) * numpy.pi / 3
        atoms = ([('C', (r*numpy.cos(a), r*numpy.sin(a), 0)) for a in ang]
                 + [('H', (R*numpy.cos(a), R*numpy.sin(a), 0)) for a in ang])
        benzene = gto.M(atom=atoms, basis='6-31g', verbose=0)
        mf = cd.cholesky(scf.RHF(benzene), tau=1e-4)
        mf.with_df.verbose = 0
        e = mf.run(conv_tol=1e-11).mo_energy
        de = numpy.diff(e)
        pairs = numpy.where(de < 1e-5)[0]
        nocc = benzene.nelectron // 2
        self.assertIn(nocc - 2, pairs)
        self.assertLess(de[pairs].max(), 1e-10)

    def test_save_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            fn = os.path.join(tmp, 'cderi.h5')
            mf = cd.cholesky(scf.RHF(mol), tau=1e-5, cderi_to_save=fn)
            mf.with_df.verbose = 0
            e1 = mf.run().e_tot
            self.assertEqual(cd.CD.cderi_shape(fn)[1], mol.nao*(mol.nao+1)//2)
            mf = cd.cholesky(scf.RHF(mol), cderi=fn)
            mf.with_df.verbose = 0
            self.assertAlmostEqual(mf.run().e_tot, e1, 10)

    def test_max_memory(self):
        c = cd.CD(mol, tau=1e-8)
        c.verbose = 0
        c.max_memory = lib.current_memory()[0] + .2
        self.assertRaises(MemoryError, c.build)

    def test_bad_method(self):
        c = cd.CD(mol, method='onestep2')
        c.verbose = 0
        self.assertRaises(ValueError, c.build)


if __name__ == '__main__':
    print('Tests for socutils.cd')
    unittest.main()
