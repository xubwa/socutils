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


def _build(tau):
    c = cd.CD(mol, tau=tau)
    c.verbose = 0
    return c.build()._cderi


class KnownValues(unittest.TestCase):
    def test_accuracy(self):
        # every integral within tau
        for tau in (1e-4, 1e-6):
            c = cd.CD(mol, tau=tau)
            c.verbose = 0
            L = c.build()._cderi
            self.assertLessEqual(abs(eri - L.T.dot(L)).max(), tau)
            self.assertEqual(c.metric_chol.shape, (L.shape[0],) * 2)
            self.assertEqual(sum(U.shape[1] for _, U in c.pivots), L.shape[0])

    def test_method(self):
        # method is kept for old scripts: two-step only
        e = cd.cholesky(scf.RHF(mol), tau=1e-5, method='twostep').run().e_tot
        self.assertAlmostEqual(cd.cholesky(scf.RHF(mol), tau=1e-5).run().e_tot, e, 12)
        self.assertRaises(ValueError, cd.cholesky, scf.RHF(mol), method='onestep')

    def test_compact_columns(self):
        # in-place removal of columns from the rows of a flat buffer
        n, ncol = 7, 11
        buf = numpy.random.default_rng(0).random(n * ncol + 13)
        ref = buf[:n*ncol].reshape(n, ncol).copy()
        keep = numpy.array([0, 3, 4, 9, 10])
        for blksize in (1, 3, None):
            b = buf.copy()
            out = cd._compact_columns(b, n, ncol, keep, blksize)
            self.assertTrue(numpy.shares_memory(out, b))
            self.assertAlmostEqual(abs(out[:n] - ref[:, keep]).max(), 0, 14)

    def test_outcore(self):
        # vectors written in blocks of pairs (df.DF layout): the same vectors
        from pyscf.ao2mo.outcore import _load_from_h5g
        import h5py
        c = cd.CD(mol, tau=1e-6)
        c.verbose = 0
        L = c.build()._cderi
        with tempfile.TemporaryDirectory() as tmp:
            fn = os.path.join(tmp, 'cderi.h5')
            log = lib.logger.new_logger(c, 0)
            for buflen in (7, 100, None):
                c._rebuild_outcore(mol, c.pivots, fn, 4000, log, buflen)
                with h5py.File(fn, 'r') as f:
                    nblk = len(f['j3c'])
                    L1 = _load_from_h5g(f['j3c'], 0, L.shape[0])
                self.assertAlmostEqual(abs(L1 - L).max(), 0, 12)
                if buflen == 7:
                    self.assertGreater(nblk, 1)

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
        # the estimate fails before the decomposition; with force the check
        # while the vectors are made still raises
        c = cd.CD(mol, tau=1e-8)
        c.verbose = 0
        c.max_memory = lib.current_memory()[0] - 100
        with self.assertRaisesRegex(MemoryError, 'estimated'):
            c.build()
        c.force = True
        self.assertRaises(MemoryError, c.build)

    def test_memory_estimate(self):
        # on the high side: above the number of vectors
        for tau in (1e-3, 1e-4, 1e-6, 1e-8):
            self.assertGreater(cd.estimate_nvec(mol.nao, tau), _build(tau).shape[0])


if __name__ == '__main__':
    print('Tests for socutils.cd')
    unittest.main()
