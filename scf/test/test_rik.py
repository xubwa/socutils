#!/usr/bin/env python
'''
socutils.scf.rik: the screened L^-1 (P|ij) builders (in core and HDF5) against
pyscf's df.incore.cholesky_eri, the SCF through an out-of-core factor, and
socutils.cd Cholesky vectors through rik.get_jk.

    python scf/test/test_rik.py
'''

import tempfile
import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto, scf, df, lib
from socutils.scf import rik

WATER2 = '''
O  0.000000  0.000000  0.000000
H  0.756950  0.000000  0.585882
H -0.756950  0.000000  0.585882
O  0.000000  0.000000  6.000000
H  0.876934  0.133858  6.359583
H -0.280961 -0.841423  6.359583
'''


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom=WATER2, basis='cc-pvdz', verbose=0)
        cls.auxmol = df.addons.make_auxmol(cls.mol, 'cc-pvdz-jkfit')
        cls.ref = df.incore.cholesky_eri(cls.mol, auxmol=cls.auxmol)

    def test_cholesky_eri(self):
        B = rik.cholesky_eri(self.mol, self.auxmol)
        self.assertAlmostEqual(abs(B - self.ref).max(), 0, 10)
        # any auxiliary basis (an automatically generated one)
        from pyscf.df.autoaux import autoaux
        aux2 = df.addons.make_auxmol(self.mol, autoaux(self.mol))
        B2 = rik.cholesky_eri(self.mol, aux2)
        self.assertAlmostEqual(abs(B2 - df.incore.cholesky_eri(self.mol, auxmol=aux2)).max(), 0, 9)

    def test_cholesky_eri_h5(self):
        with tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR) as f:
            # a tiny memory budget: several column blocks
            npair = self.ref.shape[1]
            resident = numpy.empty((45, npair))
            self.assertTrue(rik.cholesky_eri_h5(self.mol, self.auxmol, f.name, max_cols=200, resident=resident))
            dfobj = df.DF(self.mol)
            dfobj._cderi = f.name
            B = numpy.vstack(list(dfobj.loop(50)))
            self.assertTrue(rik._h5_contiguous(dfobj) is not None)
            B1 = numpy.vstack([b.copy() for b in rik.iter_factor(dfobj, 37)])
            self.assertEqual(abs(resident - B[:45]).max(), 0)
            dfobj._rik_resident, dfobj._rik_resident_src = resident, f.name
            B2 = numpy.vstack([b.copy() for b in rik.iter_factor(dfobj, 20)])
            self.assertEqual(abs(B2 - B).max(), 0)
        self.assertAlmostEqual(abs(B - self.ref).max(), 0, 10)
        self.assertEqual(abs(B1 - B).max(), 0)

    def test_scf_outcore(self):
        mf0 = scf.RHF(self.mol).density_fit(auxbasis='cc-pvdz-jkfit').run(conv_tol=1e-11)
        mf = rik.apply(scf.RHF(self.mol).density_fit(auxbasis='cc-pvdz-jkfit'))
        mf.max_memory = 1          # forces the HDF5 factor
        mf.run(conv_tol=1e-11)
        self.assertTrue(isinstance(mf.with_df._cderi, str))
        self.assertAlmostEqual(mf.e_tot, mf0.e_tot, 9)

    def test_cd(self):
        from socutils.cd import cd
        mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='cc-pvdz', verbose=0)
        e0 = scf.RHF(mol).run(conv_tol=1e-11).e_tot
        mf = rik.apply(cd.cholesky(scf.RHF(mol), tau=1e-8)).run(conv_tol=1e-11)
        self.assertAlmostEqual(mf.e_tot, e0, 7)
        mf1 = cd.cholesky(scf.RHF(mol), with_df=mf.with_df).run(conv_tol=1e-11)
        self.assertAlmostEqual(mf.e_tot, mf1.e_tot, 10)


if __name__ == '__main__':
    unittest.main()
