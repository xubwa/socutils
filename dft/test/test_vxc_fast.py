#!/usr/bin/env python
'''
socutils.dft.vxc_fast.nr_rks against pyscf's nr_rks (with and without the
orbital tags of an SCF density), and an SCF through vxc_fast.apply.

    python dft/test/test_vxc_fast.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto, scf, dft, lib
from socutils.dft import vxc_fast

WATER2 = '''
O  0.000000  0.000000  0.000000
H  0.756950  0.000000  0.585882
H -0.756950  0.000000  0.585882
O  0.000000  0.000000  3.000000
H  0.876934  0.133858  3.359583
H -0.280961 -0.841423  3.359583
'''


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom=WATER2, basis='cc-pvdz', verbose=0)
        mf = scf.RHF(cls.mol).run()
        cls.dm = mf.make_rdm1()
        cls.dm_tagged = lib.tag_array(cls.dm, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)

    def test_nr_rks(self):
        mol = self.mol
        for xc in ('lda,vwn', 'pbe', 'b3lyp', 'tpss'):
            mf = dft.RKS(mol, xc=xc)
            mf.grids.level = 2
            mf.grids.build(with_non0tab=True)
            ni = mf._numint
            n0, e0, v0 = ni.nr_rks(mol, mf.grids, xc, self.dm)
            for dm in (self.dm, self.dm_tagged):
                n1, e1, v1 = vxc_fast.nr_rks(ni, mol, mf.grids, xc, dm)
                self.assertAlmostEqual(n1, n0, 10)
                self.assertAlmostEqual(e1, e0, 10)
                self.assertAlmostEqual(abs(v1 - v0).max(), 0, 11)

    def test_scf(self):
        for xc in ('pbe', 'pbe0'):
            e = []
            for fast in (False, True):
                mf = dft.RKS(self.mol, xc=xc)
                mf.grids.level = 2
                mf.conv_tol = 1e-11
                if fast:
                    vxc_fast.apply(mf)
                e.append(mf.kernel())
            self.assertAlmostEqual(e[1], e[0], 9)


if __name__ == '__main__':
    unittest.main()
