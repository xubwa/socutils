#!/usr/bin/env python
'''
socutils.scf.md_j: MD J engine J matrices against pyscf's direct J, and the SCF
hook (J-only builds from the J engine, J + K from jk_multi, incremental Fock
builds) against plain pyscf SCF energies.

    python scf/test/test_md_j.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto, scf, dft
from socutils.scf import md_j

WATER2 = '''
O  0.000000  0.000000  0.000000
H  0.756950  0.000000  0.585882
H -0.756950  0.000000  0.585882
O  0.000000  0.000000  3.000000
H  0.876934  0.133858  3.359583
H -0.280961 -0.841423  3.359583
'''
# symmetric: pairs of coincident pair centres (Boys functions at t ~ 0)
PROPANE = '''C 0 0 0; C 1.249173 0.884052 0; C 2.498346 0 0
H -0.628924 0 0.890229; H -0.628924 0 -0.890229; H 1.249173 1.517 0.890229; H 1.249173 1.517 -0.890229
H 3.127270 0 0.890229; H 3.127270 0 -0.890229; H -0.512 -0.95 0; H 3.010 -0.95 0'''


class KnownValues(unittest.TestCase):
    def test_get_j(self):
        rng = numpy.random.default_rng(1)
        for atom, basis in ((WATER2, 'cc-pvdz'), (PROPANE, 'sto-3g'),
                            ('O 0 0 0; H .757 0 .586; H -.757 .1 .586', 'cc-pvtz')):
            mol = gto.M(atom=atom, basis=basis, verbose=0)
            d = rng.standard_normal((3, mol.nao, mol.nao))
            d = d + d.transpose(0, 2, 1)
            ref = scf.hf.get_jk(mol, d, with_k=False)[0]
            self.assertAlmostEqual(abs(md_j.get_j(mol, d) - ref).max(), 0, 10)
            # a non-symmetric density
            a = rng.standard_normal((mol.nao, mol.nao))
            ref = scf.hf.get_jk(mol, a, hermi=0, with_k=False)[0]
            self.assertAlmostEqual(abs(md_j.get_j(mol, a, hermi=0) - ref).max(), 0, 10)

    def test_scf(self):
        mol = gto.M(atom=WATER2, basis='cc-pvdz', verbose=0)
        for xc in ('hf', 'pbe', 'pbe0'):
            e = []
            for mode in (None, False, True):
                mf = scf.RHF(mol) if xc == 'hf' else dft.RKS(mol, xc=xc)
                if xc != 'hf':
                    mf.grids.level = 2
                mf.conv_tol = 1e-11
                if mode is not None:
                    md_j.apply(mf, incremental=mode)
                e.append(mf.kernel())
            self.assertAlmostEqual(e[1], e[0], 9)
            self.assertAlmostEqual(e[2], e[0], 9)


if __name__ == '__main__':
    unittest.main()
