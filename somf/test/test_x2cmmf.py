#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
X2CMMF reference from a converged DHF (socutils.somf.x2cmmf).

1. The 2c orbitals are orthonormal, the X2CMMF Fock matrix is diagonal in
   them with the positive-energy DHF eigenvalues, and the energy is the DHF
   energy; DHF and RDHF.
2. With double-group symmetry the orbitals keep the DHF irreps and are
   symmetry pure in the 2c basis.
3. The reference is fixed: kernel(), get_hcore() and a foreign density
   raise.
'''

import unittest
import numpy
from pyscf import gto, scf

from socutils.lib import zquatev
from socutils.symm import spinor_basis as sb
from socutils.scf import double_group_symm as dgs
from socutils.somf import x2cmmf

# pyscf.scf.dhf.RDHF imports the pip zquatev, which fails with numpy 2
scf.dhf.zquatev = zquatev


def _dhf(cls, mol):
    mf = cls(mol)
    mf.verbose = 0
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    return mf


def setUpModule():
    global mol, mol_symm
    mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='6-31g', verbose=0)
    mol_symm = sb.to_symm_frame(gto.M(atom='H 0 0 0; Br 0 0 1.41', basis='6-31g',
                                      symmetry=True, verbose=0))


def tearDownModule():
    global mol, mol_symm
    del mol, mol_symm


class KnownValues(unittest.TestCase):
    def _check_reference(self, mf4c, mf2c):
        n2c = mf4c.mol.nao_2c()
        c = mf2c.mo_coeff
        s = mf2c.get_ovlp()
        self.assertAlmostEqual(abs(c.conj().T.dot(s).dot(c) - numpy.eye(n2c)).max(), 0, 9)
        f = c.conj().T.dot(mf2c.get_fock()).dot(c)
        self.assertAlmostEqual(abs(f - numpy.diag(f.diagonal())).max(), 0, 6)
        self.assertAlmostEqual(abs(mf2c.mo_energy - mf4c.mo_energy[n2c:]).max(), 0, 6)
        self.assertAlmostEqual(mf2c.e_tot, mf4c.e_tot, 10)
        self.assertAlmostEqual(mf2c.energy_tot(dm=mf2c.make_rdm1()), mf4c.e_tot, 10)
        self.assertEqual(mf2c.mo_occ.sum(), mf4c.mol.nelectron)

    def test_dhf(self):
        mf4c = _dhf(scf.dhf.DHF, mol)
        mf2c = mf4c.x2cmmf()
        self.assertIsInstance(mf2c, x2cmmf.UHF)
        self._check_reference(mf4c, mf2c)

    def test_rdhf(self):
        mf4c = _dhf(scf.dhf.RDHF, mol)
        mf2c = mf4c.x2cmmf()
        self.assertIsInstance(mf2c, x2cmmf.RHF)
        self._check_reference(mf4c, mf2c)

    def test_symmetry(self):
        for cls, cls2c in ((dgs.SymAdaptedDHF, x2cmmf.SymAdaptedUHF),
                           (dgs.SymAdaptedRDHF, x2cmmf.SymAdaptedRHF)):
            mf4c = _dhf(cls, mol_symm)
            mf2c = mf4c.x2cmmf()
            self.assertIsInstance(mf2c, cls2c)
            self._check_reference(mf4c, mf2c)
            n2c = mol_symm.nao_2c()
            orbsym = mf2c.get_orbsym(mf2c.mo_coeff)
            self.assertTrue((orbsym == mf4c.get_orbsym(mf4c.mo_coeff)[n2c:]).all())
            so, irrep_id = mf2c._dg_symm_orb()
            c = numpy.asarray(mf2c.mo_coeff)
            self.assertTrue((dgs.label_orb_symm(c, so, irrep_id) == orbsym).all())

    def test_fixed_reference(self):
        mf2c = _dhf(scf.dhf.DHF, mol).x2cmmf()
        self.assertRaises(RuntimeError, mf2c.kernel)
        self.assertRaises(NotImplementedError, mf2c.get_hcore)
        dm = mf2c.make_rdm1()
        self.assertRaises(ValueError, mf2c.get_fock, dm=dm * .9)
        self.assertRaises(ValueError, mf2c.energy_elec, dm * .9)


if __name__ == '__main__':
    print('X2CMMF reference')
    unittest.main()
