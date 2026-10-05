'''
Tests for the X2C model-potential Hamiltonian (somf/x2cmp.py).

The 'x2cmp' flavor takes the one-centre part of J-K from the atomic
calculations and integrates only the multi-centre part (needs
lib/libsocvhf.so).  That path must reproduce the full molecular J/K build.
'''

import unittest
import numpy as np

from pyscf import gto

from socutils.scf import spinor_hf
from socutils.somf import x2cmp


def _hcore(mol, atomic_1c, **kwargs):
    mf = spinor_hf.SCF(mol).x2cmp(**kwargs)
    mf.with_x2c.atomic_1c = atomic_1c
    return mf.with_x2c.get_hcore(mol), mf.with_x2c


@unittest.skipIf(x2cmp._mc_prescreen_pointers() is None, 'lib/libsocvhf.so is not built')
class KnownValues(unittest.TestCase):
    def _compare(self, mol, **kwargs):
        h_ref, x_ref = _hcore(mol, False, **kwargs)
        h, x = _hcore(mol, True, **kwargs)
        self.assertAlmostEqual(abs(h - h_ref).max(), 0, 9)
        self.assertAlmostEqual(abs(x.h4c - x_ref.h4c).max(), 0, 10)
        self.assertAlmostEqual(abs(x.veff_2c - x_ref.veff_2c).max(), 0, 10)
        self.assertAlmostEqual(abs(x.soc_matrix - x_ref.soc_matrix).max(), 0, 9)

    def test_dirac_coulomb_breit(self):
        mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=0)
        self._compare(mol)

    def test_dirac_coulomb_gaunt(self):
        mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=0)
        self._compare(mol, with_breit=False)

    def test_dirac_coulomb_open_shell_atoms_gaussian_nuclei(self):
        mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='ccpvdz',
                    nucmod='G', verbose=0)
        self._compare(mol, with_gaunt=False, with_breit=False)


if __name__ == '__main__':
    print('Tests for the X2C model potential')
    unittest.main()
