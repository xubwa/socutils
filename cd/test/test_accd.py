#!/usr/bin/env python
'''Tests of the acCD auxiliary basis (socutils.cd.accd).'''

import unittest
import numpy
from pyscf import gto, scf, df
from socutils.cd import accd


def _df_error(mol, aux):
    w = df.DF(mol, auxbasis=aux).build()
    B = numpy.vstack(list(w.loop()))
    return abs(mol.intor('int2e', aosym='s4') - B.T.dot(B)).max()


class KnownValues(unittest.TestCase):
    def test_atom_error(self):
        # one centre: the pivot products of the atomic CD (residual < tau) are
        # fitted, so the integrals of the atom are reproduced to about tau
        # (strictly guaranteed only without the fit, i.e. for the exact aCD)
        tau = 1e-4
        for basis in ('cc-pvtz', 'unc-cc-pvtz'):
            mol = gto.M(atom='O 0 0 0', basis=basis, spin=2, verbose=0)
            self.assertLess(_df_error(mol, accd.accd_basis(mol, tau)), tau)

    def test_polarization_products(self):
        # the angular momenta come from the pivots: the products of the f
        # (O) and d (H) polarization functions of cc-pVTZ are represented,
        # L up to 2 l_max (autoaux stops at l_max + 1)
        mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='cc-pvtz', verbose=0)
        aux = accd.accd_basis(mol)
        self.assertEqual(max(s[0] for s in aux['O']), 6)
        self.assertEqual(max(s[0] for s in aux['H']), 4)

    def test_contraction(self):
        mol = gto.M(atom='Cl 0 0 0', basis='cc-pvtz', spin=1, verbose=0)
        shells = accd.accd_basis(mol)['Cl']
        nprim = [len(s) - 1 for s in shells]
        nctr = [len(s[1]) - 1 for s in shells]
        # contracted orbital basis: general contractions that save functions
        self.assertTrue(any(c > 1 for c in nctr))
        for p, c in zip(nprim, nctr):
            self.assertTrue(c == 1 and p == 1 or c <= accd.CONTRACT_RATIO * p)
            self.assertLessEqual(p, accd.NPRIM_MAX)
        # uncontracted orbital basis: uncontracted auxiliary functions
        mol = gto.M(atom='Cl 0 0 0', basis='unc-cc-pvtz', spin=1, verbose=0)
        for s in accd.accd_basis(mol)['Cl']:
            self.assertEqual((len(s) - 1, len(s[1]) - 1), (1, 1))

    def test_atom_labels(self):
        mol = gto.M(atom='O 0 0 0; H1 0 .757 .587; H2 0 -.757 .587',
                    basis={'O': 'cc-pvdz', 'H1': 'cc-pvdz', 'H2': 'cc-pvtz'}, verbose=0)
        aux = accd.accd_basis(mol)
        self.assertEqual(set(aux), {'O', 'H1', 'H2'})
        self.assertGreater(max(s[0] for s in aux['H2']), max(s[0] for s in aux['H1']))

    def test_scf(self):
        mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='cc-pvtz', verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis=accd.accd_basis(mol))
        self.assertAlmostEqual(mf.run(conv_tol=1e-11).e_tot, -76.05711970563, 8)

    def test_not_supported(self):
        mol = gto.M(atom='O 0 0 0', basis='cc-pvdz', spin=2, cart=True, verbose=0)
        self.assertRaises(NotImplementedError, accd.accd_basis, mol)


if __name__ == "__main__":
    print("Tests of the acCD auxiliary basis")
    unittest.main()
