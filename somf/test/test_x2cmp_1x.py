'''
Tests for the 'x2cmp_1x' flavor of somf/x2cmp.py (x2cmp with one-centre
exchange, cf. SOMF(1X)) and its projector-form variant 'x2cmp_1x_proj'.

  * the analytic J of the atomic densities (somf/onecenter_j.py) reproduces
    the molecular get_j build;
  * 'x2cmp-1x' / 'x2cmp-1x-proj' spellings and the proj_1x switch select the
    same Hamiltonians as the flavors;
  * the energies stay close to 'x2cmp' (one-centre K is the only approximation),
    also with diffuse functions added.
'''

import os
import contextlib
import unittest
import numpy as np

from pyscf import gto

from socutils.scf import spinor_hf


def _quiet(f):
    with open(os.devnull, 'w') as fn, contextlib.redirect_stdout(fn):
        return f()


def _run(mol, flavor, scf=True, **attrs):
    mf = spinor_hf.SCF(mol).x2cmp(flavor=flavor, with_gaunt=False, with_breit=False)
    for key, val in attrs.items():
        setattr(mf.with_x2c, key, val)
    h = _quiet(lambda: mf.with_x2c.get_hcore(mol))
    e = _quiet(mf.kernel) if scf else None
    return h, mf.with_x2c, e


def _diffuse_basis(k, ratio=2.5):
    '''uncontracted cc-pVDZ with k extra even-tempered s and p shells'''
    bas = {}
    for el in ('H', 'Cl'):
        raw = gto.mole.uncontracted_basis(gto.basis.load('cc-pvdz', el))
        extra = []
        for l in (0, 1):
            emin = min(x[-1][0] for x in raw if x[0] == l)
            extra += [[l, [emin / ratio**(i + 1), 1.]] for i in range(k)]
        bas[el] = sorted(raw + extra, key=lambda sh: sh[0])
    return bas


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='H 0 0 0; Cl 0 0 1.275', basis='cc-pvdz', verbose=0)

    def test_analytic_j(self):
        h, x, _ = _run(self.mol, 'x2cmp_1x', scf=False)
        h_ref, x_ref, _ = _run(self.mol, 'x2cmp_1x', scf=False, j_1x='direct')
        self.assertAlmostEqual(abs(x.h4c - x_ref.h4c).max(), 0, 7)
        self.assertAlmostEqual(abs(x.veff_2c - x_ref.veff_2c).max(), 0, 9)
        self.assertAlmostEqual(abs(h - h_ref).max(), 0, 9)

    def test_alias_and_switch(self):
        h1 = _run(self.mol, 'x2cmp-1x-proj', scf=False)[0]
        h2, x2, _ = _run(self.mol, 'x2cmp_1x', scf=False, proj_1x=True)
        self.assertEqual(x2.amf_type, 'x2cmp_1x')
        self.assertAlmostEqual(abs(h1 - h2).max(), 0, 10)
        self.assertEqual(_run(self.mol, 'x2cmp-1x', scf=False)[1].amf_type, 'x2cmp_1x')

    def test_energy_vs_x2cmp(self):
        e_mp = _run(self.mol, 'x2cmp')[2]
        e_1x = _run(self.mol, 'x2cmp_1x')[2]
        e_1xp = _run(self.mol, 'x2cmp_1x_proj')[2]
        # HCl/cc-pVDZ, Dirac-Coulomb: -1.18e-5 (block), -2.8e-6 (projector)
        self.assertAlmostEqual(e_1x - e_mp, -1.18e-5, 6)
        self.assertAlmostEqual(e_1xp - e_mp, -2.76e-6, 6)

    def test_diffuse(self):
        mol = gto.M(atom='H 0 0 0; Cl 0 0 1.275', basis=_diffuse_basis(6), verbose=0)
        e_mp = _run(mol, 'x2cmp')[2]
        e_1xp = _run(mol, 'x2cmp_1x_proj')[2]
        self.assertLess(abs(e_1xp - e_mp), 1e-5)


if __name__ == '__main__':
    print('Tests for the x2cmp_1x flavor')
    unittest.main()
