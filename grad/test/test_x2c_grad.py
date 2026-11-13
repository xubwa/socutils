#!/usr/bin/env python
'''Tests of the X2C nuclear gradients of GHF (grad/ghf_grad with the
one-electron part from somf/x2cmp.hcore_grad, the effective densities of the
decoupling).

Reference values (z components, Eh/Bohr; SCF converged to conv_tol 1e-12,
conv_tol_grad 1e-9).  The analytic gradients were checked during development
against Richardson-extrapolated finite differences (steps 2e-3, 4e-3, 8e-3
Bohr) of the energy and of tr(D h_X2C) at fixed D: H2Te and AuH (x2c-svpall,
x2camf and x2cmp without Gaunt/Breit), I2 (dyall-v2z, x2camf) agree to
1e-9 - 1e-8, the floor of the finite differences.  The gradients sum to zero
over the atoms (translational invariance).
'''

import unittest
from pyscf import gto
from socutils.scf import ghf
from socutils.grad import ghf_grad


def _grad(atom, flavor, **kw):
    mol = gto.M(atom=atom, basis='cc-pvdz', verbose=0)
    mf = getattr(ghf.GHF(mol), flavor)(**kw)
    mf.verbose = 0
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.run()
    g = ghf_grad.Gradients(mf)
    g.verbose = 0
    return g.kernel()


class KnownValues(unittest.TestCase):
    def check(self, de, z):
        self.assertAlmostEqual(abs(de[:, :2]).max(), 0, 12)
        self.assertAlmostEqual(abs(de.sum(axis=0)).max(), 0, 12)
        for a, b in zip(de[:, 2], z):
            self.assertAlmostEqual(a, b, 8)

    def test_x2camf(self):
        # Gaunt + Breit: one-centre spin-orbit terms, no gradient contribution
        de = _grad('H 0 0 0; Br 0 0 1.41', 'x2camf')
        self.check(de, (-0.00122199306, 0.00122199306))

    def test_x2cmp(self):
        de = _grad('H 0 0 0; F 0 0 0.92', 'x2cmp', with_gaunt=False, with_breit=False)
        self.check(de, (-0.02345082926, 0.02345082926))

    def test_x2cmp_gaunt_breit_raises(self):
        # no Gaunt/Breit derivative integrals in libcint
        mol = gto.M(atom='H 0 0 0; F 0 0 0.92', basis='cc-pvdz', verbose=0)
        mf = ghf.GHF(mol).x2cmp()
        mf.verbose = 0
        mf.run()
        g = ghf_grad.Gradients(mf)
        g.verbose = 0
        self.assertRaises(NotImplementedError, g.kernel)


if __name__ == '__main__':
    print('Tests of the X2C nuclear gradients')
    unittest.main()
