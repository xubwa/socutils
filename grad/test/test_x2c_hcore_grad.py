#!/usr/bin/env python
'''
X2C hcore term of the nuclear gradient: the adjoint (somf.x2c_grad.hcore_grad,
one decoupling per density) against the per-perturbation generators (eigh and
the original sqrtm implementation of get_hfw1) and finite differences.

    python grad/test/test_x2c_hcore_grad.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto
from socutils.scf import spinor_hf, spinor_ghf
from socutils.somf import x2c_grad
import socutils.grad.ghf_grad  # noqa: F401


def _mol(z=0.):
    return gto.M(atom='H 0 0 %.8f; Br 0.3 0 1.414' % z, basis='cc-pvdz', verbose=0)


def _scf(mol, gb=True, flavor='x2camf'):
    if flavor == 'x2cmp':
        mf = spinor_hf.RHF(mol).x2cmp(with_gaunt=gb, with_breit=gb)
    else:
        mf = spinor_hf.RHF(mol).x2camf(flavor=flavor, with_gaunt=gb, with_breit=gb)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    return mf.run()


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = _mol()
        cls.mf = _scf(cls.mol)

    def tearDown(self):
        x2c_grad.HCORE_GRAD, x2c_grad.HFW1_IMPL = 'adjoint', 'eigh'

    def test_adjoint_vs_generators(self):
        mol, mf = self.mol, self.mf
        dm = mf.make_rdm1()
        adj = x2c_grad.hcore_grad(mf.with_x2c, dm, mol, basis='spinor')
        for impl in ('eigh', 'sqrtm'):
            x2c_grad.HFW1_IMPL = impl
            gen = x2c_grad.hcore_grad_generator_spinor(mf.with_x2c, mol)
            ref = numpy.array([numpy.einsum('xij,ji->x', gen(i), dm).real for i in range(mol.natm)])
            self.assertAlmostEqual(abs(adj - ref).max(), 0, 10, msg=impl)
        x2c_grad.HFW1_IMPL = 'eigh'
        g = spinor_ghf.to_ghf(mf)
        dmg = g.make_rdm1()
        gen = x2c_grad.hcore_grad_generator_spin_orbital(g.with_x2c, mol)
        ref = numpy.array([numpy.einsum('xij,ji->x', gen(i), dmg).real for i in range(mol.natm)])
        adj = x2c_grad.hcore_grad(g.with_x2c, dmg, mol, basis='spin_orbital')
        self.assertAlmostEqual(abs(adj - ref).max(), 0, 10)

    def test_gradient_paths(self):
        gs = {}
        for hg in ('adjoint', 'perturb'):
            x2c_grad.HCORE_GRAD = hg
            gs[hg] = self.mf.Gradients().kernel()
        self.assertAlmostEqual(abs(gs['adjoint'] - gs['perturb']).max(), 0, 10)

    def test_finite_difference(self):
        '''x2camf (Gaunt + Breit, Dirac-Coulomb), x2cmp (Dirac-Coulomb: the 4c J-K[D4c] and 2c J-K[D2c]
        derivatives), sf1e (spin-free 4c matrices)'''
        h = 1e-3
        for flavor, gb in (('x2camf', True), ('x2camf', False), ('x2cmp', False), ('sf1e', False)):
            mf = self.mf if (flavor, gb) == ('x2camf', True) else _scf(self.mol, gb, flavor)
            gz = mf.Gradients().kernel()[0, 2]
            e = [_scf(_mol(s * h), gb, flavor).e_tot for s in (2, 1, -1, -2)]
            fd = (-e[0] + 8 * e[1] - 8 * e[2] + e[3]) / (12 * h) * 0.52917721092
            self.assertAlmostEqual(gz, fd, 7, msg='%s gaunt+breit %s' % (flavor, gb))

    def test_unsupported_flavour(self):
        mf = spinor_hf.RHF(self.mol).x2cmp(with_gaunt=True, with_breit=True)
        mf.max_cycle = 1
        mf.kernel()
        with self.assertRaises(NotImplementedError):
            mf.Gradients().kernel()

if __name__ == '__main__':
    print('Full tests for the X2C hcore gradient')
    unittest.main()
