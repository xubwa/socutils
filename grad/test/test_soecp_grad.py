#!/usr/bin/env python
'''
Gradients with a spin-orbit ECP (crenbl on Se, no X2C): the derivative
integrals of ECPso (socutils.grad.ecp_so_deriv), GKS ground-state gradients
with the grid response for an ECP atom, and TDA excited-state gradients.

    python grad/test/test_soecp_grad.py
'''

import unittest
import socutils  # noqa: F401
from pyscf import gto, tdscf
from socutils.scf import ghf
from socutils.dft import gks
from socutils.grad import ecp_so_deriv, tdghf_grad
import socutils.grad.gks_grad  # noqa: F401
import socutils.hessian.gks  # noqa: F401  (mf.Hessian, used by the Z-vector gradient)

ATOM = 'Se 0 0 0; H 0 1.2 0.95; H 0 -1.1 0.9'
STATE = 4


def _mol(coords=None):
    mol = gto.M(atom=ATOM, basis={'Se': 'crenbl', 'H': 'cc-pvdz'}, ecp={'Se': 'crenbl'}, verbose=0)
    if coords is not None:
        mol = mol.set_geom_(coords, unit='Bohr', inplace=False)
    return mol


def _scf(mol, xc=None, dm0=None):
    if xc is None:
        mf = ghf.GHF(mol)
    else:
        mf = gks.GKS(mol, xc=xc)
        mf.grids.atom_grid = (50, 194)
        mf.collinear = 'ncol'
    mf.with_soc = True
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel(dm0=dm0)
    return mf


def _fd(f, ia, x, h):
    coords = _mol().atom_coords()
    v = []
    for d in (2 * h, h, -h, -2 * h):
        c = coords.copy()
        c[ia, x] += d
        v.append(f(_mol(c)))
    return (-v[0] + 8 * v[1] - 8 * v[2] + v[3]) / (12 * h)


class KnownValues(unittest.TestCase):
    def test_ecpso_deriv_integrals(self):
        mol = _mol()
        gen = ecp_so_deriv.ecpso_deriv_generator(mol)
        for ia in range(mol.natm):
            for x in range(3):
                fd = _fd(lambda m: m.intor('ECPso'), ia, x, 1e-4)
                self.assertAlmostEqual(abs(gen(ia)[x] - fd).max(), 0, 11)

    def test_ecpso_deriv2_integrals(self):
        mol = _mol()
        g2 = ecp_so_deriv.ecpso_deriv2_generator(mol)
        for L, y in ((0, 1), (2, 2)):
            for K in range(mol.natm):
                fd = _fd(lambda m: ecp_so_deriv.ecpso_deriv_generator(m)(K), L, y, 1e-4)
                self.assertAlmostEqual(abs(g2(K, L)[:, y] - fd).max(), 0, 11)

    def test_ghf_hessian(self):
        mol = gto.M(atom='H 0 0 0; Br 0.3 0 1.414', basis={'Br': 'crenbl', 'H': 'cc-pvdz'},
                    ecp={'Br': 'crenbl'}, verbose=0)
        mf = _scf(mol)
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(abs(hess.sum(axis=1)).max(), 0, 8)
        dm0 = mf.make_rdm1()
        h = 2e-3
        c0 = mol.atom_coords()
        g = []
        for d in (2 * h, h, -h, -2 * h):
            c = c0.copy()
            c[1, 2] += d
            g.append(_scf(mol.set_geom_(c, unit='Bohr', inplace=False), None, dm0).Gradients().kernel())
        col = (-g[0] + 8 * g[1] - 8 * g[2] + g[3]) / (12 * h)
        self.assertAlmostEqual(abs(hess[:, 1, :, 2] - col).max(), 0, 7)

    def test_gks_gradient_ecp_grid_response(self):
        '''the Becke radii of an ECP atom follow its atomic number'''
        mf = _scf(_mol(), 'pbe')
        g = mf.Gradients()
        g.grid_response = True
        g = g.kernel()
        dm0 = mf.make_rdm1()
        for ia, x in ((0, 1), (2, 2)):
            fd = _fd(lambda m: _scf(m, 'pbe', dm0).e_tot, ia, x, 2e-3)
            self.assertAlmostEqual(g[ia, x], fd, 8)

    def test_tda_hf(self):
        mf = _scf(_mol())
        td = tdscf.TDA(mf)
        td.nstates = 5
        td.conv_tol = 1e-10
        td.kernel()
        tg = tdghf_grad.Gradients(td)
        g = tg.grad_omega(STATE)
        tg.method = 'cphf'
        self.assertAlmostEqual(abs(g - tg.grad_omega(STATE)).max(), 0, 10)
        self.assertAlmostEqual(abs(g.sum(axis=0)).max(), 0, 10)
        dm0 = mf.make_rdm1()

        def omega(m):
            t = tdscf.TDA(_scf(m, None, dm0))
            t.nstates = 5
            t.conv_tol = 1e-10
            return t.kernel()[0][STATE - 1]
        # states 4 and 5 are 1.5 mEh apart: small step for the five-point formula
        self.assertAlmostEqual(g[2, 2], _fd(omega, 2, 2, 1e-3), 7)
        self.assertAlmostEqual(g[0, 1], _fd(omega, 0, 1, 1e-3), 6)


if __name__ == '__main__':
    print('Full tests for gradients with a spin-orbit ECP')
    unittest.main()
