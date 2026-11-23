#!/usr/bin/env python
'''
RKS Hessian grid response (socutils.hessian.rhf_fast.RKSHessian.grid_response): the strict mode (True, all XC
parts with the grid response) is the derivative of the grid-response gradient (socutils.grad.rhf_fast,
grid_response True) on a grid moving with the atoms; the default ('auto') has none.  H2O 6-31G*, PBE and TPSS,
a coarse grid so that the grid response is visible.

    python hessian/test/test_rks_grid_response.py
'''

import unittest
import numpy
from pyscf import gto, dft
from socutils.grad import rhf_fast as grad_fast
from socutils.hessian import rhf_fast as hess_fast

H = 1e-3


def _mol(coords=None):
    mol = gto.M(atom='O 0 0 0.1; H 0 0.75 -0.45; H 0 -0.78 -0.5', basis='6-31g*', verbose=0)
    if coords is not None:
        mol = mol.set_geom_(coords, unit='Bohr', inplace=False)
    return mol


def _scf(mol, xc, dm0=None):
    mf = dft.RKS(mol, xc=xc)
    mf.grids.atom_grid = (30, 50)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel(dm0=dm0)
    return mf


def _grad(mf):
    g = grad_fast.RKSGradients(mf)
    g.grid_response = True
    return g.kernel()


def _check(test, xc):
    mol = _mol()
    mf = _scf(mol, xc)
    hs = hess_fast.RKSHessian(mf)
    hs.grid_response = True
    h = hs.kernel()
    hd = hess_fast.RKSHessian(mf).kernel()
    dm0 = mf.make_rdm1()
    c0 = mol.atom_coords()
    for ia, x in ((0, 2), (1, 1)):
        gs = []
        for d in (H, -H):
            c = c0.copy()
            c[ia, x] += d
            gs.append(_grad(_scf(_mol(c), xc, dm0)))
        fd = (gs[0] - gs[1]) / (2 * H)              # d g / d R_{ia, x}, (natm, 3)
        test.assertAlmostEqual(abs(h[ia, :, x, :] - fd).max(), 0, 5)
    # the translational sum rule holds with the grid response
    test.assertAlmostEqual(abs(h.sum(axis=1)).max(), 0, 6)
    test.assertGreater(abs(h - hd).max(), 1e-4)


class KnownValues(unittest.TestCase):
    def test_pbe(self):
        _check(self, 'pbe')

    def test_tpss(self):
        _check(self, 'tpss')


if __name__ == '__main__':
    unittest.main()
