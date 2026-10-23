#!/usr/bin/env python
'''
GHF / GKS analytic nuclear gradients with an X2C spin-orbit hcore, against a
five-point finite difference of the SCF energy (HBr).

The X2C hcore carries ~1e-13 relative roundoff (Br 1s exponents ~6e5), which a
two-point difference with h = 1e-4 turns into ~1e-6 noise; the five-point
stencil with h = 2e-3 keeps both truncation (~h^4) and noise below 1e-8.

    python grad/test/test_ghf_gks_grad.py
'''

import unittest
import socutils  # noqa: F401
from pyscf import gto
from socutils.scf import ghf
from socutils.dft import gks
import socutils.grad.ghf_grad  # noqa: F401  (GHF.Gradients)
import socutils.grad.gks_grad  # noqa: F401  (GKS.Gradients)

BOHR = 0.52917721092
H = 2e-3


def _mol(dz=0.):
    return gto.M(atom='H 0 0 %.12f; Br 0 0 1.414' % (dz * BOHR), basis='cc-pvdz', verbose=0)


def _scf(mol, xc=None):
    if xc is None:
        mf = ghf.GHF(mol)
    else:
        mf = gks.GKS(mol, xc=xc)
        mf.grids.atom_grid = (50, 194)
    mf = mf.x2camf()
    mf.conv_tol = 1e-12
    return mf


def _fd(xc, dm0):
    e = [_scf(_mol(d), xc).kernel(dm0=dm0) for d in (2 * H, H, -H, -2 * H)]
    return (-e[0] + 8 * e[1] - 8 * e[2] + e[3]) / (12 * H)


class KnownValues(unittest.TestCase):
    def test_ghf_x2camf(self):
        mf = _scf(_mol())
        mf.kernel()
        g = mf.Gradients().kernel()
        self.assertAlmostEqual(abs(g.sum(axis=0)).max(), 0, 10)
        self.assertAlmostEqual(g[0, 2], _fd(None, mf.make_rdm1()), 7)

    def test_gks_pbe0_x2camf(self):
        # complex, spin-off-diagonal density matrices through J and K
        mf = _scf(_mol(), 'pbe0')
        mf.kernel()
        gobj = mf.Gradients()
        gobj.grid_response = True
        g = gobj.kernel()
        self.assertAlmostEqual(abs(g.sum(axis=0)).max(), 0, 8)
        self.assertAlmostEqual(g[0, 2], _fd('pbe0', mf.make_rdm1()), 7)


if __name__ == '__main__':
    print('Full tests for GHF/GKS gradients with X2C SOC')
    unittest.main()
