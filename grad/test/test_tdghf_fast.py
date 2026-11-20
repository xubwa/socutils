#!/usr/bin/env python
'''
Fast TDA excited-state gradient (socutils.grad.tdghf_fast) against the GHF Z-vector path
(tdghf_grad.grad_elec, Gradients.fast = False; the two Z-vector solvers' tolerances differ, ~1e-8) and,
for a RIJ + COSX spinor reference (where the GHF path differentiates exact J / K), against finite
differences of the excitation energy.  H2Se/cc-pVDZ, X2CAMF.

    python grad/test/test_tdghf_fast.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto, tdscf
from socutils.scf import ghf
from socutils.dft import gks, dft
from socutils.tdscf import gks as td_gks
from socutils.grad import tdghf_grad
import socutils.hessian.gks  # noqa: F401

STATE = 4
H = 2e-3


def _mol(coords=None):
    mol = gto.M(atom='Se 0 0 0; H 0 1.2 0.95; H 0 -1.1 0.9', basis='cc-pvdz', verbose=0)
    if coords is not None:
        mol = mol.set_geom_(coords, unit='Bohr', inplace=False)
    return mol


def _run(kind, mol, dm0=None):
    if kind == 'ghf':
        mf = ghf.GHF(mol).x2camf()
    elif kind == 'gks':
        mf = gks.GKS(mol, xc='pbe0').x2camf()
        mf.grids.atom_grid = (40, 110)
        mf.collinear = 'ncol'
    else:
        mf = dft.RKS(mol, xc='pbe0').x2camf()
        mf.grids.atom_grid = (40, 110)
        if kind == 'spinor_rijcosx':
            mf.rijcosx = True
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel(dm0=dm0)
    td = td_gks.TDA(mf) if kind.startswith('spinor') else tdscf.TDA(mf)
    td.nstates = 5
    td.conv_tol = 1e-10
    td.kernel()
    return mf, td


def _grads(kind):
    mf, td = _run(kind, _mol())
    tg = td.Gradients() if kind.startswith('spinor') else tdghf_grad.Gradients(td)
    gf = tg.grad_omega(STATE)
    tg.fast = False
    go = tg.grad_omega(STATE)
    return mf, gf, go


class KnownValues(unittest.TestCase):
    def test_ghf_hf(self):
        gf, go = _grads('ghf')[1:]
        self.assertAlmostEqual(abs(gf - go).max(), 0, 7)

    def test_gks_pbe0(self):
        gf, go = _grads('gks')[1:]
        self.assertAlmostEqual(abs(gf - go).max(), 0, 7)

    def test_spinor_pbe0(self):
        gf, go = _grads('spinor')[1:]
        self.assertAlmostEqual(abs(gf - go).max(), 0, 7)

    def test_spinor_rijcosx_fd(self):
        mf, td = _run('spinor_rijcosx', _mol())
        g = td.Gradients().grad_omega(STATE)
        dm0 = mf.make_rdm1()
        c0 = _mol().atom_coords()
        w = []
        for d in (2 * H, H, -H, -2 * H):
            c = c0.copy()
            c[1, 2] += d
            w.append(_run('spinor_rijcosx', _mol(c), dm0)[1].e[STATE - 1])
        fd = (-w[0] + 8 * w[1] - 8 * w[2] + w[3]) / (12 * H)
        self.assertAlmostEqual(g[1, 2], fd, 5)


if __name__ == '__main__':
    unittest.main()
