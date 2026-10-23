#!/usr/bin/env python
'''
Nuclear derivatives of spinor (j-adapted) SCF / KS / TDA through the GHF
representation (socutils.scf.spinor_ghf) against the same quantities computed
directly in the GHF basis.

    python scf/test/test_spinor_derivs.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto
from socutils.scf import spinor_hf, ghf, spinor_ghf
from socutils.dft import dft as sdft, gks
from socutils.tdscf import gks as td_gks
import socutils.grad.ghf_grad  # noqa: F401
import socutils.grad.gks_grad  # noqa: F401


def _hbr():
    return gto.M(atom='H 0 0 0; Br 0.3 0 1.414', basis='cc-pvdz', verbose=0)


def _run(mf, xc):
    if xc is not None:
        mf.grids.atom_grid = (40, 110)
    mf = mf.x2camf()
    mf.conv_tol = 1e-12
    return mf.run()


def _spinor(mol, xc=None, kr=False):
    if xc is None:
        return _run((spinor_hf.RHF if kr else spinor_hf.SCF)(mol), None)
    return _run(sdft.SpinorDFT(mol, xc=xc), xc)


def _ghf(mol, xc=None):
    return _run(ghf.GHF(mol) if xc is None else gks.GKS(mol, xc=xc), xc)


class KnownValues(unittest.TestCase):
    def test_energy_and_gradient(self):
        mol = _hbr()
        for xc, kr in ((None, False), (None, True), ('pbe', False)):
            ms, md = _spinor(mol, xc, kr), _ghf(mol, xc)
            self.assertAlmostEqual(spinor_ghf.check_energy(ms), 0, 10)
            gs, gd = ms.Gradients(), md.Gradients()
            if xc:
                gs.grid_response = gd.grid_response = True
            self.assertAlmostEqual(abs(gs.kernel() - gd.kernel()).max(), 0, 7)

    def test_hessian(self):
        mol = _hbr()
        ms, md = _spinor(mol, 'pbe'), _ghf(mol, 'pbe')
        self.assertAlmostEqual(abs(ms.Hessian().kernel() - md.Hessian().kernel()).max(), 0, 7)

    def test_tda_gradient(self):
        mol = gto.M(atom='Se 0 0 0; H 0 1.2 0.95; H 0 -1.1 0.9', basis='cc-pvdz', verbose=0)
        ms, md = _spinor(mol, 'pbe'), _ghf(mol, 'pbe')
        tds, tdd = ms.TDA(), td_gks.TDA(md)
        for td in (tds, tdd):
            td.nstates = 4
            td.kernel()
        self.assertAlmostEqual(abs(numpy.asarray(tds.e) - numpy.asarray(tdd.e)).max(), 0, 9)
        gs, gd = tds.Gradients(), tdd.Gradients()
        gs.grid_response = gd.grid_response = True
        self.assertAlmostEqual(abs(gs.kernel(state=4) - gd.kernel(state=4)).max(), 0, 8)


if __name__ == '__main__':
    print('Full tests for spinor nuclear derivatives')
    unittest.main()
