#!/usr/bin/env python
'''
GHF / GKS analytic Hessians (socutils.hessian.ghf, .gks) with an X2C
spin-orbit hcore.

The reference is a five-point finite difference of the analytic gradient
(grad/test/test_ghf_gks_grad.py).  cc-pVDZ, not 6-31G, for Br: the
uncontracted 6-31G Br set is nearly linearly dependent and the X2C decoupling
loses ~1e-8 relative precision there.

    python hessian/test/test_ghf_gks_hess.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto, scf
from socutils.scf import ghf
from socutils.dft import gks
from socutils.hessian import ghf as ghf_hess
import socutils.hessian.gks  # noqa: F401
import socutils.grad.ghf_grad  # noqa: F401
import socutils.grad.gks_grad  # noqa: F401

H = 2e-3


def _hbr(coords=None):
    mol = gto.M(atom='H 0 0 0; Br 0.3 0 1.414', basis='cc-pvdz', verbose=0)
    if coords is not None:
        mol = mol.set_geom_(coords, unit='Bohr', inplace=False)
    return mol


def _scf(mol, xc=None, dm0=None):
    if xc is None:
        mf = ghf.GHF(mol)
    else:
        mf = gks.GKS(mol, xc=xc)
        mf.grids.atom_grid = (99, 590)
    mf = mf.x2camf()
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel(dm0=dm0)
    return mf


def _fd_column(xc, dm0, ja=1, y=2):
    '''d gradient / dR_{ja,y} (natm, 3) by a five-point difference.'''
    coords = _hbr().atom_coords()
    g = []
    for d in (2 * H, H, -H, -2 * H):
        c = coords.copy()
        c[ja, y] += d
        gobj = _scf(_hbr(c), xc, dm0).Gradients()
        if xc is not None:
            gobj.grid_response = True
        g.append(gobj.kernel())
    return (-g[0] + 8 * g[1] - 8 * g[2] + g[3]) / (12 * H)


class KnownValues(unittest.TestCase):
    def test_cphf_complex(self):
        '''the response of a complex GHF reference is real-linear in U'''
        mf = _scf(_hbr())
        occ = mf.mo_occ > 0
        nmo, nocc = occ.size, occ.sum()
        fx = ghf_hess.gen_vind(mf, mf.mo_coeff, mf.mo_occ)
        rng = numpy.random.RandomState(2)
        h1 = (rng.randn(1, nmo, nocc) + 1j * rng.randn(1, nmo, nocc)) * 1e-2
        mo1 = ghf_hess.cphf_solve(fx, mf.mo_energy, mf.mo_occ, h1, numpy.zeros_like(h1),
                                  tol=1e-12, max_cycle=100)[0]
        e = mf.mo_energy
        vir = ~occ
        res = (e[vir][:, None] - e[occ]) * mo1[0][vir] + fx(mo1)[0][vir] + h1[0][vir]
        self.assertAlmostEqual(abs(res).max(), 0, 10)

    def test_ghf_equals_rhf(self):
        mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='6-31g', verbose=0)
        hr = scf.RHF(mol).run(conv_tol=1e-12).Hessian().kernel()
        hg = ghf.GHF(mol).run(conv_tol=1e-12).Hessian().kernel()
        self.assertAlmostEqual(abs(hg - hr).max(), 0, 7)

    def test_ghf_x2camf(self):
        mf = _scf(_hbr())
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(abs(hess.sum(axis=1)).max(), 0, 6)
        col = _fd_column(None, mf.make_rdm1())
        self.assertAlmostEqual(abs(hess[:, 1, :, 2] - col).max(), 0, 6)

    def test_gks_pbe_x2camf(self):
        mf = _scf(_hbr(), 'pbe')
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(abs(hess.sum(axis=1)).max(), 0, 8)
        col = _fd_column('pbe', mf.make_rdm1())
        # no grid response in the off-diagonal blocks: ~1e-6
        self.assertAlmostEqual(abs(hess[:, 1, :, 2] - col).max(), 0, 5)


if __name__ == '__main__':
    print('Full tests for GHF/GKS Hessians with X2C SOC')
    unittest.main()
