#!/usr/bin/env python
'''
TDA excited-state gradients on a GHF / GKS reference with an X2C spin-orbit
hcore (socutils.grad.tdghf_grad), H2Se/cc-pVDZ, X2CAMF.

    python grad/test/test_tdghf_grad.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto, tdscf
from socutils.scf import ghf
from socutils.dft import gks
from socutils.grad import tdghf_grad
import socutils.hessian.gks  # noqa: F401

H = 2e-3
STATE = 4          # well separated from its neighbours (states 1-3 are near-degenerate)


def _mol(coords=None):
    mol = gto.M(atom='Se 0 0 0; H 0 1.2 0.95; H 0 -1.1 0.9', basis='cc-pvdz', verbose=0)
    if coords is not None:
        mol = mol.set_geom_(coords, unit='Bohr', inplace=False)
    return mol


def _run(mol, xc=None, dm0=None):
    if xc is None:
        mf = ghf.GHF(mol).x2camf()
    else:
        mf = gks.GKS(mol, xc=xc).x2camf()
        mf.grids.atom_grid = (40, 110)
        mf.collinear = 'ncol'
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel(dm0=dm0)
    td = tdscf.TDA(mf)
    td.nstates = 4
    td.conv_tol = 1e-9
    td.kernel()
    return mf, td


def _fd(xc, dm0, ia, x):
    coords = _mol().atom_coords()
    w = []
    for d in (2 * H, H, -H, -2 * H):
        c = coords.copy()
        c[ia, x] += d
        w.append(_run(_mol(c), xc, dm0)[1].e[STATE - 1])
    return (-w[0] + 8 * w[1] - 8 * w[2] + w[3]) / (12 * H)


class KnownValues(unittest.TestCase):
    def test_hf(self):
        mf, td = _run(_mol())
        tg = tdghf_grad.Gradients(td)
        g = tg.grad_omega(STATE)
        tg.method = 'cphf'
        self.assertAlmostEqual(abs(g - tg.grad_omega(STATE)).max(), 0, 9)
        self.assertAlmostEqual(abs(g.sum(axis=0)).max(), 0, 10)
        self.assertAlmostEqual(abs(g[:, 0]).max(), 0, 10)      # planar: no x force
        self.assertAlmostEqual(g[2, 2], _fd(None, mf.make_rdm1(), 2, 2), 7)

    def test_pbe_grid_response(self):
        mf, td = _run(_mol(), 'pbe')
        tg = tdghf_grad.Gradients(td)
        tg.grid_response = True
        g = tg.grad_omega(STATE)
        tg.method = 'cphf'
        self.assertAlmostEqual(abs(g - tg.grad_omega(STATE)).max(), 0, 9)
        self.assertAlmostEqual(abs(g.sum(axis=0)).max(), 0, 9)
        self.assertAlmostEqual(g[2, 2], _fd('pbe', mf.make_rdm1(), 2, 2), 7)

    def test_kernel_energy(self):
        '''E_f is the XC part of Tr[G_td[X] X^H]'''
        mf, td = _run(_mol(), 'pbe')
        tg = tdghf_grad.Gradients(td)
        with tg._response_env():
            occ = mf.mo_occ > 0
            z = td.xy[STATE - 1][0]
            X = mf.mo_coeff[:, ~occ].dot(z.T).dot(mf.mo_coeff[:, occ].conj().T)
            vt = td.gen_response(hermi=0)(X[None])[0]
            vj, vk = mf.get_jk(mf.mol, X, hermi=0)
            hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=0)[2]
            ref = numpy.einsum('pq,pq', vt - vj + hyb * vk, X.conj()).real
            ef = tdghf_grad.ef_energy(tg, mf.mol, mf.grids, X, mf.make_rdm1(), [0, 1, 2],
                                      mf.xc, mf._numint._xc_type(mf.xc))
        self.assertAlmostEqual(ef, ref, 12)


if __name__ == '__main__':
    print('Full tests for TDA-GHF/GKS gradients with X2C SOC')
    unittest.main()
