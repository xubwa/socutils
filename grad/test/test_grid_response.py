#!/usr/bin/env python
'''
Grid response of the XC and COSX gradients (socutils.grad.grid_response): at a fixed density the
analytic derivative of the energy on a grid moving with the atoms against central finite differences
with the grid rebuilt at the displaced geometries, translational invariance, and the default
(grid_response=False) unchanged.

    OMP_NUM_THREADS=4 python grad/test/test_grid_response.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto, scf, dft
from socutils.dft import grids as sgrids
from socutils.grad import rhf_fast, grid_response
from socutils.scf import cosx

H2O = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
STEP = 1e-4
# (atom, direction) of the finite differences checked
COORDS = ((0, 2), (1, 1), (2, 2))


def _grids(mol, kind):
    sgrids.PARTITION = kind
    g = sgrids.Grids(mol)
    g.intacc = 3.5
    g.angular_grid = 2
    return g.build(with_non0tab=True)


def _fd(mol, energy):
    out = {}
    c0 = mol.atom_coords()
    for a, x in COORDS:
        e = []
        for s in (1, -1):
            c = c0.copy()
            c[a, x] += s * STEP
            e.append(energy(mol.set_geom_(c, unit='Bohr', inplace=False)))
        out[a, x] = (e[0] - e[1]) / (2 * STEP)
    return out


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.partition = sgrids.PARTITION
        cls.mol = gto.M(atom=H2O, basis='def2-svp', verbose=0)
        cls.dm = scf.RHF(cls.mol).run(conv_tol=1e-10).make_rdm1()

    @classmethod
    def tearDownClass(cls):
        sgrids.PARTITION = cls.partition

    def _xc(self, xc, kind):
        mol, dm = self.mol, self.dm
        mf = dft.RKS(mol, xc=xc)
        mf.grids = _grids(mol, kind)
        g = rhf_fast.Gradients(mf)
        g0 = rhf_fast.xc_grad(g, mol, dm)
        g1 = rhf_fast.xc_grad(g, mol, dm, grid_response=True)
        fd = _fd(mol, lambda m: dft.numint.NumInt().nr_rks(m, _grids(m, kind), xc, dm)[1])
        for (a, x), v in fd.items():
            self.assertAlmostEqual(g1[a, x], v, delta=2e-8)
        self.assertLess(abs(g1.sum(axis=0)).max(), 1e-11)
        self.assertGreater(abs(g0.sum(axis=0)).max(), 1e-7)       # the fixed grid is not invariant

    def test_xc_lda_becke(self):
        self._xc('lda,vwn', 'becke')

    def test_xc_pbe_ssf(self):
        self._xc('pbe', 'ssf')

    def test_xc_pbe_block_path(self):
        fused = rhf_fast.USE_FUSED_XC
        rhf_fast.USE_FUSED_XC = False
        try:
            self._xc('pbe', 'becke')
        finally:
            rhf_fast.USE_FUSED_XC = fused

    def _cosx(self, kind, fit=True):
        mol, dm = self.mol, self.dm

        def kobj(m):
            sgrids.PARTITION = kind
            return cosx.COSX(m, cosx.make_grids(m, **cosx.gridx_args(m, 1)), fit=fit, tol=1e-14).build()
        k = kobj(mol)
        g1 = k.grad(dm, tol=1e-14, grid_response=True)
        fd = _fd(mol, lambda m: kobj(m).energy(dm, tol=1e-14))
        for (a, x), v in fd.items():
            self.assertAlmostEqual(g1[a, x], v, delta=1e-7)
        self.assertLess(abs(g1.sum(axis=0)).max(), 1e-11)
        # the point terms are minus the sum of the fixed-grid derivatives over the atoms
        g0 = k.grad(dm, tol=1e-14)
        self.assertGreater(abs(g0.sum(axis=0)).max(), 1e-6)

    def test_cosx_becke(self):
        self._cosx('becke')

    def test_cosx_ssf_nofit(self):
        self._cosx('ssf', fit=False)

    def test_partition_weights(self):
        # sum_g w_g f(u_g), f moving with the point: only the weights respond
        mol = gto.M(atom='C 0 0 -1.1; Cl 0 0 0.68; H 1.03 0 -1.45; H -0.515 0.892 -1.45; H -0.515 -0.892 -1.45',
                    basis='sto-3g', verbose=0)
        for kind in ('becke', 'ssf'):
            def energy(m, ret=False):
                g = _grids(m, kind)
                u = g.coords - m.atom_coords()[numpy.maximum(g.atm_idx, 0)]
                f = numpy.where(g.atm_idx >= 0, numpy.exp(-.3 * (u**2).sum(1)) * (1 + .2 * u[:, 0]), 0)
                return (g, f) if ret else numpy.dot(g.weights, f)
            g, f = energy(mol, True)
            ana = grid_response.partition_grad(g, f)
            self.assertLess(abs(ana.sum(axis=0)).max(), 1e-11)
            for (a, x), v in _fd(mol, energy).items():
                self.assertAlmostEqual(ana[a, x], v, delta=1e-7)

    def test_default_auto(self):
        mol = self.mol
        mf = dft.RKS(mol, xc='pbe')
        mf.grids = _grids(mol, 'becke')
        mf.conv_tol = 1e-11
        mf.kernel()
        g = rhf_fast.Gradients(mf)
        self.assertEqual(g.grid_response, 'auto')
        ref = mf.Gradients().kernel()                   # pyscf, fixed grid
        g.grid_response = False
        self.assertLess(abs(g.kernel() - ref).max(), 1e-7)
        # 'auto': the XC grid response (a pure functional: the whole gradient) equals True
        g.grid_response = 'auto'
        ga = g.kernel()
        g.grid_response = True
        self.assertLess(abs(ga - g.kernel()).max(), 1e-10)
        g.grid_response = 'auto'
        g.grid_response = True
        g1 = g.kernel()
        self.assertLess(abs(g1.sum(axis=0)).max(), 1e-7)
        self.assertLess(abs(g1 - ref).max(), 5e-3)     # coarse grid

    def test_auto_cosx_near_convergence(self):
        # 'auto' with COSX: the COSX grid response only once the previous gradient's max |g| fell below
        # COSX_GRID_RESPONSE_GMAX; the XC response always
        mol = self.mol
        mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis='def2-universal-jfit', only_dfj=True)
        mf.grids = _grids(mol, 'becke')
        mf.conv_tol = 1e-11
        mf = cosx.apply(mf)
        mf.kernel()

        def grad(mode, cgr=None):
            g = rhf_fast.Gradients(mf)
            g.grid_response = mode
            if cgr is not None:
                g.cosx_grid_response = cgr
            return g.kernel()
        xc_only = grad(True, cgr=False)
        full = grad(True)
        mf._cosx['last_gmax'] = 1.                      # far from convergence
        self.assertLess(abs(grad('auto') - xc_only).max(), 1e-10)
        mf._cosx['last_gmax'] = .1 * rhf_fast.COSX_GRID_RESPONSE_GMAX
        self.assertLess(abs(grad('auto') - full).max(), 1e-10)
        mf._cosx['last_gmax'] = None                    # no previous gradient: XC only
        self.assertLess(abs(grad('auto') - xc_only).max(), 1e-10)
        self.assertGreater(abs(full - xc_only).max(), 1e-8)


if __name__ == '__main__':
    unittest.main()
