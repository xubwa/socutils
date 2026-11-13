#!/usr/bin/env python
'''
Non-collinear (ncol) and multi-collinear (mcol) two-component DFT:
socutils.dft.noncollinear.

    python dft/test/test_noncollinear.py
'''

import unittest
import numpy as np
from pyscf import gto
from pyscf import dft as pyscf_dft
from socutils.dft import gks, noncollinear
from socutils.dft import dft as spinor_dft

try:
    import mcfun  # noqa: F401
    HAVE_MCFUN = True
except ImportError:
    HAVE_MCFUN = False


def _rand_rho(nvar, ng=30, pol=.5, seed=1):
    rng = np.random.RandomState(seed)
    r = np.empty((4, nvar, ng))
    r[0, 0] = rng.rand(ng) * 2 + .1
    # |m| / rho uniform in [0, pol): pol close to 1 reaches the nearly fully
    # polarized (principal-axis) points of mcol
    d = rng.randn(3, ng)
    r[1:, 0] = d / np.linalg.norm(d, axis=0) * rng.rand(ng) * pol * r[0, 0]
    if pol > .99:
        # a few points beyond mcol's collinear_thrd (principal-axis branch)
        r[1:, 0, :4] *= .995 * r[0, 0, :4] / np.linalg.norm(r[1:, 0, :4], axis=0)
    if nvar > 1:
        r[0, 1:4] = rng.randn(3, ng) * .3
        r[1:, 1:4] = rng.randn(3, 3, ng) * .1
    if nvar == 5:
        # tau well above the von Weizsacker bound of each spin channel (libxc clips it)
        r[0, 4] = rng.rand(ng) + np.einsum('xg,xg->g', r[0, 1:4], r[0, 1:4]) / (8 * r[0, 0]) + 2.
        r[1:, 4] = (rng.rand(3, ng) - .5) * .05
    return r


CASES = [('lda,vwn', 'LDA', 1), ('pbe', 'GGA', 4), ('tpss', 'MGGA', 5)]


def _sl(nvar):
    return (lambda r: r[:, 0]) if nvar == 1 else (lambda r: r)


def setUpModule():
    global mol, dm0
    mol = gto.M(atom='O 0 0 0; O 0 0 1.21', basis='6-31g', spin=2, verbose=0)
    dma, dmb = pyscf_dft.UKS(mol, xc='pbe').run().make_rdm1()
    nao = mol.nao
    dm0 = np.zeros((2 * nao, 2 * nao), dtype=complex)
    dm0[:nao, :nao] = dma
    dm0[nao:, nao:] = dmb
    # tilt the spin and add a non-collinear spin texture
    rng = np.random.RandomState(7)
    x = rng.randn(nao, nao) * .02 + 1j * rng.randn(nao, nao) * .02
    dm0[:nao, nao:] = x
    dm0[nao:, :nao] = x.conj().T


def tearDownModule():
    global mol, dm0
    del mol, dm0


class KnownValues(unittest.TestCase):
    def test_ncol_derivatives(self):
        '''ncol vxc and fxc are the analytic derivatives of exc.'''
        ni = noncollinear.NumInt2C()
        ni.collinear = 'ncol'
        h = 1e-6
        for xc, xctype, nvar in CASES:
            rho = _rand_rho(nvar)
            sl = _sl(nvar)
            e, v, f = ni.noncol_eval_xc(xc, sl(rho), deriv=2, xctype=xctype)[:3]
            for c in range(4):
                for x in range(nvar):
                    rp = rho.copy()
                    rm = rho.copy()
                    rp[c, x] += h
                    rm[c, x] -= h
                    ep, vp = ni.noncol_eval_xc(xc, sl(rp), deriv=1, xctype=xctype)[:2]
                    em, vm = ni.noncol_eval_xc(xc, sl(rm), deriv=1, xctype=xctype)[:2]
                    vfd = (ep * rp[0, 0] - em * rm[0, 0]) / (2 * h)
                    ffd = (vp - vm) / (2 * h)
                    self.assertAlmostEqual(abs(v[c, x] - vfd).max() / (1 + abs(vfd).max()), 0, 5)
                    self.assertAlmostEqual(abs(f[:, :, c, x] - ffd).max() / (1 + abs(ffd).max()), 0, 5)

    def test_mcol_lda_is_ncol_lda(self):
        ni = noncollinear.NumInt2C()
        rho = _rand_rho(1)[:, 0]
        ni.collinear = 'ncol'
        r1 = ni.noncol_eval_xc('lda,vwn', rho, deriv=2)
        ni.collinear = 'mcol'
        r2 = ni.noncol_eval_xc('lda,vwn', rho, deriv=2)
        for a, b in zip(r1[:3], r2[:3]):
            self.assertAlmostEqual(abs(a - b).max(), 0, 12)

    def test_unpolarized_kernel(self):
        '''At m = 0 the ncol kernel is the isotropic (= mcol) limit.'''
        ni = noncollinear.NumInt2C()
        for xc, xctype, nvar in CASES:
            rho = _rand_rho(nvar)
            rho[1:] = 0
            ni.collinear = 'ncol'
            fn = ni.noncol_eval_xc(xc, _sl(nvar)(rho), deriv=2, xctype=xctype)[2]
            ni.collinear = 'mcol'
            fm = ni.noncol_eval_xc(xc, _sl(nvar)(rho), deriv=2, xctype=xctype)[2]
            self.assertAlmostEqual(abs(fn - fm).max(), 0, 10)

    def test_mcol_c_kernel(self):
        if noncollinear._load_libsocdft() is None:
            self.skipTest('lib/libsocdft.so not built')
        ni = noncollinear.NumInt2C()
        ni.collinear = 'mcol'
        ni.spin_samples = 26
        for xc, xctype, nvar in CASES[1:]:
            # pol=.999 puts some points beyond collinear_thrd (principal axis)
            rho = _rand_rho(nvar, pol=.999)
            ni.mcol_kernel = 'c'
            r1 = ni.noncol_eval_xc(xc, rho, deriv=2, xctype=xctype)
            ni.mcol_kernel = 'numpy'
            r2 = ni.noncol_eval_xc(xc, rho, deriv=2, xctype=xctype)
            for a, b in zip(r1[:3], r2[:3]):
                self.assertAlmostEqual(abs(a - b).max(), 0, 10)

    @unittest.skipIf(not HAVE_MCFUN, 'mcfun not installed')
    def test_mcol_vs_mcfun(self):
        # Sphere quadrature only: mcfun's principal-axis branch for nearly
        # fully polarized points takes |grad m_k| row norms and loses the sign
        # of grad s (see test_mcol_paxis_exact for ours).
        ni = noncollinear.NumInt2C()
        ni.collinear = 'mcol'
        ni.spin_samples = 26
        ni.collinear_thrd = None
        for xc, xctype, nvar in CASES[1:]:
            rho = _rand_rho(nvar, pol=.999)
            ni.mcol_backend = 'native'
            r1 = ni.noncol_eval_xc(xc, rho, deriv=2, xctype=xctype)
            ni.mcol_backend = 'mcfun'
            r2 = ni.noncol_eval_xc(xc, rho, deriv=2, xctype=xctype)
            for a, b in zip(r1[:3], r2[:3]):
                self.assertAlmostEqual(abs(a - b).max() / (1 + abs(b).max()), 0, 10)

    @unittest.skipIf(not HAVE_MCFUN, 'mcfun not installed')
    def test_mcol_lda_exact(self):
        '''mcfun's sampled LDA converges to the closed form.'''
        ni = noncollinear.NumInt2C()
        ni.collinear = 'mcol'
        rho = _rand_rho(1, pol=.999)[:, 0]
        ref = ni.noncol_eval_xc('lda,vwn', rho, deriv=2)
        ni.mcol_backend = 'mcfun'
        err = []
        for ss in (26, 302, 1202):
            ni.spin_samples = ss
            r = ni.noncol_eval_xc('lda,vwn', rho, deriv=2)
            err.append([abs(np.asarray(b).reshape(np.shape(a)) - a).max()
                        for a, b in zip(ref[:3], r[:3])])
        err = np.array(err)
        self.assertTrue(np.all(err[1:] < err[:-1]))
        self.assertAlmostEqual(err[-1, 0], 0, 6)

    def test_mcol_paxis_exact(self):
        '''For exactly collinear spin textures the principal-axis evaluation
        equals the converged sphere average.'''
        rng = np.random.RandomState(0)
        ng = 20
        ni = noncollinear.NumInt2C()
        ni.collinear = 'mcol'
        u, w = noncollinear._half_sphere(5810)
        for xc, xctype, nvar in CASES[1:]:
            r = _rand_rho(nvar, ng)
            n = rng.randn(3, ng)
            n /= np.linalg.norm(n, axis=0)
            sv = rng.randn(nvar, ng) * .1
            sv[0] = rng.rand(ng) * .6 * r[0, 0]
            r[1:] = n[:, None] * sv
            ref = noncollinear._mcol_sphere(ni, xc, r, 2, xctype, u, w)
            pax = noncollinear._mcol_paxis(ni, xc, r, 2, xctype, 200)
            for a, b in zip(pax[:3], ref[:3]):
                self.assertAlmostEqual(abs(a - b).max(), 0, 10)

    def test_fxc_matrix_fd(self):
        '''fxc response matrix = finite difference of the Vxc matrix.'''
        mf = gks.GKS(mol)
        mf.grids.atom_grid = (30, 86)
        mf.grids.build()
        ni = mf._numint
        nso = dm0.shape[0]
        rng = np.random.RandomState(3)
        dd = rng.randn(nso, nso) * 1e-2 + 1j * rng.randn(nso, nso) * 1e-2
        dd = dd + dd.conj().T
        h = 1e-4
        for col in ('ncol', 'mcol'):
            ni.collinear = col
            ni.spin_samples = 14
            for xc in ('lda,vwn', 'pbe', 'tpss'):
                vp = ni.get_vxc(mol, mf.grids, xc, dm0 + h * dd)[2]
                vm = ni.get_vxc(mol, mf.grids, xc, dm0 - h * dd)[2]
                fxc = ni.get_fxc(mol, mf.grids, xc, dm0, dd, hermi=0)
                ref = (vp - vm) / (2 * h)
                self.assertAlmostEqual(abs(fxc - ref).max() / abs(ref).max(), 0, 5,
                                       msg=f'{col} {xc}')

    def test_scf_energies(self):
        mf = gks.GKS(mol, xc='lda,vwn')
        mf.collinear = 'ncol'
        e_ncol = mf.kernel(dm0=dm0)
        ref = pyscf_dft.GKS(mol, xc='lda,vwn').set(collinear='ncol').run(dm0).e_tot
        self.assertAlmostEqual(e_ncol, ref, 8)
        mf.collinear = 'mcol'
        self.assertAlmostEqual(mf.kernel(dm0=dm0), ref, 8)

        mf = gks.GKS(mol, xc='pbe')
        mf.collinear = 'mcol'
        mf.spin_samples = 14
        e_g = mf.kernel(dm0=dm0)
        ms = spinor_dft.SpinorDFT(mol, xc='pbe')
        ms._numint.collinear = 'mcol'
        ms._numint.spin_samples = 14
        U = ms._numint.U
        e_s = ms.kernel(dm0=U.conj().T @ dm0 @ U)
        self.assertAlmostEqual(e_g, e_s, 8)


if __name__ == '__main__':
    print('Full tests for socutils.dft.noncollinear')
    unittest.main()
