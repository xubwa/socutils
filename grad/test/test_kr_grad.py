#!/usr/bin/env python
'''
Kramers-restricted two-component gradient through the scalar fast engines
(socutils.grad.kr_fast) against the GHF path (the same integrals) and finite
differences.

    python grad/test/test_kr_grad.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.dft import dft as sdft
from socutils.grad import kr_fast
from socutils.hessian import rhf_fast as hess_fast
import socutils.grad.ghf_grad  # noqa: F401
import socutils.grad.gks_grad  # noqa: F401

GEOM = 'H 0 0 0; Br 0.3 0 1.414'


def _mol(z=0., basis='cc-pvdz'):
    return gto.M(atom='H 0 0 %.8f; Br 0.3 0 1.414' % z, basis=basis, verbose=0)


def _scf(mol, xc=None, rijcosx=None):
    if xc is None:
        mf = spinor_hf.RHF(mol)
    else:
        mf = sdft.RKS(mol, xc=xc)
        mf.grids.atom_grid = (50, 194)
    mf = mf.x2camf()
    if rijcosx:
        mf.rijcosx = rijcosx
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    return mf.run()


def _both(mf, grid_response=None):
    out = []
    for fast in (True, False):
        kr_fast.ENABLE = fast
        try:
            g = mf.Gradients()
            if grid_response is not None:
                g.grid_response = grid_response
            out.append(g.kernel())
        finally:
            kr_fast.ENABLE = True
    return out


class KnownValues(unittest.TestCase):
    def test_pieces_energy(self):
        '''E_x of the GHF density = -sum_X E_K[2X] over the four real pieces (the kernels' energies)'''
        mol = _mol()
        mf = _scf(mol)
        from socutils.scf import spinor_ghf
        g = spinor_ghf.to_ghf(mf)
        dm = g.make_rdm1()
        R, A, P, Q = kr_fast.pieces(dm, mol.nao)
        self.assertGreater(abs(A).max() + abs(P).max() + abs(Q).max(), 1e-4)     # spin-orbit coupled
        vj, vk = g.get_jk(mol, dm)
        ex = -.5 * numpy.einsum('ij,ji', dm, vk).real
        e = sum(hess_fast.grad2e_jk_cart(mol, 2 * X, [(0., 1.)], mode=1, tol=1e-14)[0] for X in (R, A, P, Q))
        self.assertAlmostEqual(ex, -e, 9)
        ej = .5 * numpy.einsum('ij,ji', dm, vj).real
        self.assertAlmostEqual(ej, hess_fast.grad2e_jk_cart(mol, 2 * R, [(1., 0.)], mode=1, tol=1e-14)[0], 9)

    def test_multi_density_kernel(self):
        '''one density per group (SOCVHFderiv2e_set_dm_stride) = one call per density'''
        mol = gto.M(atom='O 0 0 0.1; H 0 0.76 -0.5; H 0.1 -0.75 -0.45', basis='cc-pvdz', verbose=0)
        rng = numpy.random.default_rng(3)
        x = rng.standard_normal((4, mol.nao, mol.nao)) * .1
        dms = [x[0] + x[0].T, x[1] - x[1].T, x[2] - x[2].T, x[3] - x[3].T]
        groups = [(1., -.7), (0., -.7), (0., -.7), (0., -.7)]
        multi = getattr(hess_fast, 'grad2e_jk_cart_multi', None)
        g = None if multi is None else multi(mol, dms, groups, tol=1e-14)
        if g is None:
            self.skipTest('libsocvhf without per-group densities')
        for d, grp, gi in zip(dms, groups, g):
            ref = hess_fast.grad2e_jk_cart(mol, d, [grp], tol=1e-14)[0]
            self.assertAlmostEqual(abs(gi - ref).max(), 0, 11)

    def test_vs_ghf(self):
        mol = _mol()
        for xc in (None, 'pbe', 'pbe0', 'tpss'):
            mf = _scf(mol, xc)
            for gr in ((None,) if xc is None else (False, True)):
                gf, gg = _both(mf, gr)
                self.assertAlmostEqual(abs(gf - gg).max(), 0, 9, msg='%s grid_response %s' % (xc, gr))

    def test_finite_difference(self):
        h = 1e-3
        for xc in (None, 'pbe0'):
            mf = _scf(_mol(), xc)
            g = mf.Gradients()
            if xc:
                g.grid_response = True
            gz = g.kernel()[0, 2]
            e = [_scf(_mol(s * h), xc).e_tot for s in (2, 1, -1, -2)]
            fd = (-e[0] + 8 * e[1] - 8 * e[2] + e[3]) / (12 * h) * 0.52917721092
            self.assertAlmostEqual(gz, fd, 6, msg=str(xc))

    def test_rij_finite_difference(self):
        '''RIJ (the SCF's auxiliary basis): the RI-J gradient with the auxiliary response, variational'''
        h = 1e-3
        mf = _scf(_mol(), 'pbe0', 'rij')
        gz = mf.Gradients().kernel()[0, 2]
        e = [_scf(_mol(s * h), 'pbe0', 'rij').e_tot for s in (2, 1, -1, -2)]
        fd = (-e[0] + 8 * e[1] - 8 * e[2] + e[3]) / (12 * h) * 0.52917721092
        self.assertAlmostEqual(gz, fd, 5)

    def test_cosx(self):
        '''RIJCOSX: the COSX K gradient of the four pieces on the final grid, within the grid error of the
        exact-K gradient at the same density'''
        mf = _scf(_mol(), 'pbe0', 'rijcosx')
        g1 = mf.Gradients().kernel()
        mf.rijcosx = 'rij'
        g0 = mf.Gradients().kernel()
        self.assertLess(abs(g1 - g0).max(), 2e-4)


if __name__ == '__main__':
    print('Full tests for the Kramers-restricted fast gradient')
    unittest.main()
