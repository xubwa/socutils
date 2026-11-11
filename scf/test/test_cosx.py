#!/usr/bin/env python
'''
socutils.scf.cosx: the COSX kernel (lib/dft/cosx.c) against a numpy
reference from libcint's int1e_grids on the same grid (no screening), the
overlap-fitted K against the exact K, and the SCF hook (RIJ + COSX) against
the exact-K SCF energy.

    python scf/test/test_cosx.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto, scf, dft
from socutils.scf import cosx

# general contractions (Cl), anion, an atom pair with coincident centres
MOL = '''O 0 0 0.1; H 0 0.76 -0.5; H 0 -0.76 -0.5; C 2.5 0.3 0.2; Cl -1.6 1.1 0.7'''
WATER2 = '''
O  0.000000  0.000000  0.000000
H  0.756950  0.000000  0.585882
H -0.756950  0.000000  0.585882
O  0.000000  0.000000  3.000000
H  0.876934  0.133858  3.359583
H -0.280961 -0.841423  3.359583
'''


def reference_k(mol, grids, dm):
    '''K_mu nu = sum_g w phi_mu sum_lam (nu lam|g) sum_kap phi_kap D_kap lam (plain, unsymmetrised)'''
    ao = dft.numint.eval_ao(mol, grids.coords)
    w = grids.weights
    ref = numpy.zeros_like(dm)
    for g0 in range(0, w.size, 400):
        sl = slice(g0, g0 + 400)
        a = mol.intor('int1e_grids', grids=grids.coords[sl])
        f = ao[sl] @ dm
        ref += (ao[sl] * w[sl, None]).T @ numpy.einsum('gnl,gl->gn', a, f)
    return ref


@unittest.skipUnless(cosx.available(gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g')), 'libsocdft without COSX')
class KnownValues(unittest.TestCase):
    def _check_kernel(self, basis):
        mol = gto.M(atom=MOL, basis=basis, charge=-1, verbose=0)
        dm = scf.RHF(mol).run().make_rdm1()
        rng = numpy.random.default_rng(1)
        dm2 = dm + .01 * rng.standard_normal(dm.shape)      # not symmetric
        k = cosx.COSX(mol, cosx.make_grids(mol, **cosx.GRIDX[1]), fit=False)
        k.tol = 0.
        k.ao_cutoff = 0.
        vk = k.get_k(numpy.array([dm, dm2]), hermi=0)
        for v, d in zip(vk, (dm, dm2)):
            ref = reference_k(mol, k.grids, d)
            self.assertLess(abs(v - ref).max(), 1e-11 * abs(ref).max())

    def test_kernel_dz(self):
        self._check_kernel('cc-pvdz')

    def test_kernel_tz(self):
        self._check_kernel('def2-tzvp')        # f shells

    def test_multi_density(self):
        '''the multi-density contraction (nset >= MULTI_MIN) against one density at a time'''
        mol = gto.M(atom=MOL, basis='def2-tzvp', charge=-1, verbose=0)
        rng = numpy.random.default_rng(2)
        dms = rng.standard_normal((6, mol.nao, mol.nao)) * .01
        dms = dms + dms.transpose(0, 2, 1)
        k = cosx.COSX(mol, cosx.make_grids(mol, **cosx.GRIDX[1]))
        k.tol = 0.
        vk = k.get_k(dms)
        ref = numpy.array([k.get_k(d) for d in dms])
        self.assertLess(abs(vk - ref).max(), 1e-12 * abs(ref).max())
        # the record-wise contraction (fallback without dsymm) and the dense one, at the default tol
        k.tol = cosx.TOL
        vk = k.get_k(dms)
        k._libdft.SOCDFTcosx_set_dense(0)
        try:
            self.assertLess(abs(k.get_k(dms) - vk).max(), 1e-14 * abs(vk).max() + 1e-15)
        finally:
            k._libdft.SOCDFTcosx_set_dense(1)
        # every block record-wise after the counting pass (kept fraction below the dense threshold)
        dmin, cosx.DENSE_MIN = cosx.DENSE_MIN, 2.
        try:
            self.assertLess(abs(k.get_k(dms) - vk).max(), 1e-14 * abs(vk).max() + 1e-15)
        finally:
            cosx.DENSE_MIN = dmin
        k.tol = 0.
        if k._libdft.SOCDFTcosx_mfloat is not None:
            cosx.set_multi_float(True)
            try:
                if k._libdft.SOCDFTcosx_mfloat():
                    vkf = k.get_k(dms)
                    self.assertLess(abs(vkf - ref).max(), 1e-5 * abs(ref).max())
            finally:
                cosx.set_multi_float(False)

    def test_fit_vs_exact(self):
        mol = gto.M(atom=WATER2, basis='cc-pvdz', verbose=0)
        mf = scf.RHF(mol).run()
        dm = mf.make_rdm1()
        vk0 = mf.get_k(mol, dm)
        e0 = -.25 * numpy.einsum('ij,ij', dm, vk0)
        for lev, thr in ((2, 2e-4), (3, 1e-4)):
            vk = cosx.COSX(mol, cosx.make_grids(mol, **cosx.GRIDX[lev])).get_k(dm)
            self.assertLess(abs(-.25 * numpy.einsum('ij,ij', dm, vk) - e0), thr)
            self.assertLess(abs(vk - vk.T).max(), 1e-14)

    def test_energy(self):
        mol = gto.M(atom=WATER2, basis='def2-tzvp', verbose=0)
        dm = scf.RHF(mol).run().make_rdm1()
        for fit in (False, True):
            k = cosx.COSX(mol, cosx.make_grids(mol, **cosx.GRIDX[2]), fit=fit)
            k.tol = 1e-13
            e0 = numpy.einsum('ij,ij', dm, k.get_k(dm))
            self.assertAlmostEqual(k.energy(dm, tol=1e-13), e0, 9)
            self.assertAlmostEqual(k.energy(dm, tol=1e-9), e0, 6)

    def test_grad_fixed_density(self):
        '''d/dR tr(D P K_plain[D]) on a frozen grid against finite differences (plain and fitted)'''
        from pyscf import lib
        from pyscf.dft import gen_grid
        mol = gto.M(atom=MOL, basis='def2-svp', charge=-1, verbose=0)
        dm = scf.RHF(mol).run().make_rdm1()
        g0 = cosx.make_grids(mol, **cosx.GRIDX[1])

        class Frozen(lib.StreamObject):
            def __init__(self, m):
                self.coords, self.weights = g0.coords, g0.weights
                self.non0tab = numpy.ones_like(gen_grid.make_mask(m, g0.coords))

        for fit in (False, True):
            def kobj(m):
                k = cosx.COSX(m, Frozen(m), fit=fit)
                k.tol = k.ao_cutoff = 0.
                return k
            ga = kobj(mol).grad(dm)
            h = 1e-4
            for ia, x in ((0, 2), (3, 0), (4, 1)):
                c = mol.atom_coords().copy()
                c[ia, x] += h
                ep = numpy.einsum('ij,ij', dm, kobj(mol.set_geom_(c, unit='Bohr', inplace=False)).get_k(dm, hermi=0))
                c[ia, x] -= 2 * h
                em = numpy.einsum('ij,ij', dm, kobj(mol.set_geom_(c, unit='Bohr', inplace=False)).get_k(dm, hermi=0))
                self.assertAlmostEqual(ga[ia, x], (ep - em) / (2 * h), 6)

    def test_hessian_pieces(self):
        '''Fock1 (d K / dR C) and the K second derivatives against finite differences on a frozen grid, and
        the sum-ruled second derivatives against the exact K ones'''
        from pyscf import lib
        from pyscf.dft import gen_grid
        from socutils.hessian import rhf_fast
        mol = gto.M(atom=WATER2, basis='cc-pvdz', verbose=0)
        mf = scf.RHF(mol).run()
        mocc = mf.mo_coeff[:, mf.mo_occ > 0]
        dm = 2 * mocc @ mocc.T
        g0 = cosx.make_grids(mol, **cosx.GRIDX[1])

        class Frozen(lib.StreamObject):
            def __init__(self, m):
                self.coords, self.weights = g0.coords, g0.weights
                self.non0tab = numpy.ones_like(gen_grid.make_mask(m, g0.coords))

        def kobj(m):
            k = cosx.COSX(m, Frozen(m))
            k.tol = k.ao_cutoff = 0.
            return k
        h, f1 = kobj(mol).hess(mocc, with_fock1=True, sum_rule=False)
        dh = 1e-5
        for ia, x in ((0, 2), (4, 0)):
            c = mol.atom_coords().copy()
            c[ia, x] += dh
            mp = mol.set_geom_(c, unit='Bohr', inplace=False)
            kp, gp = kobj(mp).get_k(dm) @ mocc, kobj(mp).grad(dm)
            c[ia, x] -= 2 * dh
            mm = mol.set_geom_(c, unit='Bohr', inplace=False)
            km, gm = kobj(mm).get_k(dm) @ mocc, kobj(mm).grad(dm)
            self.assertLess(abs(f1[ia, x] - (kp - km) / (2 * dh)).max(), 1e-6)
            self.assertLess(abs(h[ia, :, x] - (gp - gm) / (2 * dh)).max(), 1e-5)
        k = cosx.COSX(mol, cosx.make_grids(mol, **cosx.GRIDX[3]))
        ek = rhf_fast.hess2e_jk_cart(mol, dm, [(0, 1)])[0]
        self.assertLess(abs(k.hess(mocc) / 4 - ek).max(), 1e-4)

    def test_scf(self):
        mol = gto.M(atom=WATER2, basis='cc-pvdz', verbose=0)
        mf0 = dft.RKS(mol, xc='pbe0').density_fit(only_dfj=True)
        e0 = mf0.kernel()
        mf = cosx.apply(dft.RKS(mol, xc='pbe0').density_fit(only_dfj=True))
        e = mf.kernel()
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(e, e0, 4)

    def test_gridx_args_heavy(self):
        '''the Lebedev table of the GRIDX levels is raised for elements beyond Z = 18, by the element and not
        by the (ECP valence) charge'''
        light = gto.M(atom='S 0 0 0; H 0 .96 .92; H 0 -.96 .92', basis='def2-svp', verbose=0)
        for lev in (0, 1, 2, 3):
            self.assertEqual(cosx.gridx_args(light, lev), cosx.GRIDX[lev])
        for atom, ecp in (('Fe 0 0 0', None), ('Br 0 0 0; H 0 0 1.41', None), ('Rb 0 0 0; F 0 0 2.27', 'def2-svp')):
            spin = 4 if atom.startswith('Fe') else 0
            mol = gto.M(atom=atom, basis='def2-svp', ecp=ecp, spin=spin, verbose=0)
            if ecp:
                self.assertLess(mol.atom_charges().max(), 18)       # Rb: 9 valence electrons
            for lev in (0, 1, 2):
                args = cosx.gridx_args(mol, lev)
                self.assertEqual(args['angular_grid'], cosx.HEAVY_ANGULAR)
                self.assertEqual(args['intacc'], cosx.GRIDX[lev]['intacc'])
            self.assertEqual(cosx.gridx_args(mol, 3), cosx.GRIDX[3])

    def test_grid_kept_after_gradient(self):
        '''a gradient / Hessian leaves an atom index in mol._env (with_rinv_at_nucleus): the COSX objects must
        not take that for another molecule and rebuild on the default grid (the final GRIDX3 energy of a
        second kernel() was a GRIDX2 one), and another geometry keeps the object's grid level'''
        mol = gto.M(atom=WATER2, basis='def2-svp', verbose=0)
        mf = cosx.apply(dft.RKS(mol, xc='pbe0').density_fit(auxbasis='def2-universal-jfit', only_dfj=True))
        mf.conv_tol = 1e-10
        e = mf.kernel()
        npts = {lev: k.grids.weights.size for lev, k in mf._cosx['k'].items()}
        self.assertGreater(npts[3], npts[2])
        with mol.with_rinv_at_nucleus(mol.natm - 1):
            pass
        self.assertAlmostEqual(mf.kernel(), e, 8)
        self.assertEqual({lev: k.grids.weights.size for lev, k in mf._cosx['k'].items()}, npts)
        # another geometry: the same level
        k = mf._cosx['k'][3]
        c = mol.atom_coords()
        c[0, 0] += .05
        k.mol = mol.set_geom_(c, unit='Bohr', inplace=False)
        k.build()
        ref = cosx.make_grids(k.mol, **cosx.GRIDX[3])
        self.assertEqual(k.grids.weights.size, ref.weights.size)
        self.assertAlmostEqual(abs(k.grids.coords - ref.coords).max(), 0, 12)


if __name__ == '__main__':
    unittest.main()
