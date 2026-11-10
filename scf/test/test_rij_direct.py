#!/usr/bin/env python
'''
socutils.scf.rij_direct: the direct RI-J passes (rho_P = (P|ij) D_ij, J_ij = (ij|P) d_P) against
pyscf's three-centre integrals, the far field included (a molecule spread over 20 Angstrom), and the
RIJ SCF of socutils.scf.md_j with the direct engine against pyscf's density-fitted J.

    python scf/test/test_rij_direct.py
'''

import unittest
import numpy
import socutils  # noqa: F401
from pyscf import gto, df, dft
from socutils.scf import rij_direct, md_j

# a chain spread over ~20 Angstrom: most auxiliary primitives in the far field of most pairs
CHAIN = '''O 0 0 0; H .757 0 .586; H -.757 0 .586
C 4 0 0; H 4.63 .63 .63; H 4.63 -.63 -.63; H 3.37 .63 -.63; H 3.37 -.63 .63
Cl 9 1 0; H 9 2.27 0
N 14 0 1; H 14.5 .8 1.3; H 14.5 -.8 1.3; H 13.9 0 .0
S 19 0 0; H 19.9 .9 0; H 18.1 .9 0'''


@unittest.skipUnless(rij_direct._load() is not None, 'libsocvhf without rij_direct')
class KnownValues(unittest.TestCase):
    def test_passes(self):
        mol = gto.M(atom=CHAIN, basis='cc-pvdz', verbose=0)
        aux = df.addons.make_auxmol(mol, 'def2-universal-jfit')
        eng = rij_direct.RIJDirect(mol, aux)
        int3c = df.incore.aux_e2(mol, aux, 'int3c2e', aosym='s1')
        rng = numpy.random.default_rng(3)
        c = rng.standard_normal((mol.nao, 12))
        dm = c.dot(c.T)
        rho = eng.get_rho(dm)
        self.assertAlmostEqual(abs(rho - numpy.einsum('ijP,ij->P', int3c, dm)).max(), 0, 9)
        d = rng.standard_normal(aux.nao)
        vj = eng.get_vj_fit(d)
        self.assertAlmostEqual(abs(vj - numpy.einsum('ijP,P->ij', int3c, d)).max(), 0, 9)
        # J = (ij|P) M^-1 (P|kl) D_kl
        j2c = aux.intor('int2c2e')
        ref = numpy.einsum('ijP,P->ij', int3c, numpy.linalg.solve(j2c, numpy.einsum('ijP,ij->P', int3c, dm)))
        self.assertAlmostEqual(abs(eng.get_j(dm) - ref).max(), 0, 9)

    def test_tz(self):
        # f functions in the bra (f f pairs, degree 6) and g on the auxiliary side
        mol = gto.M(atom='O 0 0 0; H .757 0 .586; H -.757 0 .586; Br 6 0 1; H 7.4 0 1', basis='cc-pvtz', verbose=0)
        aux = df.addons.make_auxmol(mol, 'def2-universal-jfit')
        self.assertTrue(rij_direct.available(mol, aux))
        eng = rij_direct.RIJDirect(mol, aux)
        int3c = df.incore.aux_e2(mol, aux, 'int3c2e', aosym='s1')
        rng = numpy.random.default_rng(5)
        c = rng.standard_normal((mol.nao, 8))
        dm = c.dot(c.T)
        ref = numpy.einsum('ijP,ij->P', int3c, dm)
        self.assertAlmostEqual(abs(eng.get_rho(dm) - ref).max() / abs(ref).max(), 0, 12)
        d = rng.standard_normal(aux.nao)
        ref = numpy.einsum('ijP,P->ij', int3c, d)
        self.assertAlmostEqual(abs(eng.get_vj_fit(d) - ref).max() / abs(ref).max(), 0, 12)

    def test_grad(self):
        # the RIJ gradient against the libcint three-centre pass of hessian.rhf_fast.ri_j_grad and FD
        from socutils.hessian import rhf_fast as hess_fast
        mol = gto.M(atom=CHAIN, basis='cc-pvdz', verbose=0)
        aux = df.addons.make_auxmol(mol, 'def2-universal-jfit')
        rng = numpy.random.default_rng(7)
        c = rng.standard_normal((mol.nao, 10))
        dm = c.dot(c.T) * .1
        g = rij_direct.RIJDirect(mol, aux).grad(dm)
        old = hess_fast.RIJ_GRAD_DIRECT
        hess_fast.RIJ_GRAD_DIRECT = False
        try:
            ref = hess_fast.ri_j_grad(mol, aux, dm)
        finally:
            hess_fast.RIJ_GRAD_DIRECT = old
        if ref is not None:
            self.assertAlmostEqual(abs(g - ref).max(), 0, 9)

        def ej(m):
            e = rij_direct.RIJDirect(m, df.addons.make_auxmol(m, 'def2-universal-jfit'))
            rho = e.get_rho(dm)
            return .5 * rho.dot(e.fit(rho))
        h = 1e-4
        for ia, x in ((0, 2), (8, 0), (13, 1)):
            co = mol.atom_coords().copy()
            co[ia, x] += h
            ep = ej(mol.set_geom_(co, unit='bohr', inplace=False))
            co[ia, x] -= 2 * h
            em = ej(mol.set_geom_(co, unit='bohr', inplace=False))
            self.assertAlmostEqual((ep - em) / (2 * h), g[ia, x], 6)

    def test_rij_scf(self):
        mol = gto.M(atom=CHAIN, basis='cc-pvdz', verbose=0)
        ref = dft.RKS(mol, xc='pbe').density_fit(auxbasis='def2-universal-jfit', only_dfj=True)
        ref.conv_tol = 1e-10
        e_ref = ref.kernel()
        mf = dft.RKS(mol, xc='pbe').density_fit(auxbasis='def2-universal-jfit', only_dfj=True)
        mf.conv_tol = 1e-10
        old = md_j.RIJ_DIRECT
        md_j.RIJ_DIRECT = True
        try:
            md_j.apply(mf)
            # the first build already from the direct engine
            dm = mf.get_init_guess()
            vj = mf.get_j(mol, dm)
            self.assertTrue(mf._rij_cache['last'] is not None)
            self.assertAlmostEqual(abs(vj - ref.get_j(mol, dm)).max(), 0, 7)     # loose first pass
            mf._rij_cache['last'] = None
            e = mf.kernel()
            self.assertTrue(mf.converged and mf._rij_cache['direct'] is not None)
        finally:
            md_j.RIJ_DIRECT = old
        self.assertAlmostEqual(e, e_ref, 8)


if __name__ == '__main__':
    unittest.main()
