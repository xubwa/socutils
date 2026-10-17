#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Realification of Kramers-restricted orbitals (K = G T, socutils.symm.spinor_basis.realify_mo).

1. K is an antiunitary involution preserving the overlap; no operation G
   for a C2h molecule.
2. Kramers-restricted X2C (2c) and Dirac (4c) orbitals are K invariant, and
   the MO Fock matrix, the X2CAMF one-electron Hamiltonian and the MO
   two-electron integrals are real; D2h, C2v, Coov, Dooh.
3. The energies equal the C1 run (Kramers restricted and unrestricted).
'''

import unittest
import numpy
from pyscf import gto, scf, lib

from socutils.lib import zquatev
from socutils.symm import spinor_basis as sb
from socutils.scf import spinor_hf
from socutils.scf import double_group_symm as dgs

# pyscf.scf.dhf.RDHF imports the pip zquatev, which fails with numpy 2
scf.dhf.zquatev = zquatev

GEOMETRIES = {
    'C2H4': ('C 0 0 .67; C 0 0 -.67; H 0 .92 1.23; H 0 -.92 1.23; H 0 .92 -1.23; H 0 -.92 -1.23', 'C2y'),
    'H2O' : ('O 0 0 0; H 0 .757 .587; H 0 -.757 .587', 'sxz'),
    'HBr' : ('H 0 0 0; Br 0 0 1.41', 'sxz'),
    'Cl2' : ('Cl 0 0 0; Cl 0 0 1.99', 'sxz'),
}


def _mol(geom, basis='6-31g'):
    return sb.to_symm_frame(gto.M(atom=geom, basis=basis, symmetry=True, verbose=0))


def _run(mf):
    mf.verbose = 0
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    return mf


def _unbarred(mf):
    '''The first orbital of each Kramers pair (interleaved (u, Tu) layout).'''
    c = numpy.asarray(mf.mo_coeff)
    return c[:, 0::2]


class KnownValues(unittest.TestCase):
    def test_realification_op(self):
        for name, (geom, op) in GEOMETRIES.items():
            mol = _mol(geom)
            self.assertEqual(sb.realification_op(mol, sb._default_group(mol)), op)
            # K is an antiunitary involution on the spinor AOs that preserves S
            c = numpy.random.default_rng(1).normal(size=(mol.nao_2c(), 3)) * (1 + .5j)
            self.assertAlmostEqual(abs(sb.apply_k(mol, op, sb.apply_k(mol, op, c)) - c).max(), 0, 12)
            s = mol.intor('int1e_ovlp_spinor')
            kc = sb.apply_k(mol, op, c)
            self.assertAlmostEqual(abs(kc.conj().T.dot(s).dot(kc) - c.T.dot(s.conj()).dot(c.conj())).max(), 0, 12)
        # planar trans-CHF=CHF: C2h, no extra operation
        mol = _mol('C 0 .66 0; C 0 -.66 0; F 1.1 1.4 0; F -1.1 -1.4 0; H -.9 1.2 0; H .9 -1.2 0')
        self.assertEqual(mol.groupname, 'C2h')
        self.assertIsNone(sb.realification_op(mol, 'C2h'))

    def test_x2camf_rhf_real(self):
        for name, (geom, op) in GEOMETRIES.items():
            mol = _mol(geom)
            mf = _run(dgs.SymAdaptedRHF(mol).x2camf())
            c = numpy.asarray(mf.mo_coeff)
            u = _unbarred(mf)
            self.assertAlmostEqual(abs(sb.apply_k(mol, op, u) - u).max(), 0, 9, name)
            fock = c.conj().T.dot(mf.get_fock()).dot(c)
            hcore = c.conj().T.dot(mf.get_hcore()).dot(c)
            self.assertAlmostEqual(abs(fock.imag).max(), 0, 10, name)
            self.assertAlmostEqual(abs(hcore.imag).max(), 0, 10, name)
            # a few occupied/virtual orbitals of each kind, all four indices
            idx = numpy.r_[mf.mo_occ.nonzero()[0][-6:], (mf.mo_occ == 0).nonzero()[0][:6]]
            eri = mol.ao2mo(c[:, idx], intor='int2e_spinor')
            self.assertAlmostEqual(abs(eri.imag).max(), 0, 10, name)
            e_c1 = _run(spinor_hf.RHF(mol).x2camf()).e_tot
            self.assertAlmostEqual(mf.e_tot, e_c1, 9, name)

    def test_uhf_unchanged(self):
        geom = GEOMETRIES['H2O'][0]
        mol = _mol(geom)
        mol.charge, mol.spin = 1, 1
        mol.build()
        e_sym = _run(dgs.SymAdaptedUHF(mol).x2camf()).e_tot
        e_c1 = _run(spinor_hf.UHF(mol).x2camf()).e_tot
        self.assertAlmostEqual(e_sym, e_c1, 9)

    def test_rdhf_real(self):
        for name in ('H2O', 'HBr'):
            geom, op = GEOMETRIES[name]
            mol = _mol(geom, 'sto-3g')
            mf = _run(dgs.SymAdaptedRDHF(mol))
            n2c = mol.nao_2c()
            c = numpy.asarray(mf.mo_coeff)
            fock = c.conj().T.dot(mf.get_fock()).dot(c)
            self.assertAlmostEqual(abs(fock.imag).max(), 0, 10, name)
            u = _unbarred(mf)
            ku = numpy.vstack((sb.apply_k(mol, op, u[:n2c]), sb.apply_k(mol, op, u[n2c:])))
            self.assertAlmostEqual(abs(ku - u).max(), 0, 9, name)
            e_c1 = _run(scf.dhf.RDHF(mol)).e_tot
            self.assertAlmostEqual(mf.e_tot, e_c1, 9, name)


if __name__ == '__main__':
    print('Realification of Kramers-restricted orbitals')
    unittest.main()
