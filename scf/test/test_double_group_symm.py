#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Tests for double-group symmetry in the spinor (2c) and Dirac (4c) SCF
(socutils.scf.double_group_symm).

1. Energies equal the C1 calculation (same molecule, same frame) for the
   Kramers-unrestricted and -restricted 2c SCF with the X2C one-electron
   Hamiltonian, and for Kramers-restricted DHF; C2, C2h, Cs, Coov, Dooh, Ci.
2. Orbitals are symmetry pure (orbsym from the coefficients equals the tag),
   Kramers partners are degenerate, occupied orbitals come first.
3. irrep_nelec: fixing the ground-state occupation gives the same energy; a
   different occupation gives a higher energy with exactly that occupation;
   Kramers partners get the same count automatically.
4. A molecule outside its symmetry frame is refused.
'''

import unittest
import numpy
from scipy.spatial.transform import Rotation
from pyscf import gto, scf, symm
from pyscf.x2c import x2c

from socutils.lib import zquatev
from socutils.symm import spinor_basis as sb
from socutils.symm import double_group as dg
from socutils.scf import spinor_hf
from socutils.scf import double_group_symm as dgs

# pyscf.scf.dhf.RDHF imports the pip zquatev, which fails with numpy 2
scf.dhf.zquatev = zquatev

GEOMETRIES = {
    'H2O' : 'O 0 0 0; H 0 .757 .587; H 0 -.757 .587',
    'C2H4': 'C 0 0 .67; C 0 0 -.67; H 0 .92 1.23; H 0 -.92 1.23; H 0 .92 -1.23; H 0 -.92 -1.23',
    'NH3' : 'N 0 0 .1; H .94 0 -.27; H -.47 .814 -.27; H -.47 -.814 -.27',
    'HBr' : 'H 0 0 0; Br 0 0 1.41',
    'Cl2' : 'Cl 0 0 0; Cl 0 0 1.99',
    'C2H2F2': 'C .7 .2 .1; C -.7 -.2 -.1; F 1.2 1 .3; F -1.2 -1 -.3; H .9 -.9 1.2; H -.9 .9 -1.2',
}


def _mol(name, basis='6-31g', seed=11):
    m0 = gto.M(atom=GEOMETRIES[name], basis='sto-3g', symmetry=True, verbose=0)
    rot = Rotation.random(random_state=seed).as_matrix()
    coords = m0.atom_coords().dot(rot.T)
    mol_lab = gto.M(atom=[(m0.atom_symbol(i), c) for i, c in enumerate(coords)],
                    unit='Bohr', basis=basis, symmetry=True, verbose=0)
    return mol_lab, sb.to_symm_frame(mol_lab)


def _run(cls, mol, x2c_hcore=True, **attrs):
    mf = cls(mol)
    mf.verbose = 0
    mf.conv_tol = 1e-11
    if x2c_hcore:
        mf.with_x2c = x2c.SpinorX2CHelper(mol)
    for k, v in attrs.items():
        setattr(mf, k, v)
    mf.kernel()
    return mf


def setUpModule():
    global MOLS
    MOLS = {name: _mol(name) for name in GEOMETRIES}


def tearDownModule():
    global MOLS
    del MOLS


class KnownValues(unittest.TestCase):
    def test_energy_vs_c1_2c(self):
        for name, (mol_lab, mol) in MOLS.items():
            for ref, cls in ((spinor_hf.SpinorSCF, dgs.SymAdaptedUHF),
                             (spinor_hf.KRHF, dgs.SymAdaptedRHF)):
                e_ref = _run(ref, mol).e_tot
                mf = _run(cls, mol)
                self.assertTrue(mf.converged)
                self.assertAlmostEqual(mf.e_tot, e_ref, 9, (name, cls.__name__))
                self._check_orbitals(mf, kramers=cls is dgs.SymAdaptedRHF)

    def test_energy_vs_c1_rdhf(self):
        for name in ('H2O', 'C2H4', 'NH3'):
            mol = MOLS[name][1]
            e_ref = _run(scf.dhf.RDHF, mol, x2c_hcore=False).e_tot
            mf = _run(dgs.SymAdaptedRDHF, mol, x2c_hcore=False)
            self.assertTrue(mf.converged)
            self.assertAlmostEqual(mf.e_tot, e_ref, 9, name)
            self._check_orbitals(mf, kramers=True, dirac=True)
            # Kramers-unrestricted DHF with symmetry lands on the same state
            mf_u = _run(dgs.SymAdaptedDHF, mol, x2c_hcore=False)
            self.assertAlmostEqual(mf_u.e_tot, e_ref, 9, name)

    def _check_orbitals(self, mf, kramers, dirac=False):
        orbsym = mf.get_orbsym()
        so, irrep_id = mf._dg_symm_orb()
        self.assertTrue(numpy.array_equal(dgs.label_orb_symm(numpy.asarray(mf.mo_coeff), so, irrep_id),
                                          orbsym))
        n = mf.mo_energy.size
        occ = mf.mo_occ > 0
        nneg = n // 2 if dirac else 0
        self.assertTrue(numpy.all(occ[nneg:nneg + mf.mol.nelectron]))
        if kramers:
            # every energy of irrep G appears in conj(G)
            gname = mf.doublegroupname
            for ir in set(orbsym):
                e1 = numpy.sort(mf.mo_energy[orbsym == ir])
                e2 = numpy.sort(mf.mo_energy[orbsym == dg.conj(ir, gname)])
                self.assertAlmostEqual(abs(e1 - e2).max(), 0, 9)

    def test_irrep_nelec(self):
        mol = MOLS['C2H4'][1]
        mf0 = _run(dgs.SymAdaptedRHF, mol)
        so, irrep_id = mf0._dg_symm_orb()
        names = mol.irrep_name_spinor
        nocc0 = dict(zip(names, dgs.irrep_occupation(mf0.get_orbsym(), mf0.mo_occ, irrep_id)))
        self.assertEqual(nocc0, {'1Eg': 4, '2Eg': 4, '1Eu': 4, '2Eu': 4})
        # ground state fixed explicitly; only one irrep of each Kramers pair given
        mf1 = _run(dgs.SymAdaptedRHF, mol, irrep_nelec={'1Eg': 4, '1Eu': 4})
        self.assertAlmostEqual(mf1.e_tot, mf0.e_tot, 9)
        # excited occupation (one Kramers pair moved from u to g)
        mf2 = _run(dgs.SymAdaptedRHF, mol, irrep_nelec={'1Eg': 5, '1Eu': 3})
        nocc2 = dict(zip(names, dgs.irrep_occupation(mf2.get_orbsym(), mf2.mo_occ, irrep_id)))
        self.assertEqual(nocc2, {'1Eg': 5, '2Eg': 5, '1Eu': 3, '2Eu': 3})
        self.assertGreater(mf2.e_tot, mf0.e_tot + .1)
        with self.assertRaises(ValueError):
            _run(dgs.SymAdaptedRHF, mol, irrep_nelec={'1Eg': 4, '2Eg': 3})
        with self.assertRaises(ValueError):
            _run(dgs.SymAdaptedRHF, mol, irrep_nelec={'1E': 4})

    def test_x2camf_df(self):
        # X2CAMF and X2CMP Hamiltonians, density fitting with the Kramers-paired MO exchange.
        # cc-pVDZ: with 6-31G the atomic Dirac-HF of Br in the x2camf C backend does not
        # converge and the process aborts (independent of symmetry).
        mol = _mol('HBr', basis='cc-pvdz')[1]
        for flavor in ('x2camf', 'x2cmp'):
            es = []
            for cls in (spinor_hf.KRHF, dgs.SymAdaptedRHF):
                mf = getattr(cls(mol), flavor)()
                mf.verbose = 0
                mf.conv_tol = 1e-10
                es.append(mf.kernel())
            self.assertAlmostEqual(es[0], es[1], 8, flavor)
        es = []
        for cls in (spinor_hf.KRHF, dgs.SymAdaptedRHF):
            mf = cls(mol).x2camf().density_fit(auxbasis='def2-universal-jkfit')
            mf.verbose = 0
            mf.conv_tol = 1e-10
            es.append(mf.kernel())
            # the eig output keeps Kramers pairs interleaved (u, Tu): fast DF exchange applies
            e, c = mf.eig(mf.get_fock(), mf.get_ovlp())
            occ = mf.get_occ(e, c) > 0
            c2 = numpy.vstack(mol.sph2spinor_coeff())
            self.assertIsNot(spinor_hf._kramers_unbarred(c2.dot(numpy.asarray(c)[:, occ]), mol.nao_nr()),
                             False)
        self.assertAlmostEqual(es[0], es[1], 9)

    def test_frame_refused(self):
        mol_lab = MOLS['H2O'][0]
        with self.assertRaises(symm.PointGroupSymmetryError):
            _run(dgs.SymAdaptedRHF, mol_lab)


if __name__ == '__main__':
    print('Tests for socutils.scf.double_group_symm')
    unittest.main()
