#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Tests for the double-group symmetry-adapted spinor basis (socutils.symm.spinor_basis).

For molecules covering every supported double group, each given in a random
orientation and moved with to_symm_frame:

1. The SALCs form a unitary matrix (orthonormal and complete).
2. S, T+V, sigma.p V sigma.p and the X2C one-electron Hamiltonian are block
   diagonal by irrep; so are the 4c DHF overlap and core Hamiltonian in the
   dirac_symm_orb basis.
3. Kramers structure: the partner block of conj(Gamma) is T applied to the
   Gamma block, so its matrices are the complex conjugates; self-conjugate
   blocks (C1, Ci) pass zquatev's Kramers-structure check.
4. Labels: for an atom and a linear molecule each SALC has n = 2 m_j of its AOs.
5. Frame handling: a rotated molecule with an axis is refused with a message
   that contains the symmetrized coordinates; C1/Ci molecules are accepted in
   any orientation; to_symm_frame keeps the geometry up to a proper rotation.
'''

import unittest
import numpy
from scipy.spatial.transform import Rotation
from pyscf import gto, scf, symm
from pyscf.x2c import x2c

from socutils.lib import zquatev
from socutils.symm import spinor_basis as sb
from socutils.symm import double_group as dg

GEOMETRIES = {
    'H2O'    : ('C2',   'O 0 0 0; H 0 .757 .587; H 0 -.757 .587'),
    'CH4'    : ('C2',   'C 0 0 0; H .63 .63 .63; H -.63 -.63 .63; H -.63 .63 -.63; H .63 -.63 -.63'),
    'C2H4'   : ('C2h',  'C 0 0 .67; C 0 0 -.67; H 0 .92 1.23; H 0 -.92 1.23; H 0 .92 -1.23; H 0 -.92 -1.23'),
    'N2H2'   : ('C2h',  'N 0 .62 0; N 0 -.62 0; H .95 .9 0; H -.95 -.9 0'),
    'NH3'    : ('Cs',   'N 0 0 .1; H .94 0 -.27; H -.47 .814 -.27; H -.47 -.814 -.27'),
    'HBr'    : ('Coov', 'H 0 0 0; Br 0 0 1.41'),
    'Cl2'    : ('Dooh', 'Cl 0 0 0; Cl 0 0 1.99'),
    'Ne'     : ('Dooh', 'Ne 0 0 0'),
    'CHFClBr': ('C1',   'C 0 0 0; H .6 .6 .6; F -.7 -.7 .7; Cl -.9 .9 -.9; Br 1 -1 -1'),
    'C2H2F2Cl2': ('Ci', 'C .7 .2 .1; C -.7 -.2 -.1; F 1.2 1 .3; F -1.2 -1 -.3; Cl .9 -.9 1.2; Cl -.9 .9 -1.2'),
}


def _rotated(geom, basis, seed):
    m0 = gto.M(atom=geom, basis=basis, symmetry=True, verbose=0)
    rot = Rotation.random(random_state=seed).as_matrix()
    coords = m0.atom_coords().dot(rot.T) + numpy.array([.3, -.2, .5])
    return gto.M(atom=[(m0.atom_symbol(i), c) for i, c in enumerate(coords)],
                 unit='Bohr', basis=basis, symmetry=True, verbose=0)


def _max_off_block(mat, so):
    u = numpy.hstack(so)
    mask = numpy.zeros((u.shape[1],) * 2, dtype=bool)
    i0 = 0
    for x in so:
        mask[i0:i0+x.shape[1], i0:i0+x.shape[1]] = True
        i0 += x.shape[1]
    return abs(numpy.where(mask, 0, u.conj().T.dot(mat).dot(u))).max() / abs(mat).max()


def setUpModule():
    global MOLS
    MOLS = {}
    for k, (name, (gname, geom)) in enumerate(GEOMETRIES.items()):
        mol_lab = _rotated(geom, 'cc-pvdz', seed=k)
        mol = sb.to_symm_frame(mol_lab)
        sb.build(mol)
        MOLS[name] = (gname, mol_lab, mol)


def tearDownModule():
    global MOLS
    del MOLS


class KnownValues(unittest.TestCase):
    def test_groups(self):
        for name, (gname, mol_lab, mol) in MOLS.items():
            self.assertEqual(mol.doublegroupname, gname, name)
            self.assertEqual(mol.irrep_name_spinor,
                             [dg.irrep_id2name(gname, i) for i in mol.irrep_id_spinor])
            self.assertTrue(all(dg.is_fermion(i) for i in mol.irrep_id_spinor))

    def test_unitary(self):
        for name, (gname, mol_lab, mol) in MOLS.items():
            u = numpy.hstack(sb.get_symm_orb(mol)[0])
            self.assertEqual(u.shape, (mol.nao_2c(),) * 2)
            self.assertAlmostEqual(abs(u.conj().T.dot(u) - numpy.eye(u.shape[1])).max(), 0, 12, name)

    def test_block_diagonal_2c(self):
        for name, (gname, mol_lab, mol) in MOLS.items():
            so = sb.get_symm_orb(mol)[0]
            mats = [mol.intor('int1e_ovlp_spinor'),
                    mol.intor('int1e_kin_spinor') + mol.intor('int1e_nuc_spinor'),
                    mol.intor('int1e_spnucsp_spinor'),
                    x2c.SpinorX2CHelper(mol).get_hcore()]
            for mat in mats:
                self.assertLess(_max_off_block(mat, so), 1e-12, name)

    def test_block_diagonal_4c(self):
        for name, (gname, mol_lab, mol) in MOLS.items():
            so4, ids = sb.dirac_symm_orb(mol, sb.get_symm_orb(mol)[0], mol.irrep_id_spinor)
            u = numpy.hstack(so4)
            self.assertAlmostEqual(abs(u.conj().T.dot(u) - numpy.eye(u.shape[1])).max(), 0, 12, name)
            mf = scf.DHF(mol)
            for mat in (mf.get_ovlp(), mf.get_hcore()):
                self.assertLess(_max_off_block(mat, so4), 1e-12, name)

    def test_kramers(self):
        for name, (gname, mol_lab, mol) in MOLS.items():
            blocks = dict(zip(mol.irrep_id_spinor, sb.get_symm_orb(mol)[0]))
            h = mol.intor('int1e_kin_spinor') + mol.intor('int1e_nuc_spinor') + \
                mol.intor('int1e_spnucsp_spinor') * .01
            for ir, x in blocks.items():
                partner = dg.conj(ir, gname)
                hx = x.conj().T.dot(h).dot(x)
                if partner == ir:
                    zquatev.check_kramers_structure(hx, thresh=1e-10)
                    self.assertEqual(x.shape[1] % 2, 0)
                else:
                    y = blocks[partner]
                    # the first block of a pair generates the partner, T^2 = -1 on the way back
                    first = mol.irrep_id_spinor.index(ir) < mol.irrep_id_spinor.index(partner)
                    sign = 1 if first else -1
                    self.assertAlmostEqual(abs(y - sign * sb.time_reversal_coeff(mol, x)).max(), 0, 12)
                    hy = y.conj().T.dot(h).dot(y)
                    self.assertAlmostEqual(abs(hy - hx.conj()).max(), 0, 9, name)

    def test_linear_labels(self):
        for name in ('Ne', 'HBr', 'Cl2'):
            gname, mol_lab, mol = MOLS[name]
            two_mj = numpy.array([int(x[3].split('/')[0]) for x in mol.spinor_labels(fmt=False)])
            for ir, x in zip(mol.irrep_id_spinor, sb.get_symm_orb(mol)[0]):
                support = abs(x).max(axis=1) > 1e-10
                self.assertTrue(numpy.all(two_mj[support] == ir // 2), (name, ir))
        # Ne: parity is (-1)^l of the AOs
        gname, mol_lab, mol = MOLS['Ne']
        l = numpy.array(['spdfg'.index(x[2][-4]) for x in mol.spinor_labels(fmt=False)])
        for ir, x in zip(mol.irrep_id_spinor, sb.get_symm_orb(mol)[0]):
            support = abs(x).max(axis=1) > 1e-10
            self.assertTrue(numpy.all(l[support] % 2 == ir % 2))

    def test_1E_contains_mj_half(self):
        # DIRAC convention: m_j = +1/2 (an alpha-like spinor) belongs to 1E
        for name in ('H2O', 'NH3', 'C2H4'):
            gname, mol_lab, mol = MOLS[name]
            so, ids = sb.get_symm_orb(mol)
            labels = mol.spinor_labels(fmt=False)
            # an s spinor with m_j = +1/2 on an atom on the main axis (or at the centre)
            coords = mol.atom_coords() - numpy.asarray(mol._symm_orig)
            on_axis = [a for a in range(mol.natm) if numpy.linalg.norm(coords[a, :2]) < 1e-6]
            if not on_axis:
                continue
            mu = [i for i, x in enumerate(labels) if x[0] == on_axis[0] and x[2].endswith('s1/2')
                  and x[3] == '1/2'][0]
            weights = {dg.irrep_id2name(gname, ir): (abs(x[mu])**2).sum() for ir, x in zip(ids, so)}
            # C2h: the two C atoms are related by i, the weight splits between 1Eg and 1Eu
            w1e = sum(w for k, w in weights.items() if k.startswith('1E'))
            self.assertAlmostEqual(w1e, 1, 12, (name, weights))

    def test_frame(self):
        for name, (gname, mol_lab, mol) in MOLS.items():
            self.assertTrue(sb.symm_frame_ok(mol))
            # same molecule: interatomic distances preserved, proper rotation
            d0 = numpy.linalg.norm(mol_lab.atom_coords()[:, None] - mol_lab.atom_coords(), axis=2)
            d1 = numpy.linalg.norm(mol.atom_coords()[:, None] - mol.atom_coords(), axis=2)
            self.assertLess(abs(d0 - d1).max(), 1e-5, name)
            orig, axes = map(numpy.asarray, mol.symm_frame)
            mol.dumps()   # chkfile serialization still works
            self.assertAlmostEqual(numpy.linalg.det(axes), 1, 12)
            back = mol.atom_coords().dot(axes) + orig
            self.assertLess(abs(back - mol_lab.atom_coords()).max(), 1e-5, name)
            if gname in ('C1', 'Ci') or name == 'Ne':
                sb.symm_adapted_basis_spinor(mol_lab)   # any orientation is fine
            else:
                with self.assertRaises(symm.PointGroupSymmetryError) as ctx:
                    sb.symm_adapted_basis_spinor(mol_lab)
                msg = str(ctx.exception)
                self.assertIn('to_symm_frame', msg)
                # the printed coordinates are those of to_symm_frame
                printed = numpy.array([[float(v) for v in line.split()[1:]]
                                       for line in msg.splitlines() if line.startswith('  ')])
                self.assertAlmostEqual(abs(printed - mol.atom_coords()).max(), 0, 9)

    def test_symmetrize(self):
        # a slightly distorted water is made exactly C2v symmetric
        mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.7570000001 .587', basis='sto-3g',
                    symmetry=True, verbose=0)
        mol_s = sb.to_symm_frame(mol)
        c = mol_s.atom_coords()
        h1, h2 = c[1], c[2]
        self.assertAlmostEqual(abs(h1[:2] + h2[:2]).max(), 0, 14)
        self.assertAlmostEqual(h1[2] - h2[2], 0, 14)


if __name__ == '__main__':
    print('Tests for socutils.symm.spinor_basis')
    unittest.main()
