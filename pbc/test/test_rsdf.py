'''Tests of the range-separated Gamma-point density fitting (socutils.pbc.rsdf).

The fitting error is eliminated by construction where possible (an auxiliary
basis that spans the density exactly), so the periodic integrals themselves are
compared with the numerically exact references of coulomb_reference.
'''
import unittest
import numpy as np
from pyscf import gto, scf, df
from pyscf.pbc import gto as pgto
from pyscf.pbc.tools import super_cell
from socutils.pbc import coulomb_reference as cr

try:
    from socutils.pbc.rsdf import RSDF, _libpbcdf
    _libpbcdf()
    HAVE_LIB = True
except OSError:
    HAVE_LIB = False

FCC = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
ZETA = 20.
A_LI, A_H = 3.0, 2.0


def compact_s_cell(shift=(0, 0, 0)):
    '''Li/H rock salt with one compact s function per atom (products between
    lattice images < 1e-20), so a diagonal density is a set of Gaussian charges.'''
    r0 = 4.0
    shift = np.asarray(shift, dtype=float)
    return pgto.M(a=FCC * 2 * r0, unit='B', verbose=0, spin=0,
                  atom=[['Li', shift], ['H', shift + [r0, 0, 0]]],
                  basis={'Li': [[0, [A_LI, 1.]]], 'H': [[0, [A_H, 1.]]]})


# auxiliary basis spanning chi^2 exactly
EXACT_AUX = {'Li': [[0, [2 * A_LI, 1.]]], 'H': [[0, [2 * A_H, 1.]]]}


def coulomb_energy(rsdf, dm):
    vj, _ = rsdf.get_jk(dm, with_k=False)
    return (.5 * np.einsum('ij,ji', dm, vj) + np.einsum('ij,ji', dm, rsdf.get_nuc())
            + rsdf.energy_nuc())


@unittest.skipUnless(HAVE_LIB, 'libsocpbcdf.so not built (make)')
class KnownValues(unittest.TestCase):
    def test_exact_fit_gaussian_charges(self):
        '''Zero fitting error: J, V_ne and E_nn together reproduce the Ewald
        energy of the Gaussian charges, independent of omega.'''
        cell = compact_s_cell()
        dm = np.diag([2., 2.])
        x = cell.atom_coords()
        a = cell.lattice_vectors()
        # Gaussian charges without the nuclear self energies (E_nn excludes them)
        ref = cr.gaussian_charges_recip(a, [x[0], x[0], x[1], x[1]], [3, -2, 1, -2], [ZETA, 6., ZETA, 4.])
        ref -= cr.gaussian_self_energy([3, 1], [ZETA, ZETA])
        for omega in (0.4, 0.8, 1.5):
            r = RSDF(cell, EXACT_AUX, omega=omega, precision=1e-12, nuc_zeta=ZETA).build()
            self.assertAlmostEqual(coulomb_energy(r, dm), ref, 11, msg='omega %g' % omega)

    def test_exact_fit_tight_dense(self):
        '''Core-like functions (exponents 1e4 and 30) and steep nuclei in a dense
        lattice (Li-H 2 bohr): the regime where pyscf's lattice-sum estimators
        fail.  Reference: Ewald sum of the Gaussian charges.'''
        r0, a_li, a_h, zeta = 2.0, 1e4, 30., 1e6
        cell = pgto.M(a=FCC * 2 * r0, unit='B', verbose=0, spin=0,
                      atom=[['Li', (0, 0, 0)], ['H', (r0, 0, 0)]],
                      basis={'Li': [[0, [a_li, 1.]]], 'H': [[0, [a_h, 1.]]]})
        aux = {'Li': [[0, [2 * a_li, 1.]]], 'H': [[0, [2 * a_h, 1.]]]}
        dm = np.diag([2., 2.])
        x = cell.atom_coords()
        ref = cr.gaussian_charges_ewald(cell.lattice_vectors(), [x[0], x[0], x[1], x[1]], [3, -2, 1, -2],
                                        [zeta, 2 * a_li, zeta, 2 * a_h])
        ref -= cr.gaussian_self_energy([3, 1], [zeta, zeta])
        for omega in (0.5, 1.0):
            r = RSDF(cell, aux, omega=omega, precision=1e-12, nuc_zeta=zeta).build()
            self.assertAlmostEqual(coulomb_energy(r, dm), ref, 9, msg='omega %g' % omega)

    def test_exact_fit_matrices(self):
        '''With the exact auxiliary basis, J and V_ne match the reference matrices.'''
        cell = compact_s_cell()
        dm = np.diag([2., 2.])
        ref = cr.coulomb_reference(cell, dm, ZETA, ke_cutoff=1200, rpair=12)
        r = RSDF(cell, EXACT_AUX, omega=0.6, precision=1e-12, nuc_zeta=ZETA).build()
        vj = r.get_jk(dm, with_k=False)[0]
        # both conventions drop G = 0 of every charge-carrying part, so the parts compare
        self.assertLess(abs(vj - ref['vj']).max(), 1e-11)
        self.assertLess(abs(r.get_nuc() - ref['vnuc']).max(), 1e-11)

    def test_translation_and_supercell(self):
        dm = np.diag([2., 2.])
        e0 = coulomb_energy(RSDF(compact_s_cell(), EXACT_AUX, precision=1e-12, nuc_zeta=ZETA).build(), dm)
        e1 = coulomb_energy(RSDF(compact_s_cell([.37, -1.1, .52]), EXACT_AUX, precision=1e-12,
                                 nuc_zeta=ZETA).build(), dm)
        self.assertAlmostEqual(e1, e0, 10)
        sc = super_cell(compact_s_cell(), [2, 1, 1])
        e2 = coulomb_energy(RSDF(sc, EXACT_AUX, precision=1e-12, nuc_zeta=ZETA).build(), np.diag([2.] * 4))
        self.assertAlmostEqual(e2, 2 * e0, 9)

    def test_zero_dimensional_vs_molecular_df(self):
        '''Same auxiliary basis in a box and in a molecule: the fitted Coulomb
        energies approach each other as the box grows (fitting error cancels).'''
        mol = gto.M(atom='H 0 0 -0.7; H 0 0 0.7', unit='B', basis='cc-pvdz', verbose=0)
        dm = scf.RHF(mol).run().make_rdm1()
        aux = df.addons.aug_etb(mol, beta=2.0)
        mdf = df.DF(mol, auxbasis=aux).build()
        m = cr.molecular_coulomb(mol, dm, zeta=ZETA)
        e_mol = .5 * np.einsum('ij,ji', dm, mdf.get_jk(dm, with_k=False)[0]) + m['e_en'] + m['e_nn']
        e_nn_self = cr.gaussian_self_energy(mol.atom_charges(), [ZETA] * 2)
        diffs = []
        for L in (16., 22.):
            r = RSDF(cr.molecule_in_box(mol, L), aux, omega=0.5, precision=1e-11, nuc_zeta=ZETA).build()
            diffs.append(coulomb_energy(r, dm) - (e_mol - e_nn_self))
        # image interaction ~ A/L^5: ratio close to (22/16)^5
        self.assertLess(abs(diffs[1]), 1e-5)
        self.assertLess(abs(diffs[1]), abs(diffs[0]))

    def test_blocked_disk_storage(self):
        '''Many AO row blocks with the vectors on disk (HDF5) reproduce the
        in-memory build, for a real and a complex non-Hermitian density.'''
        mol = gto.M(atom='H 0 0 -0.7; H 0 0 0.7', unit='B', basis='cc-pvdz', verbose=0)
        cell = cr.molecule_in_box(mol, 9.)
        aux = df.addons.aug_etb(mol, beta=2.0)
        r0 = RSDF(cell, aux, precision=1e-10, nuc_zeta=ZETA).build()
        r1 = RSDF(cell, aux, precision=1e-10, nuc_zeta=ZETA)
        r1.max_memory = 1e-3     # one shell per row block
        r1.cderi_to_disk = True
        r1.build()
        self.assertFalse(isinstance(r1.cderi, np.ndarray))
        rng = np.random.default_rng(3)
        nao = cell.nao
        d = rng.normal(size=(nao, nao)) + 1j * rng.normal(size=(nao, nao))
        for dm in (d.real + d.real.T, d):
            j0, k0 = r0.get_jk(dm)
            j1, k1 = r1.get_jk(dm)
            self.assertLess(abs(j1 - j0).max(), 1e-12)
            self.assertLess(abs(k1 - k0).max(), 1e-12)
        self.assertLess(abs(r1.get_nuc() - r0.get_nuc()).max(), 1e-12)
        # J of a complex density through loop() vs an explicit unpacked contraction
        from pyscf import lib
        L = lib.unpack_tril(np.vstack(list(r0.loop())))
        self.assertLess(abs(np.einsum('Pij,P->ij', L, np.einsum('Pij,ji->P', L, d)) - r0.get_jk(d)[0]).max(), 1e-12)

    def test_occ_ri_exchange(self):
        '''K from occupied factors (tagged dm) equals K from the density matrix,
        for real/fractional, complex, and spin-orbital (GHF) orbitals.'''
        from pyscf import lib
        mol = gto.M(atom='H 0 0 -0.7; H 0 0 0.7', unit='B', basis='cc-pvdz', verbose=0)
        cell = cr.molecule_in_box(mol, 9.)
        r = RSDF(cell, df.addons.aug_etb(mol, beta=2.0), precision=1e-10, nuc_zeta=ZETA).build()
        nao = cell.nao
        rng = np.random.default_rng(7)
        occ = np.array([2., 1.3, .4, 0., .7])
        for c in (rng.normal(size=(nao, 5)), rng.normal(size=(nao, 5)) + 1j * rng.normal(size=(nao, 5))):
            dm = (c * occ) @ c.conj().T
            j0, k0 = r.get_jk(dm, hermi=0)
            j1, k1 = r.get_jk(lib.tag_array(dm, mo_coeff=c, mo_occ=occ), hermi=0)
            self.assertLess(abs(k1 - k0).max(), 1e-12)
            self.assertLess(abs(j1 - j0).max(), 1e-12)
        # two densities at once (UHF-like tags)
        c2 = rng.normal(size=(2, nao, 5))
        dm2 = np.einsum('xpi,i,xqi->xpq', c2, occ, c2)
        k0 = r.get_jk(dm2, with_j=False)[1]
        k1 = r.get_jk(lib.tag_array(dm2, mo_coeff=c2, mo_occ=np.array([occ, occ])), with_j=False)[1]
        self.assertLess(abs(k1 - k0).max(), 1e-12)
        # spin-orbital, non-collinear complex spinors
        cg = rng.normal(size=(2 * nao, 4)) + 1j * rng.normal(size=(2 * nao, 4))
        og = np.array([1., 1., .5, .25])
        dmg = (cg * og) @ cg.conj().T
        j0, k0 = r.get_jk_ghf(dmg, hermi=1)
        j1, k1 = r.get_jk_ghf(lib.tag_array(dmg, mo_coeff=cg, mo_occ=og), hermi=1)
        self.assertLess(abs(j1 - j0).max(), 1e-12)
        self.assertLess(abs(k1 - k0).max(), 1e-12)



class GHFBlocks(unittest.TestCase):
    def test_ghf_assembly_matches_molecular_ghf(self):
        '''ghf_get_jk with a molecular scalar J/K reproduces pyscf's molecular
        GHF get_jk (correct alpha-beta slice) for a non-collinear density.'''
        from pyscf.scf import ghf as mol_ghf, hf
        from socutils.pbc.rsdf import ghf_get_jk
        mol = gto.M(atom='O 0 0 0; H 0 .7 .6; H 0 -.7 .6', basis='6-31g', verbose=0)
        n = mol.nao
        rng = np.random.default_rng(3)
        c = rng.standard_normal((2 * n, 6)) + 1j * rng.standard_normal((2 * n, 6))
        dm = c @ c.conj().T                       # Hermitian, non-collinear
        scalar = lambda d, h, wj, wk: hf.get_jk(mol, d, h, with_j=wj, with_k=wk)
        # pyscf.scf.ghf.get_jk's default jkbuild=hf.get_jk passes with_j as
        # vhfopt (positional mismatch), so give it a correct one
        jkbuild = lambda m, d, h, wj, wk, omega=None: hf.get_jk(m, d, h, with_j=wj, with_k=wk)
        for hermi, d in ((1, dm), (0, dm + .1 * rng.standard_normal(dm.shape))):
            vj, vk = ghf_get_jk(scalar, d, hermi)
            rj, rk = mol_ghf.get_jk(mol, d, hermi, jkbuild=jkbuild)
            self.assertLess(abs(vj - rj).max(), 1e-11)
            self.assertLess(abs(vk - rk).max(), 1e-11)


if __name__ == '__main__':
    unittest.main()
