'''Self-checks of the numerically exact periodic Coulomb references.

Every reference is checked against an independent route: literature Madelung
constants, two different lattice-sum formulas, the point-charge limit, and the
invariances that any periodic Coulomb code must satisfy.
'''
import unittest
import numpy as np
from pyscf.pbc import gto as pgto
from pyscf.pbc.tools import super_cell
from socutils.pbc import coulomb_reference as cr

# Madelung constants relative to the nearest-neighbour distance
MADELUNG = {'NaCl': 1.747564594633182, 'CsCl': 1.762674773070992, 'ZnS': 1.638055053388789}
FCC = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


def binary_lattices(r0=1.0):
    a_cs = 2 * r0 / np.sqrt(3)
    a_zb = 4 * r0 / np.sqrt(3)
    return {
        'NaCl': (FCC * 2 * r0, [[0, 0, 0], [r0, 0, 0]]),
        'CsCl': (np.eye(3) * a_cs, [[0, 0, 0], [a_cs / 2] * 3]),
        'ZnS': (FCC * a_zb, [[0, 0, 0], [a_zb / 4] * 3]),
    }


def compact_s_cell(shift=(0, 0, 0)):
    '''Li/H rock salt with one compact s function per atom; AO products between
    different lattice images are below 1e-20, so with a diagonal density matrix
    the electron density is exactly a set of Gaussian charges.'''
    r0 = 4.0
    shift = np.asarray(shift, dtype=float)
    return pgto.M(a=FCC * 2 * r0, unit='B', verbose=0, spin=0,
                  atom=[['Li', shift], ['H', shift + [r0, 0, 0]]],
                  basis={'Li': [[0, [3.0, 1.]]], 'H': [[0, [2.0, 1.]]]})


ZETA = 20.0
KE = 1200.
RPAIR = 12.


class KnownValues(unittest.TestCase):
    def test_madelung(self):
        for name, (a, x) in binary_lattices().items():
            for eta in (0.7, 1.5, 3.0):
                m = -cr.ewald_point_charges(a, x, [1, -1], eta=eta)
                self.assertAlmostEqual(m, MADELUNG[name], 12, msg=f'{name} eta={eta}')

    def test_gaussian_charges_split_independence(self):
        a, x = binary_lattices()['NaCl']
        for alphas in ([3., 5.], [0.8, 40.]):
            e0 = cr.gaussian_charges_recip(a, x, [1, -1], alphas)
            for eta in (0.5, 1., 2., 4.):
                e1 = cr.gaussian_charges_ewald(a, x, [1, -1], alphas, eta=eta)
                self.assertAlmostEqual(e1, e0, 13)

    def test_point_charge_limit(self):
        a, x = binary_lattices()['NaCl']
        alphas = [1e4, 1e4]
        e = cr.gaussian_charges_ewald(a, x, [1, -1], alphas) - cr.gaussian_self_energy([1, -1], alphas)
        self.assertAlmostEqual(e, -MADELUNG['NaCl'], 12)

    def test_ao_density_equals_gaussian_charges(self):
        cell = compact_s_cell()
        dm = np.diag([2., 2.])
        res = cr.coulomb_reference(cell, dm, ZETA, KE, RPAIR)
        x = cell.atom_coords()
        eg = cr.gaussian_charges_recip(cell.lattice_vectors(), [x[0], x[0], x[1], x[1]],
                                       [3, -2, 1, -2], [ZETA, 6.0, ZETA, 4.0])
        self.assertAlmostEqual(res['e_coul'], eg, 12)
        self.assertAlmostEqual(res['e_ee'], .5 * np.einsum('ij,ji', dm, res['vj']), 12)
        self.assertAlmostEqual(res['e_en'], np.einsum('ij,ji', dm, res['vnuc']), 12)
        self.assertLess(res['max_imag'], 1e-12)

    def test_translation_invariance(self):
        dm = np.diag([2., 2.])
        r0 = cr.coulomb_reference(compact_s_cell(), dm, ZETA, KE, RPAIR)
        r1 = cr.coulomb_reference(compact_s_cell([0.37, -1.1, 0.52]), dm, ZETA, KE, RPAIR)
        self.assertAlmostEqual(r1['e_coul'], r0['e_coul'], 12)
        self.assertLess(abs(r1['vj'] - r0['vj']).max(), 1e-12)
        self.assertLess(abs(r1['vnuc'] - r0['vnuc']).max(), 1e-12)

    def test_supercell_replication(self):
        '''J of a fixed periodic density does not depend on the choice of cell.'''
        cell = compact_s_cell()
        dm = np.array([[2., .3], [.3, 2.]])
        res = cr.coulomb_reference(cell, dm, ZETA, KE, RPAIR)
        sc = super_cell(cell, [2, 1, 1])
        dm_sc = np.kron(np.ones((2, 2)), dm)
        res_sc = cr.coulomb_reference(sc, dm_sc, ZETA, KE, RPAIR)
        self.assertAlmostEqual(res_sc['e_coul'], 2 * res['e_coul'], 11)
        # Folding the supercell Gamma matrix back onto cell 0 gives the primitive one
        nao = cell.nao
        for key in ('vj', 'vnuc'):
            folded = res_sc[key][:nao, :nao] + res_sc[key][:nao, nao:]
            self.assertLess(abs(folded - res[key]).max(), 1e-11)


if __name__ == '__main__':
    unittest.main()
