#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#

'''
Tests for the Abelian double-group tables in socutils.symm.double_group.

1. Multiplication tables equal DIRAC's (moltra/traone.F, GMULTA) for C1, C2/Cs,
   Ci and C2h, after mapping DIRAC's irrep names onto ours.
2. Characters equal the Altmann-Herzig tables (as printed in the DIRAC manual,
   "Symmetry-handling at the correlated level") and are multiplicative.
3. Group axioms: identity, conj gives the inverse, fermion x fermion = boson.
4. Linear groups: m_j adds, conj flips m_j, descent to C2h/C2 is n mod 4.
'''

import itertools
import unittest
import numpy

from socutils.symm import double_group as dg

FINITE = ('C1', 'Ci', 'C2', 'Cs', 'C2h')


def _dirac_table(names, entries):
    '''Symmetric DIRAC MULTB table as {(name_i, name_j): name_k}.'''
    tab = {}
    for (i, j), k in entries.items():
        tab[names[i-1], names[j-1]] = names[k-1]
        tab[names[j-1], names[i-1]] = names[k-1]
    return tab


# DIRAC GMULTA, transcribed literally (fermion irreps first, then boson irreps)
DIRAC = {
    'C1': _dirac_table(['A1/2', 'A'], {(1, 1): 2, (2, 1): 1, (2, 2): 2}),
    'C2': _dirac_table(['1E', '2E', 'A', 'B'], {
        (1, 1): 4, (2, 1): 3, (3, 1): 1, (4, 1): 2, (2, 2): 4, (3, 2): 2, (4, 2): 1,
        (3, 3): 3, (4, 3): 4, (4, 4): 3}),
    'Cs': _dirac_table(['1E', '2E', "A'", 'A"'], {
        (1, 1): 4, (2, 1): 3, (3, 1): 1, (4, 1): 2, (2, 2): 4, (3, 2): 2, (4, 2): 1,
        (3, 3): 3, (4, 3): 4, (4, 4): 3}),
    'Ci': _dirac_table(['A1/2g', 'A1/2u', 'Ag', 'Au'], {
        (1, 1): 3, (2, 1): 4, (3, 1): 1, (4, 1): 2, (2, 2): 3, (3, 2): 2, (4, 2): 1,
        (3, 3): 3, (4, 3): 4, (4, 4): 3}),
    'C2h': _dirac_table(['1Eg', '2Eg', '1Eu', '2Eu', 'Ag', 'Bg', 'Au', 'Bu'], {
        (1, 1): 6, (2, 1): 5, (5, 1): 1, (6, 1): 2, (2, 2): 6, (5, 2): 2, (6, 2): 1,
        (5, 5): 5, (6, 5): 6, (6, 6): 5,
        (3, 3): 6, (4, 3): 5, (7, 3): 1, (8, 3): 2, (4, 4): 6, (7, 4): 2, (8, 4): 1,
        (7, 7): 5, (8, 7): 6, (8, 8): 5,
        (3, 1): 8, (4, 1): 7, (7, 1): 3, (8, 1): 4, (3, 2): 7, (4, 2): 8, (7, 2): 4,
        (8, 2): 3, (7, 5): 7, (8, 5): 8, (7, 6): 8, (8, 6): 7,
        (5, 3): 3, (6, 3): 4, (5, 4): 4, (6, 4): 3}),
}

# Altmann-Herzig fermion characters (DIRAC manual), columns as OPERATOR_TABLE
ALTMANN = {
    'C1': {'A1/2': (1,)},
    'Ci': {'A1/2g': (1, 1), 'A1/2u': (1, -1)},
    'C2': {'1E': (1, 1j), '2E': (1, -1j)},
    'Cs': {'1E': (1, 1j), '2E': (1, -1j)},
    'C2h': {'1Eg': (1, 1j, 1, 1j), '2Eg': (1, -1j, 1, -1j),
            '1Eu': (1, 1j, -1, -1j), '2Eu': (1, -1j, -1, 1j)},
}
# PySCF single-group characters for the boson irreps (pyscf.symm.param)
PYSCF_BOSON = {
    'C1': {'A': (1,)},
    'Ci': {'Ag': (1, 1), 'Au': (1, -1)},
    'C2': {'A': (1, 1), 'B': (1, -1)},
    'Cs': {"A'": (1, 1), 'A"': (1, -1)},
    'C2h': {'Ag': (1, 1, 1, 1), 'Bg': (1, -1, 1, -1), 'Au': (1, 1, -1, -1), 'Bu': (1, -1, -1, 1)},
}


class KnownValues(unittest.TestCase):
    def test_dirac_multiplication_tables(self):
        for g, tab in DIRAC.items():
            ids = dg.irrep_ids(g)
            self.assertEqual(len(tab), len(ids) ** 2)
            for (a, b), c in tab.items():
                ia = dg.irrep_name2id(g, a)
                ib = dg.irrep_name2id(g, b)
                self.assertEqual(dg.irrep_id2name(g, dg.direct_prod(ia, ib, g)), c, (g, a, b))

    def test_dirac_irrep_order(self):
        self.assertEqual([dg.irrep_id2name('C2h', i) for i in dg.irrep_ids('C2h')],
                         ['1Eg', '2Eg', '1Eu', '2Eu', 'Ag', 'Bg', 'Au', 'Bu'])
        self.assertEqual([dg.irrep_id2name('C2', i) for i in dg.irrep_ids('C2')],
                         ['1E', '2E', 'A', 'B'])

    def test_characters(self):
        for ref in (ALTMANN, PYSCF_BOSON):
            for g, tab in ref.items():
                for name, chars in tab.items():
                    ir = dg.irrep_name2id(g, name)
                    for op, chi in zip(dg.OPERATOR_TABLE[g], chars):
                        self.assertAlmostEqual(dg.character(g, ir, op), chi, 12, (g, name, op))
                        bar = -chi if dg.is_fermion(ir) else chi
                        self.assertAlmostEqual(dg.character(g, ir, 'b' + op), bar, 12)
        # every irrep is covered by the two reference tables
        for g in FINITE:
            names = set(ALTMANN[g]) | set(PYSCF_BOSON[g])
            self.assertEqual(names, {dg.irrep_id2name(g, i) for i in dg.irrep_ids(g)})

    def test_characters_multiplicative(self):
        for g in FINITE:
            for a, b in itertools.product(dg.irrep_ids(g), repeat=2):
                c = dg.direct_prod(a, b, g)
                for op in dg.OPERATOR_TABLE[g]:
                    self.assertAlmostEqual(dg.character(g, c, op),
                                           dg.character(g, a, op) * dg.character(g, b, op), 12)

    def test_group_axioms(self):
        for g in FINITE:
            ids = dg.irrep_ids(g)
            self.assertEqual(len(ids), 2 * len(PYSCF_BOSON[g]))
            self.assertEqual(len(dg.irrep_ids(g, 'fermion')), len(dg.irrep_ids(g, 'boson')))
            for a in ids:
                self.assertEqual(dg.direct_prod(a, 0, g), a)
                self.assertEqual(dg.direct_prod(dg.conj(a, g), a, g), 0)
                self.assertEqual(dg.conj(dg.conj(a, g), g), a)
                for b in ids:
                    c = dg.direct_prod(a, b, g)
                    self.assertIn(c, ids)
                    self.assertEqual(dg.is_fermion(c), dg.is_fermion(a) != dg.is_fermion(b))
                # conj = complex-conjugate characters
                for op in dg.OPERATOR_TABLE[g]:
                    self.assertAlmostEqual(dg.character(g, dg.conj(a, g), op),
                                           dg.character(g, a, op).conjugate(), 12)

    def test_kramers_partners(self):
        conj = {g: {dg.irrep_id2name(g, i): dg.irrep_id2name(g, dg.conj(i, g))
                    for i in dg.irrep_ids(g, 'fermion')} for g in FINITE}
        self.assertEqual(conj['C2h'], {'1Eg': '2Eg', '2Eg': '1Eg', '1Eu': '2Eu', '2Eu': '1Eu'})
        self.assertEqual(conj['C2'], {'1E': '2E', '2E': '1E'})
        self.assertEqual(conj['Ci'], {'A1/2g': 'A1/2g', 'A1/2u': 'A1/2u'})
        self.assertEqual(conj['C1'], {'A1/2': 'A1/2'})

    def test_direct_prod_arrays(self):
        orbsym = numpy.array(dg.irrep_ids('C2h', 'fermion'))
        tab = dg.direct_prod(dg.conj(orbsym, 'C2h'), orbsym, 'C2h')
        self.assertEqual(tab.shape, (4, 4))
        # <p|O|q> totally symmetric only on the diagonal
        self.assertTrue(numpy.array_equal(tab == 0, numpy.eye(4, dtype=bool)))

    def test_linear(self):
        for g in ('Coov', 'Dooh'):
            for name in ('1/2', '-3/2', '5/2', '0', '2', '-1'):
                if g == 'Dooh':
                    for gu in 'gu':
                        self.assertEqual(dg.irrep_id2name(g, dg.irrep_name2id(g, name + gu)), name + gu)
                else:
                    self.assertEqual(dg.irrep_id2name(g, dg.irrep_name2id(g, name)), name)
            self.assertEqual(dg.irrep_name2id('Coov', '+1/2'), dg.irrep_name2id('Coov', '1/2'))
        a = dg.irrep_name2id('Dooh', '3/2u')
        b = dg.irrep_name2id('Dooh', '-1/2u')
        self.assertEqual(dg.irrep_id2name('Dooh', dg.direct_prod(a, b, 'Dooh')), '1g')
        self.assertEqual(dg.irrep_id2name('Dooh', dg.conj(a, 'Dooh')), '-3/2u')
        self.assertEqual(len(dg.irrep_ids('Coov', 'fermion', mj_max=2.5)), 6)
        with self.assertRaises(KeyError):
            dg.irrep_name2id('Dooh', '1/2')
        with self.assertRaises(KeyError):
            dg.irrep_name2id('Coov', '2/2')

    def test_descend(self):
        # homomorphism: descend(a x b) = descend(a) x descend(b)
        chains = [('Dooh', s) for s in ('Coov', 'C2h', 'Ci', 'C2', 'Cs', 'C1')] + \
                 [('Coov', 'C2'), ('Coov', 'C1')] + \
                 [('C2h', s) for s in ('Ci', 'C2', 'Cs', 'C1')] + \
                 [('C2', 'C1'), ('Cs', 'C1'), ('Ci', 'C1')]
        for g, s in chains:
            if g in ('Coov', 'Dooh'):
                ids = dg.irrep_ids(g, mj_max=3.5)
            else:
                ids = dg.irrep_ids(g)
            for a, b in itertools.product(ids, repeat=2):
                ab = dg.descend(dg.direct_prod(a, b, g), g, s)
                self.assertEqual(ab, dg.direct_prod(dg.descend(a, g, s), dg.descend(b, g, s), s), (g, s))
                self.assertIn(dg.descend(a, g, s), dg.irrep_ids(s, mj_max=3.5) if s == 'Coov' else dg.irrep_ids(s))
        # characters are preserved for the common operations
        for a in dg.irrep_ids('C2h'):
            for s in ('Ci', 'C2', 'Cs'):
                for op in dg.OPERATOR_TABLE[s]:
                    self.assertAlmostEqual(dg.character(s, dg.descend(a, 'C2h', s), op),
                                           dg.character('C2h', a, op), 12)
        # linear: C2 = C(pi)
        for a in dg.irrep_ids('Dooh', mj_max=3.5):
            self.assertAlmostEqual(dg.character('C2h', dg.descend(a, 'Dooh', 'C2h'), 'C2'),
                                   dg.character('Dooh', a, ('C', numpy.pi)), 12)
        self.assertEqual(dg.irrep_id2name('C2', dg.descend(dg.irrep_name2id('Coov', '1/2'), 'Coov', 'C2')), '2E')
        self.assertEqual(dg.irrep_id2name('C2', dg.descend(dg.irrep_name2id('Coov', '-1/2'), 'Coov', 'C2')), '1E')

    def test_double_group_of(self):
        self.assertEqual(dg.double_group_of('D2h'), 'C2h')
        self.assertEqual(dg.double_group_of('C2v'), 'C2')
        self.assertEqual(dg.double_group_of('D2'), 'C2')
        self.assertEqual(dg.double_group_of('SO3'), 'Dooh')
        self.assertEqual(dg.double_group_of('Cs'), 'Cs')
        with self.assertRaises(KeyError):
            dg.double_group_of('C3v')


if __name__ == '__main__':
    print('Tests for socutils.symm.double_group')
    unittest.main()
