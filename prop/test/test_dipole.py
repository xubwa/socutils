#!/usr/bin/env python
'''Dipole driver on wavefunctions whose MO coefficients are genuinely complex.

A symmetric H2O in the standard frame is a weak test: the spinor-basis
gamma is nearly real and ``Tr(O gamma)`` and ``Tr(O gamma^T)`` agree to
1e-5, so a transposed density convention would pass unnoticed.  Here the
molecules are rotated by generic Euler angles (rotated H2O, a nonplanar C2
H2O2), and the CASCI is rerun in orbitals carrying random phases
``exp(i theta_p)``: the level-summed dipoles must be gauge invariant while
the transposed contraction is off by 1e-2 .. 1e-1 a.u.

The operator store is an in-memory stand-in with the CfourPropStore
surface, wrapping pyscf's ``int1e_r_spinor``; the reference is the
nonrelativistic RHF / CASCI dipole with the same integrals.
'''

import unittest
import numpy
from pyscf import gto, scf, mcscf
from socutils.scf import spinor_hf
from socutils.mcscf import zcasci
from socutils.prop import cfour, observables
from socutils.prop.densities import densities
from socutils.prop.contract import to_mo

BOHR = 0.529177


class MemStore:
    '''Same surface as CfourPropStore; operators from pyscf integrals.'''
    def __init__(self, mol):
        self.mol = mol
        r = mol.intor('int1e_r_spinor')
        self._ops = {'SOX2CDPX': r[0], 'SOX2CDPY': r[1], 'SOX2CDPZ': r[2]}

    def op(self, name, atom=None, comp=None, check_hermi=True):
        return self._ops[name.upper()]

    def meta(self, name):
        return cfour.meta(name)


def euler(a, b, c):
    ca, sa, cb, sb, cc, sc = (numpy.cos(a), numpy.sin(a), numpy.cos(b),
                              numpy.sin(b), numpy.cos(c), numpy.sin(c))
    rz1 = numpy.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    ry = numpy.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz2 = numpy.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rz1 @ ry @ rz2


def nonrel_dipole(mol, dm_ao):
    return (-numpy.einsum('xpq,qp->x', mol.intor('int1e_r'), dm_ao)
            + observables.nuclear_dipole(mol))


def level_sums(mc, mus):
    '''Dipole summed over each degenerate level -- gauge invariant even when
    the solver picks a different basis inside the level.'''
    rows = [(meta['level'], mus[meta['root']]) for _, _, _, meta in densities(mc)]
    levels = sorted(set(l for l, _ in rows))
    return numpy.array([sum(v for l, v in rows if l == lv) for lv in levels])


def transpose_discriminant(mc, ops):
    worst = 0.0
    for _, g, mo, _ in densities(mc):
        omo = to_mo(ops, mo)
        right = numpy.einsum('xpq,pq->x', omo, g)
        wrong = numpy.einsum('xpq,qp->x', omo, g)
        worst = max(worst, abs(right - wrong).max())
    return worst


class DipoleComplexMO(unittest.TestCase):
    ncas, nelecas, nroots = 8, 6, 6

    def check(self, atoms, min_discriminant):
        mol = gto.M(atom=atoms, basis='sto-3g', unit='Bohr', verbose=0)
        store = MemStore(mol)
        ops = numpy.array([store.op(n) for n in ('SOX2CDPX', 'SOX2CDPY', 'SOX2CDPZ')])

        rhf = scf.RHF(mol).run()
        mf = spinor_hf.SpinorSCF(mol).run()
        self.assertGreater(abs(mf.mo_coeff.imag).max(), 0.3)
        self.assertAlmostEqual(mf.e_tot, rhf.e_tot, 10)
        mu = observables.dipole(mf, store, verbose=0)
        self.assertEqual(mu.shape, (3,))
        self.assertTrue(numpy.all(abs(mu) > 0.1))
        ref = rhf.dip_moment(unit='AU', verbose=0)
        self.assertAlmostEqual(abs(mu - ref).max(), 0, 10)

        mcn = mcscf.CASCI(rhf, self.ncas // 2, self.nelecas).run()
        refc = nonrel_dipole(mol, mcn.make_rdm1())
        mc = zcasci.CASCI(mf, self.ncas, self.nelecas)
        mc.fcisolver.nroots = self.nroots
        mc.kernel()
        mus = observables.dipole(mc, store, verbose=0)
        self.assertEqual(mus.shape, (self.nroots, 3))
        self.assertAlmostEqual(mc.e_tot[0], mcn.e_tot, 10)
        self.assertAlmostEqual(abs(mus[0] - refc).max(), 0, 10)

        # random phases on every orbital: energies and level-summed dipoles
        # invariant, while the transposed contraction would not be
        rng = numpy.random.default_rng(7)
        ph = numpy.exp(1j * rng.uniform(0, 2 * numpy.pi, mf.mo_coeff.shape[1]))
        mc2 = zcasci.CASCI(mf, self.ncas, self.nelecas)
        mc2.fcisolver.nroots = self.nroots
        mc2.kernel(mf.mo_coeff * ph)
        mus2 = observables.dipole(mc2, store, verbose=0)
        self.assertAlmostEqual(abs(mc2.e_tot - mc.e_tot).max(), 0, 10)
        self.assertAlmostEqual(abs(level_sums(mc, mus) - level_sums(mc2, mus2)).max(), 0, 10)
        self.assertGreater(transpose_discriminant(mc2, ops), min_discriminant)

    def test_h2o_rotated(self):
        xyz = numpy.array([[0, 0, 0.1], [0, 0.75, -0.5], [0, -0.75, -0.5]]) / BOHR
        xyz = xyz @ euler(0.7, 1.1, 2.3).T + numpy.array([0.3, -0.2, 0.5])
        self.check([('O', xyz[0]), ('H', xyz[1]), ('H', xyz[2])], 5e-2)

    def test_h2o2_c2(self):
        xyz = numpy.array([[0.0, 0.73, 0.0], [0.0, -0.73, 0.0],
                           [0.85, 0.95, 0.55], [-0.85, -0.95, 0.55]]) / BOHR
        xyz = xyz @ euler(0.4, 0.9, 1.7).T
        self.check([('O', xyz[0]), ('O', xyz[1]), ('H', xyz[2]), ('H', xyz[3])], 5e-3)


if __name__ == '__main__':
    unittest.main()
