'''
Tests for the 'x2cmp_1x' flavor of somf/x2cmp.py (x2cmp with one-centre
exchange, cf. SOMF(1X)) and its projector-form variant 'x2cmp_1x_proj'.

  * the analytic J of the atomic densities (somf/onecenter_j.py) reproduces
    the molecular get_j build;
  * 'x2cmp-1x' / 'x2cmp-1x-proj' spellings and the proj_1x switch select the
    same Hamiltonians as the flavors;
  * the energies stay close to 'x2cmp' (one-centre K is the only approximation),
    also with diffuse functions added.
'''

import os
import contextlib
import unittest
import numpy as np

from pyscf import gto

from socutils.scf import spinor_hf


def _quiet(f):
    with open(os.devnull, 'w') as fn, contextlib.redirect_stdout(fn):
        return f()


def _run(mol, flavor, scf=True, **attrs):
    mf = spinor_hf.SCF(mol).x2cmp(flavor=flavor, with_gaunt=False, with_breit=False)
    for key, val in attrs.items():
        setattr(mf.with_x2c, key, val)
    h = _quiet(lambda: mf.with_x2c.get_hcore(mol))
    e = _quiet(mf.kernel) if scf else None
    return h, mf.with_x2c, e


def _diffuse_basis(k, ratio=2.5):
    '''uncontracted cc-pVDZ with k extra even-tempered s and p shells'''
    bas = {}
    for el in ('H', 'Cl'):
        raw = gto.mole.uncontracted_basis(gto.basis.load('cc-pvdz', el))
        extra = []
        for l in (0, 1):
            emin = min(x[-1][0] for x in raw if x[0] == l)
            extra += [[l, [emin / ratio**(i + 1), 1.]] for i in range(k)]
        bas[el] = sorted(raw + extra, key=lambda sh: sh[0])
    return bas


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='H 0 0 0; Cl 0 0 1.275', basis='cc-pvdz', verbose=0)

    def test_analytic_j(self):
        h, x, _ = _run(self.mol, 'x2cmp_1x', scf=False)
        h_ref, x_ref, _ = _run(self.mol, 'x2cmp_1x', scf=False, j_1x='direct')
        self.assertAlmostEqual(abs(x.h4c - x_ref.h4c).max(), 0, 7)
        self.assertAlmostEqual(abs(x.veff_2c - x_ref.veff_2c).max(), 0, 9)
        self.assertAlmostEqual(abs(h - h_ref).max(), 0, 9)

    def test_alias_and_switch(self):
        h1 = _run(self.mol, 'x2cmp-1x-proj', scf=False)[0]
        h2, x2, _ = _run(self.mol, 'x2cmp_1x', scf=False, proj_1x=True)
        self.assertEqual(x2.amf_type, 'x2cmp_1x')
        self.assertAlmostEqual(abs(h1 - h2).max(), 0, 10)
        self.assertEqual(_run(self.mol, 'x2cmp-1x', scf=False)[1].amf_type, 'x2cmp_1x')

    def test_energy_vs_x2cmp(self):
        e_mp = _run(self.mol, 'x2cmp')[2]
        e_1x = _run(self.mol, 'x2cmp_1x')[2]
        e_1xp = _run(self.mol, 'x2cmp_1x_proj')[2]
        # HCl/cc-pVDZ, Dirac-Coulomb: -1.18e-5 (block), -2.8e-6 (projector)
        self.assertAlmostEqual(e_1x - e_mp, -1.18e-5, 6)
        self.assertAlmostEqual(e_1xp - e_mp, -2.76e-6, 6)

    def test_diffuse(self):
        mol = gto.M(atom='H 0 0 0; Cl 0 0 1.275', basis=_diffuse_basis(6), verbose=0)
        e_mp = _run(mol, 'x2cmp')[2]
        e_1xp = _run(mol, 'x2cmp_1x_proj')[2]
        self.assertLess(abs(e_1xp - e_mp), 1e-5)

    def test_cache(self):
        for proj in (False, True):
            h, x, _ = _run(self.mol, 'x2cmp_1x', scf=False, proj_1x=proj)
            mp = x.mp_1x
            self.assertEqual(mp['proj'], proj)
            self.assertAlmostEqual(abs(x.veff_4c - (mp['vj_4c'] - mp['vk_4c'])).max(), 0, 12)
            self.assertAlmostEqual(abs(x.veff_2c - (mp['vj_2c'] - mp['vk_2c'])).max(), 0, 12)
            # cached decoupling == the one _x2c_blocks would build from (h4c, m4c)
            cached = dict(x.x2c_intermediate)
            x.x2c_intermediate = None
            fresh = x._x2c_blocks()
            for i, key in enumerate(('a', 'e', 'x', 'st', 'r', 'l')):
                if key not in ('a',):     # eigenvectors: phase-free quantities only
                    self.assertAlmostEqual(abs(cached[key] - fresh[i]).max(), 0, 9)
            # adjoint of the cached picture change
            rng = np.random.default_rng(1)
            n4 = x.h4c.shape[0]
            h1 = rng.standard_normal((n4, n4)) + 1j * rng.standard_normal((n4, n4))
            h1 = h1 + h1.conj().T
            d2 = x.density_2c
            gh, gs = x.get_hfw1_adjoint(d2)
            self.assertAlmostEqual(abs(np.trace(d2 @ x.get_hfw1(h1)) - np.trace(gh @ h1)), 0, 7)

    def test_kernels_geometry_independent(self):
        from pyscf import scf
        mol2 = gto.M(atom='H 0 0 0; Cl 0.05 -0.02 1.31', basis='cc-pvdz', verbose=0)
        for proj in (False, True):
            x1 = _run(self.mol, 'x2cmp_1x', scf=False, proj_1x=proj)[1]
            x2 = _run(mol2, 'x2cmp_1x', scf=False, proj_1x=proj)[1]
            for a1, a2 in zip(x1.mp_1x['atoms'], x2.mp_1x['atoms']):
                for key in a1:
                    if key[0] in 'km':
                        self.assertAlmostEqual(abs(a1[key] - a2[key]).max(), 0, 9)
            if proj:
                # vk at the new geometry = cached M_K with the new overlap
                xmol2 = x2.get_xmol()[0]
                s4 = scf.DHF(xmol2).get_ovlp()
                vk = sum(s4[:, a['idx4']] @ a['m4'] @ s4[a['idx4'], :] for a in x1.mp_1x['atoms'])
                self.assertAlmostEqual(abs(vk - x2.mp_1x["vk_4c"]).max(), 0, 9)


if __name__ == '__main__':
    print('Tests for the x2cmp_1x flavor')
    unittest.main()
