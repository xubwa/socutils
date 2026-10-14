'''Periodic X2CAMF: X2CMP in its one-centre approximation (spinor_scf).

* For an atom the one-centre approximation is exact: the atom in a box
  extrapolates (A/L^3) to the molecular socutils X2CAMF energy of the atom,
  with point nuclei (RSDF zeta 1e14) and with the Gaussian nuclear model
  (cell.nucmod 'G', RSDF zeta = nuclear_zeta(cell)).
* For a molecule (HI) the one-centre blocks reproduce the molecular X2CAMF
  energy up to the dropped multi-centre relativistic terms (4e-5 Eh of the
  386 Eh relativistic effect).
* Only one-centre flavors are accepted for a cell.
'''
import unittest
import warnings
from pyscf import gto, df
from socutils.scf import spinor_hf
from socutils.pbc import coulomb_reference as cr

try:
    from socutils.pbc.rsdf import RSDF, _libpbcdf
    from socutils.pbc import spinor_scf
    _libpbcdf()
    HAVE_LIB = True
except OSError:
    HAVE_LIB = False


def _quiet(obj):
    obj.verbose = 0
    return obj


def _extrapolate(res):
    (a, ea), (b, eb) = sorted(res.items())
    return (eb * b**3 - ea * a**3) / (b**3 - a**3)


@unittest.skipUnless(HAVE_LIB, 'libsocpbcdf.so not built (make)')
class KnownValues(unittest.TestCase):
    def _atom_in_box(self, nucmod):
        atm = gto.M(atom='Kr 0 0 0', unit='B', basis='x2c-svpall', verbose=0, nucmod=nucmod)
        aux = df.addons.aug_etb(atm, beta=2.0)
        e_atom = _quiet(spinor_hf.SpinorSCF(atm).x2camf().density_fit(auxbasis=aux)).kernel()
        res = {}
        for L in (22., 28.):
            cell = cr.molecule_in_box(atm, L)
            cell.nucmod = nucmod
            cell.build(False, False)
            zeta = spinor_scf.nuclear_zeta(cell)
            r = RSDF(cell, aux, precision=1e-11, nuc_zeta=1e14 if zeta is None else zeta).build()
            res[L] = _quiet(spinor_scf.SpinorRSDF(cell, r).x2camf()).kernel()
        return _extrapolate(res), e_atom

    def test_atom_point_nucleus(self):
        e, ref = self._atom_in_box({})
        self.assertAlmostEqual(e, ref, 5)

    def test_atom_gaussian_nucleus(self):
        e, ref = self._atom_in_box({'Kr': 'G'})
        self.assertAlmostEqual(e, ref, 5)

    def test_one_centre_error_molecule(self):
        mol = gto.M(atom='I 0 0 0; H 0 0 3.04', unit='B', basis='x2c-svpall', verbose=0)
        aux = df.addons.aug_etb(mol, beta=2.0)
        e_full = _quiet(spinor_hf.SpinorSCF(mol).x2camf().density_fit(auxbasis=aux)).kernel()
        hlp = spinor_scf.OneCenterX2CMPHelper(mol, 'x2camf')
        h1c = mol.intor('int1e_kin_spinor') + mol.intor('int1e_nuc_spinor') + hlp.get_correction()
        mf = _quiet(spinor_hf.SpinorSCF(mol).density_fit(auxbasis=aux))
        mf.get_hcore = lambda *args, **kwargs: h1c
        self.assertLess(abs(mf.kernel() - e_full), 1e-4)

    def test_flavors(self):
        atm = gto.M(atom='He 0 0 0', unit='B', basis='cc-pvdz', verbose=0)
        cell = cr.molecule_in_box(atm, 8.)
        r = RSDF(cell, df.addons.aug_etb(atm, beta=2.0), nuc_zeta=1e14)
        mf = spinor_scf.SpinorRSDF(cell, r)
        for bad in ('x2cmp', 'sfmp', 'x2camf_au'):
            with self.assertRaises(ValueError):
                mf.x2cmp(bad)
        self.assertEqual(mf.x2cmp().with_x2c.flavor, 'x2cmp_1c')
        self.assertEqual(mf.x2camf().with_x2c.flavor, 'x2camf')
        # nuclear-model consistency
        with self.assertRaises(ValueError):
            cg = cr.molecule_in_box(atm, 8.)
            cg.nucmod = 'G'
            cg.build(False, False)
            spinor_scf.SpinorRSDF(cg, RSDF(cg, df.addons.aug_etb(atm, beta=2.0), nuc_zeta=1e14))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            spinor_scf.check_nuclear_model(cell, RSDF(cell, None, nuc_zeta=1e12))
            self.assertTrue(any('finite-nucleus' in str(x.message) for x in w))


if __name__ == '__main__':
    unittest.main()
