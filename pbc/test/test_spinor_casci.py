'''Gamma-point spinor HF and CASCI with RSDF (socutils.pbc.spinor_scf).

A molecule in a cubic box: with the Madelung term (exxdiv='ewald') the
remaining box dependence of HF and CASCI is the tin-foil dipole term, exactly
A/L^3, so two box sizes extrapolate to the molecular result with the same
auxiliary basis (molecular DF).  A CAS spanning only occupied orbitals is a
single determinant and must reproduce the periodic HF energy, which checks the
integral transformation (_CDERIS), h1eff and the core energy on RSDF.
'''
import unittest
import numpy as np
from pyscf import gto, df
from socutils.scf import spinor_hf
from socutils.mcscf import zcasci
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


@unittest.skipUnless(HAVE_LIB, 'libsocpbcdf.so not built (make)')
class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom='N 0 0 -1.04; N 0 0 1.04', unit='B', basis='cc-pvdz', verbose=0)
        cls.aux = df.addons.aug_etb(mol, beta=2.0)
        mm = _quiet(spinor_hf.SpinorSCF(mol).density_fit(auxbasis=cls.aux))
        cls.e_hf_mol = mm.kernel()
        mc = _quiet(zcasci.CASCI(mm, 12, 6))
        mc.kernel()
        cls.e_ci_mol = mc.e_tot
        cls.mol = mol
        cls.res = {}
        for L in (20., 26.):
            cell = cr.molecule_in_box(mol, L)
            r = RSDF(cell, cls.aux, precision=1e-11, nuc_zeta=1e14).build()
            mf = _quiet(spinor_scf.SpinorRSDF(cell, r))
            e_hf = mf.kernel()
            mc = _quiet(zcasci.CASCI(mf, 12, 6))
            mc.kernel()
            mc0 = _quiet(zcasci.CASCI(mf, 6, 6))
            mc0.kernel()
            cls.res[L] = dict(mf=mf, e_hf=e_hf, e_ci=mc.e_tot, e_cas_occ=mc0.e_tot)

    def _extrapolate(self, key):
        (a, ea), (b, eb) = [(L, self.res[L][key]) for L in sorted(self.res)]
        return (eb * b**3 - ea * a**3) / (b**3 - a**3)

    def test_box_limit_is_the_molecule(self):
        self.assertAlmostEqual(self._extrapolate('e_hf'), self.e_hf_mol, 5)
        self.assertAlmostEqual(self._extrapolate('e_ci'), self.e_ci_mol, 5)

    def test_cas_of_occupied_orbitals_is_hf(self):
        for r in self.res.values():
            self.assertAlmostEqual(r['e_cas_occ'], r['e_hf'], 9)

    def test_madelung_constant_term(self):
        '''exxdiv only shifts the energy by -nu_M N / 2.'''
        from pyscf.pbc import tools
        mf = self.res[20.]['mf']
        cell = mf.mol
        e_ewald = mf.energy_nuc()
        mf.exxdiv = None
        try:
            e_none = mf.energy_nuc()
        finally:
            mf.exxdiv = 'ewald'
        nu = tools.pbc.madelung(cell, np.zeros((1, 3)))
        self.assertAlmostEqual(e_ewald - e_none, -.5 * nu * cell.nelectron, 12)
        self.assertAlmostEqual(nu * 20., 2.8372974794806, 6)   # simple cubic


if __name__ == '__main__':
    unittest.main()
