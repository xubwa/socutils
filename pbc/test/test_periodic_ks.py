'''Gamma-point periodic spinor Kohn-Sham (socutils.pbc.spinor_ks): N2 in a box
extrapolates (A/L^3) to the molecular socutils SpinorDFT energy with the same
auxiliary basis and grid level, for a pure (PBE) and a hybrid (PBE0) functional.'''
import unittest
from pyscf import gto, df
from socutils.scf import spinor_hf
from socutils.dft import dft as sdft
from socutils.pbc import coulomb_reference as cr

try:
    from socutils.pbc.rsdf import RSDF, _libpbcdf
    from socutils.pbc import spinor_ks
    _libpbcdf()
    HAVE_LIB = True
except OSError:
    HAVE_LIB = False


@unittest.skipUnless(HAVE_LIB, 'libsocpbcdf.so not built (make)')
class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='N 0 0 -1.04; N 0 0 1.04', unit='B', basis='cc-pvdz', verbose=0)
        cls.aux = df.addons.aug_etb(cls.mol, beta=2.0)
        cls.rsdf = {L: RSDF(cr.molecule_in_box(cls.mol, L), cls.aux, precision=1e-11, nuc_zeta=1e14).build()
                    for L in (20., 26.)}

    def _check(self, xc):
        mm = sdft.SpinorDFT(self.mol, xc=xc).density_fit(auxbasis=self.aux)
        mm.verbose = 0
        mm.grids.level = 4
        e_mol = mm.kernel()
        res = {}
        for L, r in self.rsdf.items():
            mf = spinor_ks.periodic_ks(spinor_hf.SpinorSCF(r.cell), r, xc=xc)
            mf.verbose = 0
            res[L] = mf.kernel()
            self.assertTrue(mf.converged)
        (a, ea), (b, eb) = sorted(res.items())
        self.assertAlmostEqual((eb * b**3 - ea * a**3) / (b**3 - a**3), e_mol, 5)

    def test_pbe(self):
        self._check('PBE')

    def test_pbe0(self):
        self._check('PBE0')


if __name__ == '__main__':
    unittest.main()
