'''PARI prototype (socutils.pbc.pari): with every atom in every fitting domain
the pair-atomic fit is the global fit, so PARI-K (and robust PARI-K) equal the
global RSDF exchange; J of robust PARI equals global J as well.'''
import unittest
import numpy as np
from pyscf import gto, df
from socutils.pbc import coulomb_reference as cr

try:
    from socutils.pbc.rsdf import RSDF, _libpbcdf
    from socutils.pbc import pari
    _libpbcdf()
    HAVE_LIB = True
except OSError:
    HAVE_LIB = False


@unittest.skipUnless(HAVE_LIB, 'libsocpbcdf.so not built (make)')
class KnownValues(unittest.TestCase):
    def test_full_domain_is_global_fit(self):
        mol = gto.M(atom='Li 0 0 0; H 0 0 3.0', unit='B', basis='cc-pvdz', verbose=0)
        cell = cr.molecule_in_box(mol, 8.)
        r = RSDF(cell, df.addons.aug_etb(mol, beta=2.0), precision=1e-10, nuc_zeta=1e14, lindep=1e-14)
        r.keep_raw = True
        r.build()
        rng = np.random.default_rng(0)
        c = rng.normal(size=(cell.nao, 3))
        dm = c @ c.T
        vj, vk = r.get_jk(dm)
        C = pari.pari_coefficients(r, lindep=1e-14, extra=lambda A, B: range(cell.natm))
        for robust in (False, True):
            j, k = pari.pari_jk(r, dm, C, robust=robust)
            self.assertLess(abs(k - vk).max(), 1e-8)
            if robust:
                self.assertLess(abs(j - vj).max(), 1e-8)
        # two-atom domains: a genuine (local) approximation
        j, k = pari.pari_jk(r, dm, pari.pari_coefficients(r))
        self.assertGreater(abs(k - vk).max(), 1e-8)


if __name__ == '__main__':
    unittest.main()
