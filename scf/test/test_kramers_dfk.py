'''Kramers-paired MO-based DF exchange (scf.spinor_hf._DFJHF._get_k_mo).

With occupied spinors in Kramers pairs (u, Theta u) only the unbarred half is
half-transformed: K_aa = A + B*, K_ab = C - C^T from the unbarred sums.  It
must equal the full build, and an occupied set that is not paired that way
(here: unitarily mixed) must fall back to the full build.
'''
import unittest
import numpy
from pyscf import gto
from socutils.scf import spinor_hf


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='cc-pvdz', verbose=0)
        mf = spinor_hf.KRHF(cls.mol).density_fit(auxbasis='cc-pvdz-jkfit')
        mf.verbose = 0
        mf.kernel()
        cls.mf = mf

    def _k(self, c, o, kramers):
        self.mf.kramers_k = kramers
        try:
            return self.mf._get_k_mo(self.mol, c, o)
        finally:
            self.mf.kramers_k = True

    def test_paired_equals_full(self):
        c, o = self.mf.mo_coeff, self.mf.mo_occ
        c2 = numpy.vstack(self.mol.sph2spinor_coeff())
        self.assertIsNot(spinor_hf._kramers_unbarred(c2 @ c[:, o > 0], self.mol.nao_nr()), False)
        self.assertLess(abs(self._k(c, o, True) - self._k(c, o, False)).max(), 1e-12)

    def test_unpaired_falls_back(self):
        c, o = self.mf.mo_coeff.copy(), self.mf.mo_occ
        occ = numpy.where(o > 0)[0]
        rng = numpy.random.default_rng(1)
        x = rng.normal(size=(len(occ),) * 2) + 1j * rng.normal(size=(len(occ),) * 2)
        u = numpy.linalg.qr(x)[0]
        c[:, occ] = c[:, occ] @ u               # same density, pairing destroyed
        c2 = numpy.vstack(self.mol.sph2spinor_coeff())
        self.assertIs(spinor_hf._kramers_unbarred(c2 @ c[:, occ], self.mol.nao_nr()), False)
        ref = self._k(self.mf.mo_coeff, o, False)
        self.assertLess(abs(self._k(c, o, True) - ref).max(), 1e-12)

    def test_fractional_pairs(self):
        c, o = self.mf.mo_coeff, self.mf.mo_occ.copy()
        nocc = int((o > 0).sum())
        o[nocc - 2:nocc] = .5                     # last pair half occupied
        o[nocc:nocc + 2] = .5                     # next pair half occupied
        self.assertLess(abs(self._k(c, o, True) - self._k(c, o, False)).max(), 1e-12)

    def test_casci_core_kramers(self):
        '''_CDERIS (CASCI integral transformation): Kramers-paired core
        exchange/Coulomb equal the full build.'''
        from socutils.mcscf import zcasci, zmc_ao2mo
        res = {}
        for kr in (True, False):
            self.mf.kramers_k = kr
            mc = zcasci.CASCI(self.mf, 8, 4)
            mc.verbose = 0
            mc.kernel()
            er = zmc_ao2mo._CDERIS(mc, mc.mo_coeff, level=2)
            res[kr] = (mc.e_tot, er.vk_core, er.vj_core)
        self.mf.kramers_k = True
        self.assertAlmostEqual(res[True][0], res[False][0], 11)
        self.assertLess(abs(res[True][1] - res[False][1]).max(), 1e-12)
        self.assertLess(abs(res[True][2] - res[False][2]).max(), 1e-12)


if __name__ == '__main__':
    unittest.main()
