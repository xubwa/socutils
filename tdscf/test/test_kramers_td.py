#!/usr/bin/env python
'''Time-reversal adapted two-component TDA (gks.TDA.kramers) on a Kramers-restricted closed-shell reference:
K^2 = +1 on the amplitudes, A K = K A, the TR-even solve against the full one (exact and RIJ+COSX J / K),
and the halved channels of the TR-even densities.'''
import unittest
import numpy
from pyscf import gto, lib
from socutils.dft import dft, fxc_2c
from socutils.scf import spinor_jk
from socutils.tdscf import gks


def setUpModule():
    global mol, mf
    mol = gto.M(atom='Au 0 0 0; H 0 0 1.524', basis='x2c-svpall', verbose=0)
    mf = dft.RKS(mol, xc='pbe0').x2camf()
    mf.conv_tol = 1e-11
    mf.kernel()


def tearDownModule():
    global mol, mf
    del mol, mf


class KnowValues(unittest.TestCase):
    def test_operator(self):
        td = gks.TDA(mf)
        tr = td._tr_operator()
        self.assertIsNotNone(tr)
        k, (nocc, nvir) = tr
        rng = numpy.random.default_rng(0)
        z = rng.standard_normal((2, nocc, nvir)) + 1j * rng.standard_normal((2, nocc, nvir))
        self.assertLess(abs(k(k(z)) - z).max(), 1e-10)
        vind, hdiag = td.gen_vind(mf)
        az = vind(z.reshape(2, -1)).reshape(z.shape)
        akz = vind(k(z).reshape(2, -1)).reshape(z.shape)
        self.assertLess(abs(akz - k(az)).max() / abs(az).max(), 1e-9)

    def _solve(self, kramers, rijcosx):
        mf.rijcosx = rijcosx
        td = gks.TDA(mf)
        td.nstates = 6
        td.conv_tol = 1e-7
        td.kramers = kramers
        e = td.kernel()[0]
        self.assertTrue(all(td.converged))
        return e

    def test_tr_even_vs_full(self):
        for rc in ('', 'rijcosx'):
            e0 = self._solve(False, rc)
            e1 = self._solve(True, rc)
            self.assertLess(abs(e0 - e1).max(), 1e-9, rc)
        mf.rijcosx = ''

    def test_halved_channels(self):
        td = gks.TDA(mf)
        k, (nocc, nvir) = td._tr_operator()
        rng = numpy.random.default_rng(1)
        z = rng.standard_normal((1, nocc, nvir)) + 1j * rng.standard_normal((1, nocc, nvir))
        z = (z + k(z)) * .5
        mo = mf.mo_coeff
        occ = mf.mo_occ > 0
        dm = lib.einsum('xov,pv,qo->xpq', z, mo[:, ~occ], mo[:, occ].conj())
        dmg = mf._numint._dm_to_ghf(dm)
        nao = mol.nao
        aa, ab, ba, bb = dmg[:, :nao, :nao], dmg[:, :nao, nao:], dmg[:, nao:, :nao], dmg[:, nao:, nao:]
        self.assertTrue(spinor_jk._is_kramers_even(aa, ab, ba, bb))
        self.assertEqual(len(spinor_jk._pieces(aa, ab, ba, bb, True, True, 0)), 4)
        dn, dm_ = fxc_2c.channels(dmg)
        self.assertLess(abs(dn[:, 1]).max(), 1e-10 * abs(dn).max())        # Im n = 0
        self.assertLess(abs(dm_[:, :, 0]).max(), 1e-10 * abs(dm_).max())   # Re m = 0


if __name__ == '__main__':
    unittest.main()
